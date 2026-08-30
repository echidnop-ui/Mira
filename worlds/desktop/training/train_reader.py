from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from language.tokenizer import Utf8ByteTokenizer
from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.dataset import _pillow_image, load_rgb
from worlds.desktop.training.train import load_checkpoint

READER_PREFIXES = ("foveal_reader.", "visual_reader_v2.", "reading_")
HOLDOUT = (
    "СИНИЙ ДРАКОН 5931", "ФИОЛЕТОВЫЙ КРОКОДИЛЬ 7319",
    "HEIWA CAN SEE THIS 4826", "BLUE TIGER WAITS AT WINDOW 314",
    "Я ВИЖУ ТВОЁ СООБЩЕНИЕ 8427",
)
REAL = {
    "heiwa": "HEIWA TEST 7319", "red_dragon": "RED DRAGON 7319",
    "blue_tiger": "BLUE TIGER 4826", "russian_heiwa": "ПРИВЕТ HEIWA 8427",
    "russian_dragon": "СИНИЙ ДРАКОН 5931",
}
EN = ("HEIWA", "BLUE", "TIGER", "DRAGON", "WINDOW", "BUTTON", "MESSAGE", "OPEN", "SAVE", "READY", "WAITS", "SCREEN", "CAN", "SEE", "THIS", "NEW", "INPUT", "PURPLE", "CROCODILE", "AT")
RU = ("ПРИВЕТ", "СИНИЙ", "ДРАКОН", "ОКНО", "КНОПКА", "СООБЩЕНИЕ", "ВИЖУ", "ТЕБЯ", "НОВЫЙ", "ГОТОВО", "ТЕСТ", "ЖДЁТ", "ФИОЛЕТОВЫЙ", "КРОКОДИЛЬ", "ТВОЁ", "Я")
FONTS = tuple(path for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
) if Path(path).exists())


def _text(seed: int, category: str, stage: int) -> str:
    rng = random.Random(seed)
    if category == "digits":
        return f"{rng.randrange(1,29):02d}.{rng.randrange(1,13):02d}.{rng.randrange(2020,2035)} {rng.randrange(1000,10000)}"
    words = EN if category == "en" else RU
    if category == "mixed":
        words = EN + RU
    count = 1 if stage == 1 else rng.randint(2, 4 if stage < 4 else 5)
    result = " ".join(rng.choice(words) for _ in range(count))
    if stage >= 2 and rng.random() < .35: result = result.lower()
    if stage >= 3 and rng.random() < .4: result += rng.choice(("!", "?", ": OK", f" {rng.randrange(10,9999)}"))
    while len(result.encode("utf-8")) > 63:
        result = " ".join(result.split()[:-1])
    return result


def _render(text: str, seed: int, stage: int) -> tuple[Tensor, Tensor]:
    Image = _pillow_image(); from PIL import ImageDraw, ImageFont, ImageFilter
    rng = random.Random(seed); w, h = 256, 64
    dark = stage >= 2 and rng.random() < .5
    bg = rng.randint(5, 35) if dark else rng.randint(220, 252)
    fg = rng.randint(220, 255) if dark else rng.randint(0, 35)
    image = Image.new("RGB", (w, h), (bg, bg + rng.randint(-3, 3), bg + rng.randint(-3, 3)))
    draw = ImageDraw.Draw(image)
    if stage >= 3:
        bar = bg + (15 if dark else -15); draw.rectangle((0, 0, w, 10), fill=(bar, bar, bar))
        if rng.random() < .5: draw.rounded_rectangle((3, 12, w-4, h-3), radius=4, outline=(fg, fg, fg), width=1)
    font_path = FONTS[0] if stage == 1 else rng.choice(FONTS)
    max_size = 42 if stage == 1 else rng.randint(25, 43)
    font = ImageFont.truetype(font_path, max_size)
    box = draw.textbbox((0, 0), text, font=font)
    while box[2]-box[0] > w-12 and max_size > 10:
        max_size -= 1; font = ImageFont.truetype(font_path, max_size); box = draw.textbbox((0, 0), text, font=font)
    x = (w-(box[2]-box[0]))//2 + (rng.randint(-3,3) if stage >= 2 else 0)
    y = (h-(box[3]-box[1]))//2-box[1] + (rng.randint(-2,2) if stage >= 2 else 0)
    draw.text((x,y), text, font=font, fill=(fg,fg,fg))
    alignment = torch.zeros(128, dtype=torch.long)
    tokenizer = Utf8ByteTokenizer()
    for char_index, char in enumerate(text):
        left = x + float(draw.textlength(text[:char_index], font=font))
        right = x + float(draw.textlength(text[:char_index + 1], font=font))
        ids = tokenizer.encode(char)
        for byte_index, token in enumerate(ids):
            start = round((left + byte_index * (right-left) / len(ids)) / 2) + 1
            end = round((left + (byte_index+1) * (right-left) / len(ids)) / 2) - 1
            start=min(126,max(1,start)); end=min(126,max(start,end))
            alignment[start:end+1] = token
    eos_column = min(127, max(0, round((x + float(draw.textlength(text, font=font))) / 2) + 1))
    while eos_column < 127 and alignment[eos_column] != 0: eos_column += 1
    alignment[eos_column] = tokenizer.eos_id
    if stage >= 3 and rng.random() < .3: image = image.filter(ImageFilter.GaussianBlur(rng.choice((.25,.4,.6))))
    raw = bytearray(image.tobytes())
    crop=torch.frombuffer(raw, dtype=torch.uint8).reshape(h,w,3).permute(2,0,1).clone()
    return crop, alignment


class ReaderDataset(Dataset):
    def __init__(self, count: int, seed: int, stage: int, *, holdout: bool=False):
        self.count=count; self.seed=seed; self.stage=stage; self.holdout=holdout
    def __len__(self): return self.count
    def __getitem__(self, i):
        categories = ("en",) if self.stage <= 2 else (("en","digits") if self.stage == 3 else ("en","ru","mixed","digits"))
        category=categories[i%len(categories)]
        text = HOLDOUT[i] if self.holdout and i < len(HOLDOUT) else _text(self.seed+i*104729, category, self.stage)
        crop,alignment=_render(text,self.seed+i*13007,self.stage)
        return {"crop":crop,"alignment":alignment,"text":text,"category":category}


class ActiveTabDataset(Dataset):
    """Controlled pixel-only chat/forum scenes with generator-known target boxes."""
    def __init__(self, count: int, seed: int, *, holdout: bool = False):
        self.count = count; self.seed = seed; self.holdout = holdout
    def __len__(self): return self.count
    def __getitem__(self, i):
        Image = _pillow_image(); from PIL import ImageDraw
        seed = self.seed + i * 104729; rng = random.Random(seed)
        dark = rng.random() < .5; bg = rng.randint(12, 38) if dark else rng.randint(225, 250)
        image = Image.new("RGB", (512, 288), (bg, bg, bg)); draw = ImageDraw.Draw(image)
        # Browser chrome is visual context, never metadata used by the model.
        chrome = bg + (18 if dark else -18)
        draw.rectangle((0, 0, 511, rng.randint(28, 45)), fill=(chrome, chrome, chrome))
        for tab in range(rng.randint(2, 5)):
            x = 8 + tab * rng.randint(80, 115)
            draw.rounded_rectangle((x, 6, min(505, x + rng.randint(60, 100)), 26), 5,
                                   outline=(90, 90, 90), width=1)
        # Distractor messages precede the newest visible message.
        y = rng.randint(48, 72)
        for j in range(rng.randint(1, 3)):
            text = _text(seed + j * 7919, ("en", "ru", "mixed")[j % 3], 4)
            crop, _ = _render(text, seed + j * 3571, 4)
            tile = Image.fromarray(crop.permute(1, 2, 0).numpy())
            tw = rng.randint(210, 390); th = rng.randint(42, 64)
            x = rng.randint(12, 512 - tw - 12)
            image.paste(tile.resize((tw, th)), (x, y)); y += th + rng.randint(8, 18)
        category = ("en", "ru", "mixed", "digits")[i % 4]
        text = HOLDOUT[i % len(HOLDOUT)] if self.holdout and i < len(HOLDOUT) else _text(seed + 99991, category, 4)
        crop, _ = _render(text, seed + 11213, 4)
        tile = Image.fromarray(crop.permute(1, 2, 0).numpy())
        tw = rng.randint(240, 460); th = rng.randint(48, 78)
        y = min(rng.randint(max(55, y), max(56, 288 - th - 10)), 288 - th - 6) if y < 288-th-10 else 288-th-6
        x = rng.randint(8, 512 - tw - 8)
        image.paste(tile.resize((tw, th)), (x, y))
        raw = bytearray(image.tobytes())
        frame = torch.frombuffer(raw, dtype=torch.uint8).reshape(288,512,3).permute(2,0,1).clone()
        bbox = torch.tensor((x/512, y/288, (x+tw)/512, (y+th)/288), dtype=torch.float32)
        return {"frame": frame, "bbox": bbox, "text": text, "category": category}


def _targets(texts: list[str], config: NeuralDesktopConfig, device: torch.device) -> Tensor:
    tok=Utf8ByteTokenizer(); out=torch.zeros((len(texts),config.max_read_bytes+2),dtype=torch.long,device=device)
    for i,text in enumerate(texts):
        ids=tok.encode(text,add_eos=True)
        if len(ids)>out.shape[1]: raise ValueError(f"text exceeds reader budget: {text!r}")
        out[i,:len(ids)]=torch.tensor(ids,device=device)
    return out


def _logits(model: NeuralDesktopBody, crops: Tensor, targets: Tensor|None=None) -> Tensor:
    boxes=torch.tensor((0.,0.,1.,1.),device=crops.device).expand(crops.shape[0],-1)
    return model.read_region(crops.float().div(255),boxes)


def _raw_logits(model: NeuralDesktopBody, crops: Tensor, targets: Tensor | None = None) -> Tensor:
    boxes=torch.tensor((0.,0.,1.,1.),device=crops.device).expand(crops.shape[0],-1)
    return model.read_region_raw(crops.float().div(255),boxes,targets)


def _ctc_loss(raw: Tensor, targets: Tensor) -> Tensor:
    lengths=targets.ne(0).sum(-1)
    flat=torch.cat([row[:length] for row,length in zip(targets,lengths.tolist())])
    inputs=torch.full_like(lengths,raw.shape[1])
    return F.ctc_loss(raw.float().log_softmax(-1).transpose(0,1),flat,inputs,lengths,
                      blank=0,zero_infinity=True)


def _edit(a: str,b: str)->int:
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]


@torch.no_grad()
def evaluate(model, loader, device):
    tok=Utf8ByteTokenizer(); model.eval(); rows=[]; loss_sum=0.; batches=0
    for batch in loader:
        crops=batch["crop"].to(device); texts=list(batch["text"]); targets=_targets(texts,model.config,device)
        raw=_raw_logits(model,crops); logits=model.visual_reader_v2.collapse(raw); loss_sum+=float(_ctc_loss(raw,targets)); batches+=1
        probs=logits.float().softmax(-1); pred=probs.argmax(-1)
        for i,text in enumerate(texts):
            target=tok.encode(text,add_eos=True); ids=pred[i].tolist(); eos=tok.eos_id in ids; end=ids.index(tok.eos_id)+1 if eos else len(ids)
            raw=bytes(x-tok.BYTE_OFFSET for x in ids[:end] if x>=tok.BYTE_OFFSET)
            try: decoded=raw.decode("utf-8"); valid=True
            except UnicodeDecodeError: decoded=raw.decode("utf-8",errors="replace"); valid=False
            n=len(target); byte_acc=sum(x==y for x,y in zip(ids[:n],target))/n
            rows.append({"expected":text,"predicted":decoded,"exact":decoded==text and eos,"cer":_edit(decoded,text)/max(1,len(text)),
                         "byte_accuracy":byte_acc,"eos":eos and ids[len(target)-1]==tok.eos_id,"valid_utf8":valid,
                         "confidence":float(probs[i,:end].max(-1).values.mean()),"category":batch["category"][i]})
    def avg(key, rs=rows): return sum(float(x[key]) for x in rs)/max(1,len(rs))
    result={"loss":loss_sum/max(1,batches),"samples":len(rows),"exact_string_accuracy":avg("exact"),"cer":avg("cer"),
            "byte_accuracy":avg("byte_accuracy"),"eos_accuracy":avg("eos"),"invalid_utf8_rate":1-avg("valid_utf8"),"sequence_confidence":avg("confidence")}
    result["breakdown"]={c:{"samples":len(rs),"exact":avg("exact",rs),"cer":avg("cer",rs),"byte_accuracy":avg("byte_accuracy",rs)} for c in ("en","ru","mixed","digits") if (rs:=[x for x in rows if x["category"]==c])}
    return result,rows


def _box_iou(a: Tensor, b: Tensor) -> Tensor:
    lt=torch.maximum(a[:,:2],b[:,:2]); rb=torch.minimum(a[:,2:],b[:,2:])
    inter=(rb-lt).clamp_min(0).prod(-1)
    area_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1); area_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return inter/(area_a+area_b-inter+1e-6)


@torch.no_grad()
def evaluate_focus(model, loader, device):
    model.eval(); centers=[]; ious=[]
    for batch in loader:
        frames=batch["frame"].to(device).float().div(255); expected=batch["bbox"].to(device)
        predicted,_=model.locate_reading_region(model.observe(frames))
        centers.extend(((predicted[:,:2]+predicted[:,2:]-expected[:,:2]-expected[:,2:])/2).norm(dim=-1).cpu().tolist())
        ious.extend(_box_iou(predicted,expected).cpu().tolist())
    centers.sort(); ious.sort()
    return {"samples":len(centers),"median_center_error_normalized":centers[len(centers)//2],
            "mean_iou":sum(ious)/len(ious),"median_iou":ious[len(ious)//2],
            "iou_p10":ious[max(0,len(ious)//10-1)]}


def _hash_state(state, prefixes, invert=False):
    h=hashlib.sha256()
    for k in sorted(state):
        if (k.startswith(prefixes)) != invert: h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description="Isolated RGB-to-UTF8 reader training")
    p.add_argument("--checkpoint",type=Path,default=Path("state/neural_desktop.pt")); p.add_argument("--output",type=Path,default=Path("state/reader_training"))
    p.add_argument("--diagnostics",type=Path,default=Path("state/vision_diagnostics/20260829-205522")); p.add_argument("--device",default="auto")
    p.add_argument("--train-samples",type=int,default=20000); p.add_argument("--val-samples",type=int,default=1005); p.add_argument("--epochs-per-stage",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--focus-batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--baseline",action="store_true"); p.add_argument("--save-on-pass",action="store_true"); p.add_argument("--save-checkpoint",type=Path); p.add_argument("--progress-checkpoint",type=Path)
    p.add_argument("--start-stage",type=int,choices=(1,2,3,4),default=1)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); torch.manual_seed(7319); device=select_device(a.device)
    model=NeuralDesktopBody(NeuralDesktopConfig()).to(device); load_checkpoint(a.checkpoint,model); before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    for name,param in model.named_parameters(): param.requires_grad=name.startswith(READER_PREFIXES)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    # Recurrent CTC alignment is deliberately full fp32 on ROCm: fp16 recurrent
    # activations caused repeatable divergence even when the loss was promoted.
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=.01); amp=False; scaler=torch.amp.GradScaler("cuda",enabled=False)
    start=time.monotonic(); steps=0; history=[]
    per=max(1,a.train_samples//4)
    for stage in range(a.start_stage,5):
        loader=DataLoader(ReaderDataset(per,7319+stage*100000,stage),batch_size=a.batch_size,shuffle=True,num_workers=0)
        focus_loader=iter(DataLoader(ActiveTabDataset(max(per,64),8319+stage*100000),batch_size=a.focus_batch_size,shuffle=True,num_workers=0))
        for epoch in range(a.epochs_per_stage):
            model.train(); total=0.; n=0
            for batch in loader:
                crops=batch["crop"].to(device); targets=_targets(list(batch["text"]),model.config,device); opt.zero_grad(set_to_none=True)
                ctx=torch.autocast("cuda",dtype=torch.float16) if amp else nullcontext()
                with ctx:
                    logits=_raw_logits(model,crops,targets); reader_loss=_ctc_loss(logits,targets)
                    alignment=batch["alignment"].to(device)
                    alignment_weights=torch.ones(model.config.byte_vocab_size,device=device); alignment_weights[0]=.08
                    alignment_loss=F.cross_entropy(logits.float().flatten(0,1),alignment.flatten(),weight=alignment_weights)
                    reader_loss=reader_loss+.35*alignment_loss
                    try: focus_batch=next(focus_loader)
                    except StopIteration:
                        focus_loader=iter(DataLoader(ActiveTabDataset(max(per,64),8319+stage*100000+epoch),batch_size=a.focus_batch_size,shuffle=True,num_workers=0)); focus_batch=next(focus_loader)
                    frames=focus_batch["frame"].to(device).float().div(255); expected=focus_batch["bbox"].to(device)
                    with torch.no_grad(): visual=model.observe(frames)
                    predicted,confidence=model.locate_reading_region(visual)
                    focus_loss=F.smooth_l1_loss(predicted,expected)+(1-_box_iou(predicted,expected)).mean()+F.binary_cross_entropy(confidence,torch.ones_like(confidence))*.1
                    loss=reader_loss+focus_loss
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0); scaler.step(opt); scaler.update()
                total+=float(loss.detach()); n+=1; steps+=1
            history.append({"stage":stage,"epoch":epoch+1,"loss":total/n}); print(history[-1],flush=True)
            progress={"running":True,"stage":stage,"epoch":epoch+1,"epochs_per_stage":a.epochs_per_stage,
                      "loss":total/n,"steps":steps,"elapsed_seconds":round(time.monotonic()-start,1)}
            progress_path=a.output/"progress.json"; temporary=progress_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(progress,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); temporary.replace(progress_path)
            # Developmental learning is part of life, so a service restart must
            # not send the reader back to the beginning. Persist reader progress
            # after every epoch; this is NOT adopted by the live body until the
            # validation gate passes.
            if a.progress_checkpoint is not None:
                a.progress_checkpoint.parent.mkdir(parents=True,exist_ok=True)
                base=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
                base["model"]={k:v.detach().cpu() for k,v in model.state_dict().items()}
                base.pop("optimizer",None); base.pop("online_optimizer",None)
                tmp=a.progress_checkpoint.with_suffix(a.progress_checkpoint.suffix+".tmp")
                torch.save(base,tmp); tmp.replace(a.progress_checkpoint)
        if a.baseline: break
    val_loader=DataLoader(ReaderDataset(a.val_samples,900000001,4,holdout=True),batch_size=a.batch_size,num_workers=0)
    metrics,examples=evaluate(model,val_loader,device)
    focus_metrics=evaluate_focus(model,DataLoader(ActiveTabDataset(max(100,a.val_samples//4),910000001,holdout=True),batch_size=a.focus_batch_size),device)
    # Real saved PipeWire crops, never OCR/accessibility.
    real=[]
    for name,text in REAL.items():
        crop=load_rgb(a.diagnostics/name/"controlled_text_crop.png",width=256,height=64).mul(255).byte()
        class One(Dataset):
            def __len__(self): return 1
            def __getitem__(self,i): return {"crop":crop,"text":text,"category":"mixed" if any(ord(c)>127 for c in text) else "en"}
        _,row=evaluate(model,DataLoader(One(),batch_size=1),device); real.append({"name":name,**row[0]})
    after=model.state_dict(); frozen_ok=all(torch.allclose(before[k],v.detach().cpu(),rtol=0,atol=0,equal_nan=True) for k,v in after.items() if not k.startswith(READER_PREFIXES))
    passed=metrics["exact_string_accuracy"]>.95 and metrics["cer"]<.02 and metrics["eos_accuracy"]>.99 and metrics["invalid_utf8_rate"]<.005 and focus_metrics["median_center_error_normalized"]<.05 and focus_metrics["median_iou"]>.5
    report={"passed":passed,"baseline":a.baseline,"device":str(device),"trainable_parameters":trainable,"frozen_parameters":frozen,"dataset":{"train":per*(1 if a.baseline else 4),"validation":a.val_samples},"steps":steps,"seconds":time.monotonic()-start,"history":history,"validation":metrics,"focus_validation":focus_metrics,"examples":examples[:20],"real":real,"frozen_unchanged":frozen_ok,"frozen_sha_before":_hash_state(before,READER_PREFIXES,True),"frozen_sha_after":_hash_state(after,READER_PREFIXES,True)}
    (a.output/("baseline_report.json" if a.baseline else "report.json")).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.output/"progress.json").write_text(json.dumps({"running":False,"passed":passed,"validation":metrics,
                                                       "focus_validation":focus_metrics,"elapsed_seconds":round(time.monotonic()-start,1)},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if passed and a.save_on_pass:
        destination=a.save_checkpoint or a.checkpoint
        destination.parent.mkdir(parents=True,exist_ok=True)
        payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False); payload["model"]={k:v.detach().cpu() for k,v in after.items()}; payload.pop("optimizer",None)
        tmp=destination.with_suffix(destination.suffix+".tmp"); torch.save(payload,tmp); tmp.replace(destination)
    print(json.dumps({"passed":passed,"validation":metrics,"real_exact":sum(x["exact"] for x in real)/len(real),"frozen_unchanged":frozen_ok,"report":str(a.output/("baseline_report.json" if a.baseline else "report.json"))},ensure_ascii=False),flush=True)

if __name__=="__main__": main()

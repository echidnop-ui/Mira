from __future__ import annotations
import argparse, random
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopBody, NeuralDesktopConfig, select_device
from worlds.desktop.training.dataset import DesktopDemonstrationDataset

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('manifest',type=Path); ap.add_argument('--checkpoint',type=Path,default=Path('state/neural_desktop.pt'))
    ap.add_argument('--steps',type=int,default=600); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--device',default='auto')
    a=ap.parse_args(); torch.manual_seed(29); random.seed(29)
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=NeuralDesktopConfig(**source['config']); dev=select_device(a.device)
    model=NeuralDesktopBody(cfg).to(dev); model.load_state_dict(source['model'],strict=False)
    ds=DesktopDemonstrationDataset(a.manifest,cfg); idx,_=ds.split_indices(); counts=Counter(str(ds.records[i]['action']) for i in idx)
    weights=[1.0/counts[str(ds.records[i]['action'])] for i in idx]
    sampler=WeightedRandomSampler(weights,num_samples=max(len(idx),a.steps*a.batch_size),replacement=True)
    loader=DataLoader(Subset(ds,idx),batch_size=a.batch_size,sampler=sampler,num_workers=0)
    for p in model.parameters(): p.requires_grad_(False)
    modules=(model.motor_fusion,model.action_head,model.target_query,model.target_key,model.offset_head,model.key_head)
    for m in modules:
        for p in m.parameters(): p.requires_grad_(True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    model.train(); it=iter(loader); running=0.0
    for step in range(1,a.steps+1):
        try: b=next(it)
        except StopIteration: it=iter(loader); b=next(it)
        before=b['before'].to(dev); action=b['action'].to(dev); xy=b['target_xy'].to(dev); cell=b['target_cell'].to(dev); params=b['parameters'].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.no_grad(): visual=model.observe(before)
        out=model.propose(visual); loss=F.cross_entropy(out.action_logits,action.long())
        spatial=cell.ge(0)
        if bool(spatial.any()):
            # Relearn where human actions actually occurred, not where Heiwa's
            # previous mistaken action happened to change the screen.
            loss=loss+F.cross_entropy(out.target_logits[spatial],cell[spatial].long())
            loss=loss+F.smooth_l1_loss(out.target_point[spatial],xy[spatial])
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); running+=float(loss.detach())
        if step==1 or step%50==0 or step==a.steps:
            acc=float(out.action_logits.detach().argmax(-1).eq(action).float().mean())
            print({'step':step,'loss':round(running/(1 if step==1 else 50),5),'batch_action_accuracy':round(acc,3)},flush=True); running=0.0
    source['model']=model.state_dict(); source.pop('online_optimizer',None); source.pop('optimizer',None)
    source['motor_repaired']=True; source['motor_repair_version']=4
    tmp=a.checkpoint.with_suffix(a.checkpoint.suffix+'.v4.tmp'); torch.save(source,tmp); tmp.replace(a.checkpoint)
    print({'ok':True,'checkpoint':str(a.checkpoint),'actions':dict(counts),'motor_repair_version':4},flush=True); return 0
if __name__=='__main__': raise SystemExit(main())

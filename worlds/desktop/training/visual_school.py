from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time


PHRASES = (
    "Привет! Что ты видишь на экране?",
    "Напиши короткий ответ в поле ниже.",
    "BLUE TIGER WAITS AT WINDOW 314",
    "СИНИЙ ДРАКОН 5931",
    "Какой следующий шаг имеет смысл?",
    "Ответь одним коротким предложением.",
    "HEIWA CAN READ PIXELS 4826",
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _presenter(state_path: Path, event_path: Path) -> int:
    """Draw controlled pixels. Geometry is supervision, never inference input."""
    import gi
    gi.require_version("Gtk", "3.0"); gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GLib, Gtk

    rng = random.Random(8427)
    window = Gtk.Window(title="Heiwa neural visual school")
    window.set_decorated(False)
    fixed = Gtk.Fixed(); window.add(fixed)
    message = Gtk.Label(); message.set_line_wrap(True); message.set_xalign(0.0); message.set_yalign(0.5)
    composer = Gtk.Entry(); composer.set_placeholder_text("Введите ответ…")
    send = Gtk.Button(label="Отправить")
    status = Gtk.Label(label="F21 текст верно   F22 текст неверно   F23 target неверно   F24 target принят")
    for widget in (message, composer, send, status): fixed.put(widget, 0, 0)
    scene: dict[str, object] = {"id": 0, "message_text": PHRASES[0], "style": 0}

    css = Gtk.CssProvider()
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

    def normalized(widget) -> list[float]:
        allocation = widget.get_allocation(); width=max(1,fixed.get_allocated_width()); height=max(1,fixed.get_allocated_height())
        return [allocation.x/width, allocation.y/height,
                (allocation.x+allocation.width)/width, (allocation.y+allocation.height)/height]

    def publish() -> bool:
        payload = {**scene, "message_bbox": normalized(message),
                   "composer_bbox": normalized(composer), "send_bbox": normalized(send),
                   "window_size": [fixed.get_allocated_width(), fixed.get_allocated_height()],
                   "rendered_monotonic_ns": time.monotonic_ns(),
                   "typed_length": len(composer.get_text()), "typed_text_recorded": False,
                   "pixel_only": True, "atspi": False, "ocr": False}
        _atomic_json(state_path, payload); return False

    def place() -> bool:
        width=max(640,fixed.get_allocated_width()); height=max(420,fixed.get_allocated_height())
        message_width=rng.randint(int(width*.42),int(width*.72)); message_height=rng.randint(70,130)
        mx=rng.randint(25,max(26,width-message_width-25)); my=rng.randint(65,max(66,int(height*.48)))
        composer_width=rng.randint(int(width*.43),int(width*.68)); composer_height=rng.randint(46,62)
        cx=rng.randint(20,max(21,width-composer_width-150)); cy=rng.randint(int(height*.68),max(int(height*.68)+1,height-composer_height-55))
        fixed.move(message,mx,my); message.set_size_request(message_width,message_height)
        fixed.move(composer,cx,cy); composer.set_size_request(composer_width,composer_height)
        fixed.move(send,min(width-130,cx+composer_width+10),cy); send.set_size_request(120,composer_height)
        fixed.move(status,20,height-34)
        dark=bool(int(scene["style"])%2); bg="#111318" if dark else "#edf1f6"; fg="#f7fbff" if dark else "#15181e"
        accent=("#ff3b8d","#32d6ff","#ffe14d","#72ff72")[int(scene["style"])%4]
        css.load_from_data((f"window, fixed {{background:{bg};}} label {{color:{fg};font:22px 'DejaVu Sans';}} "
                            f"entry {{font:22px 'DejaVu Sans';border:3px solid {accent};border-radius:14px;padding:8px;}} "
                            f"button {{font:18px 'DejaVu Sans';background:{accent};color:#101116;border-radius:14px;}}").encode())
        GLib.timeout_add(80,publish); return False

    def log_event(role: str, action: str, target_bbox: list[float], key_code: int | None = None) -> None:
        center=[(target_bbox[0]+target_bbox[2])/2,(target_bbox[1]+target_bbox[3])/2]
        record={"timestamp_ns":time.monotonic_ns(),"role":role,"action":action,
                "target_xy":center,"target_bbox":target_bbox,"key_code":key_code,
                "scene_id":scene["id"],"typed_text_recorded":False}
        event_path.parent.mkdir(parents=True,exist_ok=True)
        with event_path.open("a",encoding="utf-8") as stream: stream.write(json.dumps(record,ensure_ascii=False)+"\n")

    def composer_press(widget, event):
        log_event("composer","left_click",normalized(composer)); return False

    def key_press(widget, event):
        # GDK hardware keycodes are XKB codes; Linux evdev physical code is offset by 8.
        log_event("composer","key_press",normalized(composer),max(0,int(event.hardware_keycode)-8)); GLib.idle_add(publish); return False

    def next_scene() -> bool:
        scene["id"]=int(scene["id"])+1; scene["message_text"]=rng.choice(PHRASES); scene["style"]=rng.randrange(8)
        message.set_text(str(scene["message_text"])); composer.set_text(""); place(); return False

    def sent(*_):
        log_event("send","left_click",normalized(send)); GLib.timeout_add(500,next_scene)

    composer.connect("button-press-event",composer_press); composer.connect("key-press-event",key_press)
    composer.connect("activate",sent); send.connect("clicked",sent); window.connect("destroy",Gtk.main_quit)
    message.set_text(str(scene["message_text"])); window.maximize(); window.present(); window.show_all()
    GLib.timeout_add(250,place); Gtk.main(); return 0


def _fresh_state(project: Path) -> None:
    """Destructively replace all learned state with architecture-compatible initialization."""
    subprocess.run(("systemctl","--user","stop","heiwa.service"),check=False)
    state=project/"state"
    # Static curriculum, runtime configuration and diagnostic images are test
    # inputs/mechanism, not learned agent state. Everything below is learned.
    for directory in (state/"embodied_runtime",state/"desktop_demonstrations",
                      state/"visual_school",state/"reader_training",state/"reader_training_smoke",
                      state/"grounding_training",state/"grounding_training_smoke",
                      state/"visual_alphabet",
                      state/"language_developmental",state/"language_natural_teacher",
                      state/"patch_backups"):
        if directory.exists():
            for path in sorted(directory.rglob("*"),reverse=True):
                if path.is_file() or path.is_symlink(): path.unlink(missing_ok=True)
                elif path.is_dir(): path.rmdir()
            directory.rmdir()
    # Checkpoints/progress/replay can also be nested or acquire a new filename.
    # Learned tensor state is never a static asset, so remove it throughout
    # state/ instead of maintaining a fragile list of checkpoint basenames.
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        for path in state.rglob(pattern):
            path.unlink(missing_ok=True)
    for path in (state/"language_curriculum"/"developmental_corrections.jsonl",
                 state/"language_curriculum"/"latest.json"):
        path.unlink(missing_ok=True)
    from core.personal import PersonalCognitiveCore
    from language.organ import NeuralLanguageConfig, NeuralLanguageOrgan
    from policy.dynamic import DynamicAffordancePolicy
    from worlds.desktop.neural import NeuralDesktopBody, NeuralDesktopConfig
    import torch
    runtime=state/"embodied_runtime"; runtime.mkdir(parents=True,exist_ok=True)
    core=PersonalCognitiveCore(); core.save_checkpoint(runtime/"core.pt")
    policy=DynamicAffordancePolicy(); policy.save_checkpoint(runtime/"policy.pt")
    language=NeuralLanguageOrgan(NeuralLanguageConfig.local()); language.save_checkpoint(state/"language_developmental_dialogue.v5-byte.pt",training_metadata={"fresh_reset":True})
    config=NeuralDesktopConfig(); model=NeuralDesktopBody(config)
    torch.save({"format":1,"config":{name:getattr(config,name) for name in config.__dataclass_fields__},
                "model":model.state_dict(),"step":0,"fresh_reset":True},state/"neural_desktop.pt")
    print(json.dumps({"reset":True,"backup":None,"core":"fresh","policy":"fresh","language":"fresh","desktop":"fresh"}),flush=True)


def _wait_state(path: Path, timeout: float=10.0) -> None:
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if path.exists(): return
        time.sleep(.1)
    raise RuntimeError("visual-school window did not become ready")


def _desktop_training_step(path: Path) -> int:
    if not path.exists(): return 0
    import torch
    payload=torch.load(path,map_location="cpu",weights_only=False)
    return max(0,int(payload.get("step",0))) if isinstance(payload,dict) else 0


def main() -> int:
    parser=argparse.ArgumentParser(description="Pixel-only interactive Desktop visual/motor school")
    parser.add_argument("--output",type=Path,default=Path("state/visual_school"))
    parser.add_argument("--seconds",type=int,default=120)
    parser.add_argument("--autonomous",action="store_true",help="run Heiwa while F23/F24 and pointer corrections teach it")
    parser.add_argument("--reset",action="store_true",help="destructively initialize all learned state; no backup")
    parser.add_argument("--no-train",action="store_true")
    parser.add_argument("--presenter",nargs=2,metavar=("STATE","EVENTS"))
    args=parser.parse_args()
    if args.presenter: return _presenter(Path(args.presenter[0]),Path(args.presenter[1]))
    if args.seconds<10: parser.error("--seconds must be >= 10")
    project=Path(__file__).resolve().parents[3]
    if args.reset: _fresh_state(project)
    args.output.mkdir(parents=True,exist_ok=True); state=args.output/"scene.json"; events=args.output/"interactions.jsonl"
    state.unlink(missing_ok=True)
    presenter=subprocess.Popen(("/usr/bin/python3","-m","worlds.desktop.training.visual_school","--presenter",str(state),str(events)),cwd=project)
    recorder=None; journal=None; service_started=False
    deferred_autonomy=False
    try:
        _wait_state(state)
        command=(sys.executable,"-m","worlds.desktop.training.record_demonstration","--output",str(args.output/"demonstrations"),
                 "--capture-input","--interval","0.25","--pixel-supervision-state",str(state),
                 "--controlled-events",str(events))
        recorder=subprocess.Popen(command,cwd=project)
        time.sleep(1.5)
        if deferred_autonomy:
            print("AUTONOMY DEFERRED: fresh motor head; this first session is user bootstrap only.",flush=True)
        elif args.autonomous:
            subprocess.run(("systemctl","--user","start","heiwa.service"),check=True); service_started=True
            journal=subprocess.Popen(("journalctl","--user","-u","heiwa.service","-f","-n","0","-o","cat"))
        print("VISUAL SCHOOL: F21/F22=reading correct/wrong; F23/F24=cursor reject/accept; Ctrl+C stops and trains.",flush=True)
        deadline=time.monotonic()+args.seconds
        while time.monotonic()<deadline and recorder.poll() is None and presenter.poll() is None: time.sleep(.2)
    except KeyboardInterrupt: pass
    finally:
        if recorder is not None and recorder.poll() is None:
            recorder.send_signal(signal.SIGINT)
            try: recorder.wait(timeout=15)
            except subprocess.TimeoutExpired: recorder.terminate(); recorder.wait(timeout=5)
        if presenter.poll() is None: presenter.terminate(); presenter.wait(timeout=5)
        if journal is not None and journal.poll() is None: journal.terminate(); journal.wait(timeout=5)
    manifest=args.output/"demonstrations"/"manifest.jsonl"
    if not args.no_train and manifest.exists() and manifest.stat().st_size:
        if service_started: subprocess.run(("systemctl","--user","stop","heiwa.service"),check=False)
        subprocess.run((sys.executable,"-m","worlds.desktop.training.train",str(manifest),"--checkpoint","state/neural_desktop.pt",
                        "--epochs","3","--batch-size","8","--accumulate","1","--lr","0.0003","--resume","--heads-only"),cwd=project,check=True)
        if service_started: subprocess.run(("systemctl","--user","start","heiwa.service"),check=False)
    print(json.dumps({"ok":True,"manifest":str(manifest),"trained":not args.no_train,
                      "autonomous":args.autonomous and not deferred_autonomy,
                      "autonomy_deferred_until_next_run":deferred_autonomy}),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())

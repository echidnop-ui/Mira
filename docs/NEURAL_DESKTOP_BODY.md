# Neural Desktop Body

## Causal path

`screen pixels → coarse CNN/Transformer → visual grounding point+bbox → target-conditioned neural motor policy → neural trajectory → ydotool/uinput → new pixels → causal completion/consequence learning`

Normal inference is `worlds.desktop.inference.NeuralDesktopInference`. It consumes only pixels, optional task context, previous visual state and the last motor action. The grounding head emits a normalized target point, bbox and confidence. The selected visual token conditions action, physical-key and trajectory heads. `NeuralDesktopWorldAdapter` exposes ordinary dynamic affordances and uses that neural target; the World never substitutes application coordinates.

The physical `DesktopMotor` remains deterministic peripheral machinery. A successful ydotool call is only an `ActionAttempt`. Before actuation the adapter stores the transition id and predicted next visual latent. The next pixel observation closes that exact transition; direct patch-feature change, prediction error and local consequence-head losses produce `CompletedAction` and `Consequence`. No global scalar reward is the central visual-control target.

## What is neural now

- compact trainable convolutional patch encoder and spatial Transformer;
- learned region queries with positions, salience and temporal persistence;
- action-conditioned GRU visual state;
- target point/bbox/confidence grounding heads;
- action, physical-key and curved trajectory heads conditioned on the grounded visual token;
- one selected high-resolution crop reader with the Language v5 UTF-8 byte vocabulary (PAD/BOS/EOS/UNK 0..3, bytes 4..259);
- next-latent, changed-region, temporal-consistency and feature-prediction losses;
- point, bbox, trajectory, action, key and spatial imitation losses from demonstrations;
- balanced action-type imitation and confidence calibration;

Default input is 512×288. The default model has 192-wide representations, two Transformer layers, six attention heads and twelve region tokens. ROCm is selected through PyTorch's CUDA-compatible device API when available; CPU remains usable.

## Bootstrap and legacy code

`worlds/desktop/teacher/` is optional offline labelling code. AT-SPI labels must never be passed to normal inference. Existing `perception.py`, `persistent_perception.py`, `chatgpt_visual.py`, `visual_screen.py` heuristics and the original `DesktopWorldAdapter` remain for regression, teacher-data collection and diagnostics. They are not the neural body's production decision path.

The model ships untrained. It does not yet reliably understand controls or safely operate arbitrary applications. The confidence/rate/emergency-stop gate is mechanical containment, not cognition. F12 stops execution when `/dev/input` is readable; creating `state/desktop.EMERGENCY_STOP` is the permission-independent stop mechanism.

## Unified autonomous runtime

The normal continuously living entry point is:

```bash
.venv-rocm/bin/python -m runtime.embodied
```

It loads `state/runtime_config.json` and continuously runs pixel observation,
Core/Policy choice, physical execution, causal after-frame observation,
FAST/WORK/SLOW neural updates, balanced replay, and atomic persistence. The
current autonomy configuration enables mouse, clicks, scroll, keyboard and
typing without per-launch permission flags. User evdev activity is external
action evidence, never implicit feedback. The live model is not frozen after
loading its checkpoint.

`systemd/heiwa.service` is the production launcher. Install/enable it with
`tools/install_heiwa_service.sh`; inspect or stop it with
`systemctl --user status heiwa` and `systemctl --user stop heiwa`.

## Data and training

Record compact downsampled WebP transitions:

```bash
.venv-rocm/bin/python -m worlds.desktop.training.record_demonstration --output state/desktop_demonstrations --capture-input
```

Input capture reads physical `/dev/input/event*` devices directly, requests monotonic kernel timestamps, and never grabs devices. Membership in group `input` is normally required: `sudo usermod -aG input "$USER"`, followed by logout/login. The recorder never invokes sudo and prints this exact diagnostic when access is unavailable.

Screen capture keeps one XDG ScreenCast session, one PipeWire remote FD and one
persistent GStreamer appsink pipeline alive for the recording. Successive frame
requests sample that stream; they do not reopen the Portal target per frame.

The manifest format is JSONL: `before`, `frame_before_ns`, `action_timestamp_ns`, `frame_after_ns`, `after`, action parameters, optional normalized `target_xy`, and layout-independent `motor_event`. Keyboard codes/timing are motor evidence only and carry no Language-teacher meaning. Relative mouse noise is collapsed into bounded trajectories. Wayland does not expose compositor-global position through evdev, so ordinary REL pointer coordinates are explicitly marked `integrated_relative`; ABS devices retain their raw axes.

Adjacent physical key presses also produce a layout-independent `type_text`
motor-program example containing key codes and timing, never decoded characters.
It trains when typing is appropriate without treating user text as Language truth.

BTN_MIDDLE is recorded if and only if the kernel produces it. On the current
user mouse the middle button does not work at the system/device level, so its
absence is not a recorder miss; wheel scrolling remains an independent event.

Validate alignment, frames and normalized targets:

```bash
.venv-rocm/bin/python -m worlds.desktop.training.validate_dataset state/desktop_demonstrations/manifest.jsonl
```

Train or resume:

```bash
.venv-rocm/bin/python -m worlds.desktop.training.train state/desktop_demonstrations/manifest.jsonl --checkpoint state/neural_desktop.pt
.venv-rocm/bin/python -m worlds.desktop.training.train state/desktop_demonstrations/manifest.jsonl --checkpoint state/neural_desktop.pt --resume
```

The following live-guidance workflow is retained only as an explicit
bootstrap/debug utility; it is not used by `runtime.embodied`. During an active guidance
window, no physical mouse intervention is an explicit session-level confirmation;
physical motion is tracked until it settles and its final position becomes the
correction target. The network's own output is never teacher evidence by itself.
If the mentor physically clicks, the final pointer position and kernel button
timestamp become a demonstrated click transition; the post-click frame is its
causal consequence. This records the mentor's click even when autonomous clicks
remain disabled.

With `--feedback-keys`, mouse guidance uses physical `1` for meaningful and `2`
for miss. `1` authorizes a physical left click when `--allow-clicks` is set;
the consequence frame is captured after that click. Keyboard exploration uses
`3` for meaningful and `4` for miss after the proposed physical key was pressed.
The collector temporarily EVIOCGRABs the rating keyboard, so rating digits do
not leak into the focused application. Positive feedback imitates the selected
action/region/key; negative feedback suppresses it.

```bash
.venv-rocm/bin/python -m worlds.desktop.training.live_guidance \
  --output state/desktop_demonstrations \
  --checkpoint state/neural_desktop.pt
```

Left/double clicks require `--allow-clicks`; right click requires the separate
`--allow-right-clicks` permission. In feedback mode, `--allow-typing` enables
one-key-at-a-time exploration through a learned layout-independent physical-key
head. Modifiers and Enter are excluded from this exploratory vocabulary. This
is motor evidence and never Language teacher data. Outside feedback mode,
typing still requires `--text-ref PATH`; that file is text selected by
Language/cognition and is explicitly not Language teacher data.

Smoke test (never invokes the actuator):

```bash
.venv-rocm/bin/python -m worlds.desktop.neural_smoke
```

Bounded overfit/evaluation (does not overwrite the checkpoint):

```bash
.venv-rocm/bin/python -m worlds.desktop.training.visual_control_overfit \
  state/desktop_demonstrations/manifest.jsonl --samples 1 --steps 80
```

Pixel-only live probes. The default moves only. The following commands
explicitly authorize one click, then one click plus short physical typing;
success is measured from causal pixel differences after the preceding motor
stage, not from AT-SPI or recurrent-state drift:

```bash
.venv-rocm/bin/python -m worlds.desktop.visual_control_live_test \
  --checkpoint state/neural_desktop.pt
.venv-rocm/bin/python -m worlds.desktop.visual_control_live_test \
  --checkpoint state/neural_desktop.pt --allow-clicks
.venv-rocm/bin/python -m worlds.desktop.visual_control_live_test \
  --checkpoint state/neural_desktop.pt --allow-clicks --allow-typing --text heiwa
```

Only records explicitly marked `teacher.language_teacher=true` may supply
`teacher.region_bbox` and `teacher.visual_text`. AT-SPI may create such offline
labels in `worlds/desktop/teacher/`; it is not imported by neural inference,
the neural World adapter or the live visual-control probe. Physical keyboard
events remain motor evidence and are never converted to Language truth.

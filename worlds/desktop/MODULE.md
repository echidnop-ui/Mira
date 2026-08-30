# Desktop World

## PURPOSE

Pixel-first neural body through which the same persistent Heiwa observes and acts
on the Linux desktop.

Desktop is an organ, not a separate agent and not the central cognition.

## PRODUCTION PATH

`portal_screencast.py`
-> real desktop pixels
-> `inference.py`
-> `neural.py::NeuralDesktopBody`
-> `neural_adapter.py::NeuralDesktopWorldAdapter`
-> `WorldFrame(Observation, dynamic Affordances, CompletedAction, Consequence)`
-> `runtime/desktop.py`
-> shared Core
-> shared Dynamic Policy
-> ActionIntent
-> `motor.py::DesktopMotor` (ydotool/uinput)
-> next real pixels
-> visual completion/consequence attribution.

AT-SPI adapters exist for bounded semantic/mentor/regression workflows, but the
normal neural Desktop production perception path is pixels.

## NEURAL BODY

`NeuralDesktopBody` contains:

- convolutional pixel encoder;
- spatial Transformer;
- learned region queries/attention;
- temporal GRU state conditioned on previous motor action;
- categorical motor head;
- target/key/confidence/trajectory heads;
- next-latent/change world-model heads;
- foveal byte reader.

It is a specialist organ. It does not own final cognitive action selection.

## ACTION SELECTION — IMPORTANT

Old behavior:

pixels -> motor categorical head -> sample ONE action -> offer ONE Affordance.

That made a collapsed motor head an effective command. Shared Policy had no
alternative to choose, so repetition could become an infinite physical loop.
`inference.py` temporarily used an action-agnostic repetition penalty/sampling to
reduce collapse.

Current behavior:

pixels -> motor distribution -> mechanically valid motor Affordances
                              -> shared DynamicAffordancePolicy
                              -> final ActionIntent.

Every candidate carries its own `neural_action_probability` and generic
`policy_prior_probability`. The motor head remains useful as a learned prior;
shared Policy can override it from causal experience.

No action-specific or repetition-count cognitive rule chooses an alternative.

## MECHANICAL AVAILABILITY

The World may filter actions that cannot currently be executed because mouse,
click, right-click, keyboard or typing capabilities are disabled. That is a
mechanism/capability decision, not cognition.

Current neural candidates include the executable subset of:

- noop;
- move_pointer;
- scroll;
- left_click;
- double_click;
- right_click;
- key_press;
- type_text.

`key_chord` is not exposed by the neural adapter until a complete chord
parameterizer exists.

## REPETITION VS ACTIVE ACTION

Repeated atomic actions on successive cycles are separate actions. A repeated
click/move/scroll problem belongs to Policy/action selection.

`ActiveAction / continue / cancel` is only correct for a physical operation that
actually remains active across transitions, e.g. a held key/button, drag or
long-running external operation. Do not use ActiveAction as a label for repeated
atomic decisions.

## CONSEQUENCE ATTRIBUTION

`neural_adapter.py` stores the causal before-frame/state for an accepted attempt.
On the next poll it compares pixel-derived patch tokens. Recurrent latent drift
alone is not sufficient evidence that a physical action changed the screen.

Input ownership is checked using evdev and `PhysicalActionArbiter`. User/foreign
interference makes attribution uncertain rather than falsely crediting Heiwa.

## LOCAL ONLINE LEARNING

`online_learning.py` trains predictive/representation/competence/grounding
paths from causal before/action/after experience and balanced replay.

Autonomous consequence learning intentionally excludes the motor policy
selector heads from self-confirming updates. Motor selector plasticity belongs
to demonstration/explicitly grounded training paths.

## MUST NOT

- replace neural pixels with hand-written brightness/contrast cognition;
- let the Desktop specialist make the final shared cognitive decision;
- create another parallel ydotool motor while `motor.py` owns physical input;
- interpret raw user input as implicit approval/rejection;
- claim completion without an observed external effect;
- treat own generated speech as teacher evidence.

## TELEMETRY

Useful fields include frame digest, neural action distribution, offered
Affordances, selected action, confidence, before/after visual change, predicted
consequence error, ownership/interference and online-learning metrics.

## FILES / READ SCOPE

For the current collapse/action-selection task read only:

- `MODULE.md`
- `neural.py`
- `inference.py`
- `neural_adapter.py`
- `motor.py` when execution changes
- `online_learning.py` when local visual learning changes
- `runtime/desktop.py` and `policy/dynamic.py` only for the shared selection
  boundary.

Do not scan Market or unrelated Language internals for this task.


## LIVE ACTION ANTI-COLLAPSE

The neural Desktop motor head is a specialist prior, not the final action selector.
Live embodied Policy samples from the shared action distribution instead of using a
permanent argmax. Owned non-noop actions that produce no observed pixel effect send
negative causal feedback to the exact Policy transition. Observable effect is treated
as competence evidence only and is not automatically rewarded as useful behavior.
There is no hardcoded `repeat N times -> cancel` rule.

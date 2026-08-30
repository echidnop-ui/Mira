# Heiwa v2 Module Map

`contracts/`
: Shared causal vocabulary: Observation, Affordance, ActionIntent,
  ActionAttempt, CompletedAction, Consequence.

`core/`
: Shared persistent cognition. Production implementation:
  `PersonalCognitiveCore` in `core/personal.py`; experts in `core/experts.py`.

`policy/`
: Shared trainable selection over dynamic Affordances. Production implementation:
  `DynamicAffordancePolicy` in `policy/dynamic.py`.

`runtime/`
: Integration/orchestration. `runtime/desktop.py` bridges Desktop WorldFrames to
  shared Core/Policy. `runtime/embodied.py` is the persistent neural Desktop +
  Language/mentor organism loop.

`worlds/desktop/`
: Pixel-first neural vision/body, portal capture, physical motor, local visual
  consequence learning and bounded teacher/regression helpers.

`language/`
: Developmental neural language organ. It is an organ of the same Heiwa, not a
  separate chatbot personality.

`worlds/market/`
: Another World of the same Heiwa. It is not the current Desktop-collapse task.

Rule: do not create a parallel replacement module when the responsibility
already has a production implementation. Read the target `MODULE.md` first.

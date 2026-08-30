# Heiwa v2

## Identity

Heiwa is one persistent hybrid neural cognitive agent. Desktop, Language, Market,
Game and future environments are Worlds/organs of the same Heiwa, not separate
personalities or separate agents.

Core principle:

World -> Observation -> shared Core -> dynamic Affordances -> shared Policy
-> ActionIntent -> ActionAttempt -> externally observed completion/consequence
-> causal learning -> next persistent state

`ActionIntent != ActionAttempt != CompletedAction`.

## Current shared Core

The verified scalable Core is `core/personal.py` (`PersonalCognitiveCore`). Its
public boundary is `CoreInput -> IntegratedCognitiveState` in
`core/interface.py`.

Internally it is heterogeneous rather than a single small recurrent cell. It
contains persistent temporal/work/slow state, a sparse router and four expert
families:

- selective SSM;
- stabilized recurrent / xLSTM-style path;
- causal TCN;
- history attention.

The Core is world-agnostic. Desktop pixels, language vectors and market vectors
must be normalized outside the Core and then integrated through the same public
contract.

## Dynamic Policy

The verified shared action selector is `policy/dynamic.py`.

A World exposes the actions that are mechanically available now. Shared Policy
scores that variable set and produces one `ActionIntent`. There is no global
hard-coded BUY/SELL/CLICK/SPEAK enum in the central cognition.

A specialist World may provide a soft `policy_prior_probability` on an
Affordance. This is guidance, not a decision: shared Policy may override the
specialist prior through causal learning.

## Current Desktop production path

The real neural Desktop path is pixel-first:

PortalScreenCast
-> real pixels
-> `NeuralDesktopBody.observe()`
-> CNN/ConvBlock + spatial Transformer + region attention + temporal GRU
-> visual state
-> neural motor distribution / target / key / trajectory proposals
-> mechanically valid Desktop Affordances
-> shared `DynamicAffordancePolicy`
-> ActionIntent
-> `DesktopMotor` / ydotool
-> ActionAttempt
-> next real pixels
-> CompletedAction + visual Consequence
-> local predictive/representation learning and shared causal feedback

Important: `NeuralDesktopBody` is a specialist organ, not the final cognitive
selector.

### Repetition / action-collapse rule

The previous live path sampled exactly one motor action inside
`NeuralDesktopInference` and exposed only that one action to shared Policy. If
the categorical motor head collapsed (historically e.g. to right-click), the
rest of Heiwa literally had no alternative to choose.

That architecture is removed.

Now the motor head exposes its distribution as a specialist prior. The Desktop
World offers multiple mechanically executable motor Affordances. Shared Policy
performs the final selection.

There is no hand-written "if repeated N times, cancel" cognitive rule and no
action-specific anti-collapse penalty.

`ActiveAction / continue / cancel` is reserved for genuinely long-running
physical operations such as held keys/buttons, drag or an external operation.
Repeated atomic clicks/moves/scrolls across separate cognitive cycles are new
actions and must be solved by Policy, not mislabeled as one ongoing action.

## Desktop perception and learning

Production perception is neural pixels, not hand-written brightness/contrast or
AT-SPI semantics. AT-SPI remains useful for bounded teacher/regression paths but
is not the normal pixel-first neural Desktop perception channel.

The neural Desktop already preserves causal before/after frames and uses
pixel-derived patch-token change to verify physical effects. Recurrent latent
drift alone is not treated as proof that the desktop physically changed.

Autonomous visual consequence learning does not directly rewrite the categorical
motor selector from its own self-confirming outcomes. Motor-policy learning is
reserved for demonstration/explicitly grounded action-selection signals; the
online learner continues to train predictive/representation/competence paths.

## Current development direction

The active program is integrated persistent causal learning across existing
organs. The current execution queue lives in staged private GitHub Issues and
is governed by `automation/symphony/WORKFLOW.md`. Runtime, Language and Memory
may be tested independently only when the test demonstrates their effect on the
shared loop; isolated module metrics are not the final success criterion.

Desktop work remains pixel-first and must make the existing neural body
participate in shared cognition rather than replacing it with another Desktop
stack. GPU/Vision work is an explicitly scheduled exclusive lane, not an
opportunistic background task.

Do not create parallel `screen.py`, `wayland.py`, another ydotool motor, or a
second small Core/Policy unless the existing module contract is intentionally
being replaced.

The next experiments should measure whether shared Policy actually escapes a
collapsed specialist motor prior during long real Desktop life, while keeping
vision, motor targets and causal consequence attribution intact.

Neural changes follow `docs/NEURAL_EXPERIMENT_PROTOCOL.md`. Autonomous Codex
work follows `docs/SYMPHONY_OPERATIONS.md`.

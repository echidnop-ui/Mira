# Heiwa v2 — Market Perception

## AI CONTEXT CONTRACT

This file is the context entry point for the Market Neural Organ.

If the task concerns:

- TCN;
- SSM;
- Attention;
- recurrent/xLSTM;
- DeepLOB;
- Market Router;
- expert fusion;
- MarketRepresentation;
- market-perception learning;
- expert availability;
- expert uncertainty;

start here.

Do NOT scan the whole repository.

Normal read order:

1. `AGENTS.md`
2. `worlds/market/MODULE.md`
3. this `worlds/market/perception/MODULE.md`
4. `worlds/market/interface.py`
5. `worlds/market/perception/interface.py`
6. only the exact perception files named by the task

Do not automatically read:

- T-Bank networking;
- order execution;
- shared Core internals;
- Language;
- Forum/Game/VTuber/Desktop Worlds;
- Memory internals;
- UI.

A local change to one expert should normally stay inside that expert plus the perception public interface.

---

## PURPOSE

Market Perception is the trainable market-specific sensory organ of Heiwa.

It converts normalized market data into a learned representation suitable for shared cognition.

It does NOT make the final BUY/SELL/HOLD decision.

It does NOT execute orders.

It does NOT own Heiwa's personality.

Its job is:

    MarketInput
        ↓
    heterogeneous neural experts
        ↓
    Router
        ↓
    learned fusion
        ↓
    MarketRepresentation

The shared Heiwa Core and Policy use that representation later.

---

## WHY HETEROGENEOUS

The market contains several different temporal and structural problems.

One architecture should not be forced to solve all of them.

Target expert families:

- TCN;
- SSM;
- Attention;
- recurrent / xLSTM;
- DeepLOB;
- optional regime / volatility / liquidity specialists.

Different experts may learn different useful representations.

The Router learns which available representations matter in the current context.

---

## TARGET TOPOLOGY

Conceptually:

                        MarketInput
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
     TCN Expert          SSM Expert       Attention Expert
        │                   │                    │
        ├────────────┐      │      ┌─────────────┤
        │            ▼      ▼      ▼             │
        │       recurrent / xLSTM                │
        │                                        │
        └──────────────┐             ┌────────────┘
                       ▼             ▼
                  Regime /        DeepLOB
                  Volatility      specialist
                       │             │
                       └─────┬───────┘
                             ▼
                       Trainable Router
                             ↓
                        Learned Fusion
                             ↓
                   MarketRepresentation

This topology is a target structure, not a requirement that every expert exist on day one.

The public interfaces should support this architecture from the beginning.

---

## IMPORTANT: EXPERTS DO NOT TRADE

No expert should directly produce an executable market action.

Bad:

    TCN → BUY

    DeepLOB → SELL

    SSM → HOLD

Good:

    expert → representation
                ↓
             Router
                ↓
      MarketRepresentation
                ↓
         shared Heiwa Core
                ↓
             Policy
                ↓
          market action

Auxiliary predictions are allowed.

For example an expert may estimate:

- future return distribution;
- volatility;
- order-flow imbalance;
- market regime;
- uncertainty;
- fill probability.

But these remain learned signals/representations, not final action execution.

---

## PUBLIC INPUT

The perception module receives `MarketInput` from:

    worlds/market/interface.py

It should not receive raw T-Bank SDK objects.

The normalization layer is responsible for translating broker data into Market-native structures.

Experts may consume different parts of `MarketInput`.

For example:

TCN:
    short / medium temporal windows

SSM:
    medium / long temporal windows

DeepLOB:
    order_book_history

Router:
    expert outputs
    availability
    optional shared context

Do not force every expert to consume every source.

---

## MULTI-TIMESCALE INPUT

The Market input may expose:

    micro
    short
    medium
    long

These are conceptual views, not fixed universal durations.

Each expert may choose its own internal processing of these windows.

Do not hardcode a single market horizon into shared Heiwa contracts.

---

## EXPERT OUTPUT

Every expert should eventually expose the same perception-local output contract.

Conceptually:

    ExpertOutput
        name
        representation
        uncertainty
        available
        auxiliary
        telemetry

The exact Python type belongs in:

    worlds/market/perception/interface.py

Shared Runtime/Core should not import expert implementation classes directly.

---

## EXPERT AVAILABILITY

An expert may be unavailable.

Examples:

- no Level 2 data → DeepLOB unavailable;
- insufficient sequence history → long-horizon SSM unavailable;
- missing feature set → specialist unavailable.

Absence must be explicit.

Do not represent "unavailable" as a normal confident zero vector unless the public contract explicitly distinguishes it.

Router must ignore or mask unavailable experts.

---

## TCN RESPONSIBILITY

TCN is a temporal convolution specialist.

Expected role:

- local temporal motifs;
- short/medium sequence structure;
- abrupt local changes;
- volatility bursts;
- repeated short patterns;
- multiscale convolutional temporal features.

TCN should not know:

- T-Bank API;
- broker order ids;
- shared autobiographical memory;
- final Policy choice.

TCN should be replaceable without changing the rest of the Market module.

---

## SSM RESPONSIBILITY

SSM is a state-space temporal specialist.

Expected role:

- long sequential context;
- efficient recurrent temporal processing;
- persistent regime dynamics;
- long-range state transitions;
- sequence compression.

The exact SSM architecture may evolve.

Possible future experiments:

- diagonal SSM;
- gated SSM;
- Mamba-like selective SSM;
- custom recurrent state-space variants.

The public expert interface should not depend on the specific SSM family.

---

## ATTENTION RESPONSIBILITY

Attention is a contextual relation specialist.

Expected role:

- non-local temporal relationships;
- distant-event comparison;
- cross-feature relations;
- regime/context interactions;
- selective access to prior sequence elements.

Attention is not required to be a giant Transformer.

Small/local/sparse attention is acceptable.

Do not automatically scale it into an LLM-style architecture.

---

## RECURRENT / XLSTM RESPONSIBILITY

Recurrent expert provides another form of persistent nonlinear sequence state.

Expected role:

- nonlinear sequential memory;
- stable recurrent context;
- complementary dynamics to SSM;
- temporal information that may be poorly represented by convolution alone.

Implementation may evolve from simple recurrent gating toward xLSTM-style structures.

Do not expose recurrent internal state outside the expert.

---

## DEEPLOB RESPONSIBILITY

DeepLOB is a specialist for order-book structure.

Expected role:

- bid/ask imbalance;
- book shape;
- micro-price structure;
- order-flow patterns;
- short-horizon microstructure representation.

DeepLOB must remain optional.

The absence of useful L2 history must not block the whole Market Neural Organ.

---

## OPTIONAL SPECIALISTS

Potential specialists may include:

- volatility;
- regime;
- liquidity;
- execution/fill probability;
- cross-instrument relations;
- instrument embeddings.

Do not add a specialist simply because it sounds useful.

Every specialist must answer:

    WHAT UNIQUE INFORMATION DOES IT REPRESENT?

and:

    WHY IS THIS NOT ALREADY COVERED BY AN EXISTING EXPERT?

---

## ROUTER RESPONSIBILITY

Router receives available expert outputs and decides how they contribute to the combined market representation.

Router may condition on:

- expert representations;
- expert uncertainty;
- expert availability;
- normalized market context;
- portfolio context;
- optional shared Heiwa context through an explicit public contract.

Router should learn routing.

Do not hardcode:

    if volatility > X:
        use_tcn()

    if l2_available:
        deeplob_weight = 0.5

Those are behaviors Router may discover, not fixed rules.

---

## ROUTER COLLAPSE

A Router may collapse onto one expert.

This is a known experimental risk.

Do not immediately solve collapse with many arbitrary bonuses.

Useful diagnostics first:

- expert selection frequency;
- router entropy;
- average expert weight;
- gradient received by each expert;
- expert availability;
- expert prediction/representation usefulness.

Only add anti-collapse mechanisms after observing a concrete failure.

Possible later mechanisms may include:

- temperature;
- entropy regularization;
- load balancing;
- sparse top-k routing;
- exploration noise;
- expert dropout.

These are experimental mechanisms, not mandatory initial rules.

---

## FUSION RESPONSIBILITY

Fusion converts routed expert information into the final `MarketRepresentation.vector`.

Router and Fusion are distinct ideas:

Router:
    which experts matter and how much

Fusion:
    how selected expert information becomes one representation

Fusion may be:

- weighted projection;
- gated projection;
- small learned MLP;
- recurrent fusion;
- attention-based fusion;
- another trainable mechanism.

Do not expose expert concatenation directly as a permanent public contract.

---

## SHARED HEIWA CONTEXT

Market perception may eventually use shared Heiwa context.

Example:

    shared_context

passed explicitly through:

    MarketNeuralOrgan.encode(...)

This must remain a public contract.

Perception code must not reach into:

    core._fast
    core._work
    core._slow

If shared context is useful, it arrives as a representation, not as Core internals.

---

## MARKET REPRESENTATION

The perception module returns:

    MarketRepresentation

defined in:

    worlds/market/interface.py

It should contain a learned vector plus compact diagnostics such as:

- uncertainty;
- expert weights;
- expert availability;
- representation diagnostics.

It must not contain broker SDK objects.

It must not contain direct execution commands.

---

## CAUSAL TRACE

If downstream learning arrives later, the Market Neural Organ may keep a minimal causal trace keyed by:

    transition_id

The trace may include:

- expert inputs/activations needed for learning;
- expert outputs;
- availability mask;
- router state;
- fusion state.

Do not store unrelated World state in this trace.

This is technical training state, not memory/autobiography.

---

## LEARNING

Public learning path:

    downstream representation gradient
        ↓
    MarketNeuralOrgan.learn(
        transition_id,
        representation_gradient
    )
        ↓
    exact causal trace
        ↓
    Fusion
        ↓
    Router
        ↓
    experts that actually contributed

Inactive experts should not receive normal causal credit for a transition they did not influence.

---

## AUXILIARY LEARNING

Experts may have auxiliary objectives if those objectives improve useful representation learning.

Examples:

TCN:
    future local return / volatility features

SSM:
    future latent-state prediction

DeepLOB:
    order-flow / short-horizon microstructure prediction

Router:
    downstream usefulness / routing objective

Auxiliary tasks must not silently replace the shared cognitive objective.

They are representation-learning aids.

---

## END-TO-END LEARNING

Long-term desired path:

    consequence / predictor / value / policy learning
        ↓
    shared Core gradient
        ↓
    MarketRepresentation gradient
        ↓
    Market Neural Organ
        ↓
    Router + Fusion + contributing experts

Do not assume every component must receive the same gradient on every transition.

Causal participation matters.

---

## PERSISTENCE

Perception persistence may include:

- expert weights;
- router weights;
- fusion weights;
- recurrent expert states;
- SSM expert states;
- optimizer/training state if needed;
- learning counters.

Causal traces may remain transient.

Do not persist T-Bank credentials here.

---

## RESET

Perception reset may clear:

- expert weights;
- router/fusion weights;
- recurrent states;
- causal traces;
- learning counters.

This is an experimental neural reset.

It does not reset the broker account.

---

## TELEMETRY

Minimum useful telemetry:

    representation_norm
    learning_updates
    last_gradient_norm

Per expert:

    available
    weight
    representation_norm
    uncertainty
    gradient_norm
    update_count

Router:

    routing entropy
    expert weights
    active expert count

Fusion:

    fusion norm
    output norm

Do not add heavy telemetry machinery before the first runnable organ.

---

## COMPUTE

Expert sizes must be configurable.

The initial system may use small dimensions for iteration speed.

The public architecture must allow later scaling.

Do not hardcode assumptions like:

    hidden_size = 24 forever

inside public Market contracts.

Implementation defaults are allowed.

---

## DEVICE BOUNDARY

Perception implementation may later use:

- CPU;
- GPU;
- PyTorch;
- another neural runtime.

The public interface should remain Python-native and framework-agnostic where practical.

Do not leak framework tensor types into unrelated modules unless a later performance decision explicitly changes this rule.

---

## FILE SHAPE

Planned local structure:

    perception/
    ├── MODULE.md
    ├── interface.py
    ├── organ.py
    ├── router.py
    ├── fusion.py
    ├── telemetry.py
    └── experts/
        ├── tcn.py
        ├── ssm.py
        ├── attention.py
        ├── recurrent.py
        └── deeplob.py

Do not create all files empty.

Create a file only when its responsibility becomes real.

`organ.py` will likely coordinate experts, Router, Fusion and causal traces.

It should not become a generic Market god object.

---

## PUBLIC INTERFACE

The public perception interface belongs in:

    perception/interface.py

It should define only perception-local contracts required across perception files.

Possible initial types:

    ExpertInput
    ExpertOutput
    MarketExpert
    RouterDecision

Do not duplicate:

    MarketInput
    MarketRepresentation

because those already belong to:

    worlds/market/interface.py

---

## READ SCOPE FOR AI CODERS

### Change one expert

Read:

    AGENTS.md
    worlds/market/perception/MODULE.md
    worlds/market/perception/interface.py
    exact expert file

Do not read other experts unless the shared expert contract changes.

---

### Change Router

Read:

    worlds/market/perception/MODULE.md
    worlds/market/perception/interface.py
    worlds/market/perception/router.py
    worlds/market/perception/fusion.py

Use experts only through `ExpertOutput`.

---

### Change Fusion

Read:

    worlds/market/perception/MODULE.md
    worlds/market/perception/interface.py
    worlds/market/perception/fusion.py

Read Router only if its output contract is insufficient.

---

### Change full Market Neural Organ coordination

Read:

    worlds/market/perception/MODULE.md
    worlds/market/perception/interface.py
    worlds/market/perception/organ.py
    worlds/market/perception/router.py
    worlds/market/perception/fusion.py

Do not inspect every expert implementation unless necessary.

---

### Change causal learning

Read:

    worlds/market/perception/MODULE.md
    worlds/market/perception/interface.py
    worlds/market/perception/organ.py
    router/fusion implementation
    only experts whose learning path changes

If the public MarketRepresentation gradient contract changes, also read:

    worlds/market/interface.py

---

### Connect perception to shared Core

Read:

    worlds/market/perception/MODULE.md
    worlds/market/interface.py
    core/MODULE.md
    core/interface.py
    runtime/MODULE.md
    runtime integration file

Do not inspect individual expert internals.

---

## CONTEXT BUDGET RULE

Do not read all experts to change one expert.

Do not read T-Bank networking to change the Router.

Do not read Core neural internals to change Market perception.

Do not read Market execution to change TCN.

Use interfaces first.

Expand scope only for a concrete contract reason.

---

## CHANGE SCOPE RULE

Before modifying perception, identify:

    TARGET:
    expert / router / fusion / organ / learning

    PUBLIC CONTRACT CHANGE:
    yes / no

    FILES REQUIRED:
    ...

    EXTERNAL PUBLIC INTERFACE REQUIRED:
    none / exact file

Keep the task local whenever possible.

---

## MUST NOT

Market Perception must not:

- execute trades;
- choose the final market action;
- call T-Bank directly;
- own global autobiographical identity;
- directly modify shared Self;
- import Core internals;
- hide expert absence;
- train inactive experts from unrelated outcomes;
- fabricate counterfactual outcomes as observed truth;
- require reading every expert for a local change;
- expose implementation tensors across unrelated modules without an explicit architecture decision.

---

## DESIGN TEST

The boundary is healthy when:

- TCN can be replaced without reading T-Bank code.
- SSM can be changed without reading Attention implementation.
- Router can work from `ExpertOutput` without knowing expert internals.
- Shared Core can consume `MarketRepresentation` without knowing the expert topology.
- Market execution can submit an order without knowing how the representation was produced.
- A coding agent can modify one expert with a small context window.

That is the intended design.

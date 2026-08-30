# Heiwa v2 — AI Development Router

This repository is intentionally designed for AI-assisted development with a limited context window.

The primary engineering rule is:

> Do not read the whole repository to solve a local task.

Heiwa is divided into modules with explicit responsibilities, public interfaces, and local context maps.

A coding agent should normally be able to modify one cognitive capability after reading only:

1. this file;
2. the target module's `MODULE.md`;
3. the target module's public interface;
4. the specific implementation files listed by that module.

Repository-wide exploration is NOT the default workflow.

---

# 1. PROJECT IDENTITY

Heiwa is a persistent hybrid cognitive agent.

It is NOT primarily a chatbot.

The same Heiwa may eventually interact with many Worlds:

- financial markets;
- forum/chat;
- games;
- multiple game characters;
- game architecture/building systems;
- VTuber environments;
- desktop/computer environments;
- future virtual or physical environments.

These are different environments of one persistent agent.

Do not create a separate independent "Heiwa personality" for every World.

Conceptually:

    Market ─────┐
    Forum ──────┤
    Game ───────┤
    VTuber ─────┼──→ one persistent Heiwa
    Desktop ────┤
    Future ─────┘

The central cognitive system should remain world-agnostic.

---

# 2. FIRST RULE FOR EVERY TASK

Before reading implementation files, determine the target module.

Write mentally or explicitly:

    TARGET MODULE:
    <module>

    TASK TYPE:
    internal
    integration
    public-contract

    INITIAL READ SCOPE:
    <files>

    PUBLIC CONTRACT CHANGE:
    yes / no

    EXTERNAL MODULES REQUIRED:
    none

Then read only the initial scope.

Do NOT begin with:

    find .
    grep the entire repository
    reading every Python file
    scanning every MODULE.md

unless the task is explicitly repository-wide.

---

# 3. CONTEXT BUDGET RULE

Context is an engineering resource.

Do not read a file because it "might be related."

Use this order:

1. `AGENTS.md`
2. target `MODULE.md`
3. public contracts
4. target `interface.py`
5. exact implementation file relevant to the task

Expand the scope only after discovering a concrete reason.

Bad:

    "Memory may affect this, so I will inspect all memory files."

Good:

    "The Core receives MemoryContext.
     I need only the public MemoryContext contract."

---

# 4. MODULE NAVIGATION

Use this table to choose the initial module.

## Core / internal state

Keywords and concepts:

- Core
- hidden state
- FAST
- WORK
- SLOW
- SSM
- recurrent state
- temporal dynamics
- representation learning
- core gradient
- state persistence

Start with:

    core/MODULE.md

Do NOT initially read:

    worlds/
    language/
    market integration
    UI
    graph internals

---

## Policy / action selection

Keywords and concepts:

- choose action
- policy
- action scoring
- affordance scoring
- exploration policy
- action probability
- decision
- available actions

Start with:

    policy/MODULE.md

---

## Memory

Keywords and concepts:

- memory
- remember
- recall
- retrieval
- graph
- episodic memory
- semantic memory
- consolidation
- MemoryQuery
- GNN
- MemoryContext

Start with:

    memory/MODULE.md

---

## Learning / consequences

Keywords and concepts:

- consequence
- delayed learning
- transition learning
- value target
- replay
- causal credit
- predictor learning
- feedback routing

Start with:

    learning/MODULE.md

---

## Motivation

Keywords and concepts:

- curiosity
- uncertainty
- interest
- impulse
- initiative
- information gain
- learning progress
- internal drive

Start with:

    motivation/MODULE.md

---

## Language

Keywords and concepts:

- language
- words
- phrases
- grammar
- receptive learning
- expressive learning
- realization
- utterance
- construction
- CONTINUE / STOP

Start with:

    language/MODULE.md

---

## Runtime

Keywords and concepts:

- cognition cycle
- transition lifecycle
- event routing
- consequence routing
- orchestration
- connecting modules
- pending transition

Start with:

    runtime/MODULE.md

Runtime should coordinate modules.

Runtime should NOT become the brain itself.

---

## Market / T-Bank

Keywords and concepts:

- T-Bank
- Tinkoff Invest API
- shares
- instrument
- price
- candle
- order book
- portfolio
- order
- sandbox
- real account
- market execution

Start with:

    worlds/market/MODULE.md

For a T-Bank API problem, do NOT inspect Core internals unless the public contract is demonstrably insufficient.

---

## Forum / chat

Keywords and concepts:

- forum
- chat
- messages
- replies
- reactions
- publishing

Start with:

    worlds/forum/MODULE.md

---

## Game

Keywords and concepts:

- game
- character
- movement
- objects
- building
- combat
- game world
- game architect

Start with:

    worlds/game/MODULE.md

---

## VTuber

Keywords and concepts:

- VTuber
- stream
- facial expression
- animation
- avatar
- voice output
- live interaction

Start with:

    worlds/vtuber/MODULE.md

---

## Desktop

Keywords and concepts:

- desktop
- files
- applications
- mouse
- keyboard
- screen
- computer control

Start with:

    worlds/desktop/MODULE.md

---

# 5. CONTRACTS

Shared cross-module structures belong in:

    contracts/

Contracts should remain:

- small;
- explicit;
- world-agnostic when possible;
- implementation-agnostic.

Examples may include:

    Observation
    Affordance
    ActionIntent
    ActionAttempt
    CompletedAction
    Consequence
    FeedbackEvent
    CoreInput
    CoreOutput
    MemoryContext

Do NOT place implementation logic in `contracts/`.

Do NOT create a new shared contract merely because two files happen to use similar data.

Create shared contracts when there is a real module boundary.

---

# 6. PUBLIC INTERFACE RULE

Each major module should expose a small obvious public surface.

Preferred pattern:

    core/interface.py
    policy/interface.py
    memory/interface.py
    language/interface.py
    worlds/market/interface.py

Other modules should use these interfaces rather than importing implementation internals.

Bad:

    from core.multiscale import _FastState
    from memory.graph import _GraphNode
    from policy.network import _PolicyLayer

Good:

    from core.interface import CoreOutput
    from memory.interface import MemoryContext
    from policy.interface import PolicyDecision

Implementation files should be treated as private to their module unless documented otherwise.

---

# 7. SCOPE EXPANSION RULE

If the initial module is insufficient, expand scope minimally.

Before opening another module, identify:

    SCOPE EXPANSION

    Need:
    <specific public file>

    Reason:
    <concrete technical reason>

    Do not need:
    <other internals of that module>

Example:

    TARGET MODULE:
    runtime

    Need:
    memory/interface.py

    Reason:
    Runtime must pass MemoryContext to Core.

    Do not need:
    memory/query.py
    memory/graph.py
    memory/gnn.py

Opening one public interface does not justify scanning the entire neighboring module.

---

# 8. CROSS-MODULE TASKS

Some tasks genuinely cross module boundaries.

Example:

    "Connect MemoryContext to Core."

Then the expected read scope may be:

    memory/MODULE.md
    memory/interface.py
    core/MODULE.md
    core/interface.py
    runtime/MODULE.md
    runtime integration file

Even then, do NOT automatically read:

    memory graph internals
    Core neural implementation
    Policy implementation
    Language
    Worlds

unless the concrete change requires them.

---

# 9. MODULE.md RULE

Every significant module should contain a `MODULE.md`.

Its purpose is not generic documentation.

It is the local context map for developers and AI coding agents.

Each `MODULE.md` should explain:

    PURPOSE
    INPUT
    OUTPUT
    OWNS STATE
    PUBLIC API
    DEPENDS ON
    MUST NOT
    WHAT DOES IT CAUSE?
    WHAT TRAINS IT?
    PERSISTENCE
    TELEMETRY
    FILES
    READ SCOPE FOR AI CODERS
    CONTEXT BUDGET RULE

A task inside a module should normally begin by reading its `MODULE.md`.

---

# 10. MODULE RESPONSIBILITY RULE

Use:

> ONE MODULE = ONE COMPLETE RESPONSIBILITY

Do not use:

> ONE FUNCTION = ONE FILE

Avoid both extremes:

- giant 5000-line god objects;
- hundreds of 20-line files that require constant navigation.

General orientation:

    Python file:
    often ~150–600 lines

    major module:
    often ~3–8 related files

These are guidelines, not hard limits.

Logical responsibility is more important than line count.

---

# 11. WORLD BOUNDARY

World adapters connect Heiwa to external environments.

A World may:

- observe external state;
- normalize it into Observation;
- expose available Affordances;
- execute ActionIntent;
- report ActionAttempt;
- report CompletedAction;
- report environmental consequences.

A World must NOT make Heiwa's cognitive decisions.

Bad:

    if market_price_dropped:
        buy()

Bad:

    if user_said_hello:
        reply_hello()

Good:

    World produces Observation + Affordances.

    Heiwa Policy decides what to do.

---

# 12. DYNAMIC AFFORDANCES

Heiwa does not have one permanent global enumeration of every possible external action.

Worlds expose available actions dynamically.

Examples:

    market.buy
    market.sell
    market.cancel_order

    forum.reply
    forum.react

    game.move
    game.take
    game.build

    vtuber.change_expression

    desktop.open_file

Policy should reason over available affordances rather than require the central cognitive architecture to know every future World action.

Adding a new World should not require editing a global action enum.

---

# 13. CAUSAL LEARNING RULE

For every trainable cognitive module ask:

    WHAT DOES IT CAUSE?

    WHAT TRAINS IT?

A module should receive learning credit only for consequences it could causally influence.

Example:

MemoryQuery selected memory.

If:

    selected memory
    → MemoryContext
    → influenced Core/Policy
    → influenced Action
    → consequence arrived

then MemoryQuery may receive causal learning evidence.

If the selected memory was never used by the decision path, do NOT train MemoryQuery from that consequence.

---

# 14. TRANSITION ID

Meaningful cognitive activity should be traceable by `transition_id`.

Conceptually:

    transition
    ├── Observation
    ├── available Affordances
    ├── Core state
    ├── MemoryContext
    ├── Policy decision
    ├── ActionIntent
    ├── ActionAttempt
    ├── CompletedAction
    └── Consequence

Delayed learning must return to the correct causal transition.

Do not train arbitrary current state from an unrelated old consequence.

---

# 15. INTENTION IS NOT EXECUTION

Keep these concepts separate:

    ActionIntent
    ActionAttempt
    CompletedAction
    Consequence

Examples:

Wanting to buy a share is not the same as sending an order.

Sending an order is not the same as getting filled.

Wanting to speak is not the same as publishing a message.

A rejected or suppressed action must not create false success history.

---

# 16. MECHANISM VS COGNITIVE DECISION

Fundamental Heiwa principle:

> MECHANISM can be programmed.
> COGNITIVE DECISION should preferably be learned.

Deterministic infrastructure is acceptable for:

- API calls;
- databases;
- persistence;
- serialization;
- event delivery;
- execution;
- permissions;
- UI;
- normalization.

Avoid cognitive shortcuts such as:

    if unknown:
        ask()

    if greeting:
        say("hello")

    if price_drop > X:
        buy()

    if curiosity > threshold:
        explore()

Mechanics execute decisions.

They should not secretly replace cognition.

---

# 17. CORE BOUNDARY

Core represents persistent internal dynamics.

Core should not know:

- T-Bank SDK;
- forum HTTP routes;
- Minecraft;
- VTuber software;
- UI components;
- SQLite graph implementation.

World-specific data should be transformed before entering Core.

Core should expose representations through its public interface.

---

# 18. MEMORY BOUNDARY

Memory owns stored experience and retrieval mechanisms.

Core should not directly query graph storage.

Preferred direction:

    Memory
        ↓
    MemoryContext
        ↓
    Core / cognition

Do not make Core aware of memory database internals.

---

# 19. LANGUAGE BOUNDARY

Language is an organ, not Heiwa itself.

Language handles things such as:

- receptive linguistic learning;
- expressive competence;
- constructions;
- realization;
- utterances.

Language should not secretly become Policy.

Policy deciding to act and Language realizing a linguistic action are separate responsibilities.

---

# 20. RUNTIME BOUNDARY

Runtime coordinates the cognitive cycle.

Runtime may connect:

    World
    Memory
    Core
    Policy
    Learning

Runtime should not accumulate cognitive heuristics.

If `runtime/cognition_cycle.py` starts containing rules for:

- buying;
- remembering specific semantic concepts;
- deciding whether to speak;
- grammar;
- curiosity thresholds;

the architecture is drifting toward a god object.

Move responsibility to the proper module.

---

# 21. CURRENT FIRST WORLD

The first planned runnable Heiwa v2 World is:

    Market / T-Bank Invest

Initial goal:

    market observation
        ↓
    Core
        ↓
    available Affordances
        ↓
    Policy
        ↓
    market action
        ↓
    execution
        ↓
    CompletedAction
        ↓
    Consequence
        ↓
    learning
        ↓
    next state

Language is not required for this first vertical slice.

Start with T-Bank sandbox.

A real backend may later use a small experimental real-money account.

Sandbox and real execution should remain explicitly distinguishable.

---

# 22. OLD HEIWA

The old Heiwa implementation is not the architecture of record for v2.

Do NOT mechanically copy the old repository.

Old Heiwa is a source of:

- useful mechanisms;
- experiments;
- architectural lessons;
- previously discovered bugs.

When porting an old mechanism:

    identify responsibility
        ↓
    define clean v2 boundary
        ↓
    extract useful mechanism
        ↓
    remove legacy coupling
        ↓
    integrate through public interface

Do not reproduce historical coupling simply because it already exists.

---

# 23. KNOWN HISTORICAL FAILURE MODES

Avoid reintroducing these old classes of bugs:

- intent treated as completed action;
- rejected speech creating phantom action history;
- own generated speech becoming teacher evidence;
- autonomous output self-reinforcement;
- idle ticks producing runaway autobiography;
- memory selection trained on outcomes it did not influence;
- telemetry-only neural modules that never affect cognition;
- deterministic unknown → ASK shortcuts;
- UI metadata leaking into semantic cognition;
- ordinary dialogue interpreted as control commands;
- giant manually weighted intrinsic reward formulas;
- downstream heads learning while Core representations remain effectively frozen.

Do not assume these bugs exist in v2.

They are architectural lessons.

---

# 24. RUNNABLE FIRST

Do not spend weeks creating empty architecture.

Prefer an early vertical slice.

Current target:

    world-agnostic contracts
        ↓
    Core
        ↓
    dynamic Affordances
        ↓
    Policy
        ↓
    T-Bank Market World
        ↓
    execution
        ↓
    Consequence
        ↓
    Learning
        ↓
    telemetry

As soon as this loop runs, experiment with it.

Do not require every future module to exist first.

---

# 25. EXPERIMENTAL DEVELOPMENT

Heiwa v2 is experimental.

Prefer:

    small real change
    → run
    → observe
    → diagnose
    → change again

Do not add large amounts of:

- compatibility infrastructure;
- migration code;
- abstraction layers;
- fallback systems;
- formal tests;

unless they solve a concrete current problem.

Live diagnostic experiments are acceptable.

---

# 26. REPOSITORY SEARCH POLICY

Repository-wide search is allowed when the task itself is repository-wide.

Examples:

- rename a public contract everywhere;
- find all imports of an interface;
- remove a deprecated module;
- investigate an unknown integration leak.

For normal local work, do not start with repository-wide exploration.

If searching globally becomes necessary, search for a specific symbol or contract.

Avoid broad exploratory dumping of unrelated files into context.

---

# 27. MODIFYING PUBLIC CONTRACTS

Changing a public contract is more expensive than changing module internals.

Before changing one, identify:

    PUBLIC CONTRACT:
    <name>

    WHY CURRENT CONTRACT IS INSUFFICIENT:
    <reason>

    KNOWN CONSUMERS:
    <modules>

Then inspect only consumers of that contract.

Do not scan every module.

---

# 28. DEPENDENCY DIRECTION

Prefer dependencies through public abstractions.

Implementation internals should not form hidden cycles.

Especially avoid:

    Core internals ↔ Policy internals
    Memory internals ↔ Runtime internals
    World internals ↔ Core internals

If two modules require extensive knowledge of each other's internal state,
the module boundary is probably wrong.

---

# 29. AI HANDOFF STYLE

When finishing a task, report compactly:

    TARGET MODULE:
    ...

    CHANGED:
    ...

    PUBLIC CONTRACT CHANGED:
    yes/no

    NEXT LIKELY MODULE:
    ...

Do not produce a huge repository summary after every local change.

The local `MODULE.md` and code should carry most of the persistent context.

---

# 30. SOURCE OF TRUTH

When an actual repository/archive is available:

> The current code is the source of truth.

Documentation describes intended architecture.

If code and documentation disagree:

1. identify the disagreement;
2. do not silently assume either is correct;
3. fix the appropriate side as part of the task if necessary.

Do not invent files or behavior that have not been observed.

---

# 31. FINAL CONTEXT RULE

The desired development pattern is:

    User:
    "Change Memory Query."

    Agent:
    reads AGENTS.md
        ↓
    reads memory/MODULE.md
        ↓
    reads memory/interface.py
        ↓
    reads query implementation
        ↓
    changes only Memory module

NOT:

    User:
    "Change Memory Query."

    Agent:
    reads 80 Python files
        ↓
    reconstructs entire Heiwa architecture
        ↓
    consumes most context before editing anything

The repository architecture exists partly to prevent that failure mode.

---

# 32. DESIGN TEST

A module boundary is healthy when:

> A competent developer or coding agent can understand and change the module
> by reading its MODULE.md, public contracts, public interface, and its own
> implementation files.

If understanding a local module requires reading unrelated World, UI,
Language, Memory, and Core internals, the architecture needs correction.

---

# 33. SYMPHONY AUTOMATED WORK

Automated Codex work is governed by:

    automation/symphony/WORKFLOW.md
    docs/SYMPHONY_OPERATIONS.md

GitHub Issues and their structured stage comments are the active execution
queue. Historical NEXT sections in handoff documents are not authority to
override a newer staged Issue.

An automated agent must perform exactly the role named by its stage label. A
Tester does not edit. A Fixer requires reproducible Tester evidence and makes
one fix. An Integrator independently verifies and never force-updates `main`.

Removing `symphony-ready` is the emergency stop for an Issue.

---

# 34. NEURAL EXPERIMENTS

For neural training, checkpoint, online-learning or representation changes,
also follow:

    docs/NEURAL_EXPERIMENT_PROTOCOL.md

The required cadence is:

    EXPERIMENT -> ANALYSIS -> ONE FIX -> EXPERIMENT

Do not accept a neural change from falling loss alone. Require an independent
gate, causal attribution, save/reload evidence and retention of previously
accepted skills.

# Heiwa Symphony Operations

This document describes the automation policy used for the private Heiwa v2
repository. It is for Codex development agents, not for Heiwa's cognitive
runtime.

## Boundary

OpenAI Symphony provides scheduling, issue polling, isolated per-issue
workspaces, Codex app-server sessions, retries, reconciliation, configuration
reload and observability. It is not a general workflow engine and does not
decide how Heiwa should be designed, tested or merged.

The Tester/Fixer/Integrator pipeline is Heiwa policy implemented in
`automation/symphony/WORKFLOW.md` and GitHub labels.

## Local installation

- CPU launcher: `/home/echidna/.local/bin/heiwa-symphony`
- GPU launcher: `/home/echidna/.local/bin/heiwa-symphony-gpu`
- CPU workflow: `automation/symphony/WORKFLOW.md`
- GPU workflow: `automation/symphony/GPU_WORKFLOW.md`
- secret: `/home/echidna/.config/symphony/heiwa/secrets/github-token`
- CPU workspaces: `/home/echidna/.local/share/symphony/heiwa-workspaces`
- GPU workspaces: `/home/echidna/.local/share/symphony/heiwa-gpu-workspaces`
- GPU artifacts: `/home/echidna/.local/share/symphony/heiwa-gpu-artifacts`
- CPU logs: `/home/echidna/.local/state/symphony/heiwa-logs`
- GPU logs: `/home/echidna/.local/state/symphony/heiwa-gpu-logs`
- CPU dashboard: `http://127.0.0.1:4040/`
- GPU dashboard: `http://127.0.0.1:4041/`

The secret must remain outside Git and have mode `0600`.

Start after login or reboot:

```bash
heiwa-symphony
heiwa-symphony-gpu
```

User services are installed as `heiwa-symphony.service` and
`heiwa-symphony-gpu.service`. Inspect them with:

```bash
systemctl --user status heiwa-symphony heiwa-symphony-gpu
```

The launcher refuses a missing/unsafe token and a duplicate listener. `Ctrl+C`
must stop the service through `SIGTERM`; it must not leave a hidden Erlang VM
holding port 4040.

## Dispatch contract

An open private GitHub Issue is eligible only with:

```text
symphony-ready
stage:tester | stage:fixer | stage:integrator
```

Exactly one stage label is allowed.

### Tester

- runs a decisive baseline and independent probe;
- edits no code;
- identifies the first reproducible failure and one quantitative gate;
- records `TESTER HANDOFF` in the Issue;
- advances only a bounded, evidence-backed defect to `stage:fixer`.

### Fixer

- requires the Tester evidence;
- reproduces it before editing;
- makes one root-cause fix;
- reruns the same benchmark and relevant regressions;
- publishes a separate remote branch/commit;
- records `FIXER HANDOFF` and advances to `stage:integrator`.

### Integrator

- independently checks the candidate diff and scope;
- reruns the reproducer, regressions and persistence/reload checks;
- accepts only through a non-force GitHub merge/ref operation;
- otherwise marks `needs-human` and stops instead of looping automatically.

Successful Issues receive `stage:done` and close. A blocked or ambiguous Issue
loses `symphony-ready`, so it cannot consume more Codex turns unnoticed.

## Budget controls

Current limits:

- at most 3 concurrent agents across independent Issues;
- exactly 1 Codex turn per stage invocation;
- at most one automatic Tester/Fixer/Integrator pass per Issue;
- 30 minute turn timeout and 3 minute stall timeout;
- no recursive issue creation;
- no automatic Fixer/Integrator retry loop.

Symphony exposes token telemetry, but that display is an operational usage
counter, not a project acceptance metric. Stop dispatch by removing
`symphony-ready`; closing an Issue also makes it ineligible and causes
reconciliation to stop an active worker.

## Creating a good task

Every Issue should state:

```text
TARGET MODULE
OBSERVED FAILURE OR QUESTION
EXECUTABLE BASELINE
INDEPENDENT HOLDOUT OR NEGATIVE CONTROL
SUCCESS GATE
ALLOWED FILES
FORBIDDEN SCOPE
PERSISTENCE/RELOAD REQUIREMENT
```

Do not enqueue vague requests such as "improve Heiwa". Split them at real
module or experiment boundaries, not arbitrary file boundaries.

GitHub's `Symphony staged task` Issue Form collects these fields and applies
the initial dispatch labels automatically.

## Enabled useful features

- workflow hot reload for future dispatches;
- persistent isolated workspace per Issue;
- clean workspaces fast-forward to current `main` before a new stage, while
  dirty candidate workspaces are preserved;
- label-gated dispatch;
- bounded parallelism;
- retry/backoff for infrastructure failure;
- stall and turn timeouts;
- automatic stop when an Issue becomes ineligible;
- dashboard plus structured logs;
- durable handoff through Issue comments and remote commits.

## Available but intentionally not enabled

- destructive workspace reset before every stage: it would discard useful
  staged evidence;
- automatic force-push or blind merge to `main`;
- unlimited continuation turns;
- independent GPU workers competing for the same RX 6800;
- automatic recursive task generation;
- public repository or public logs;
- automatic PR creation, until the token and review policy explicitly allow it.

## GPU lane

GPU experiments are serialized by `GPU_WORKFLOW.md` and require all labels:

```text
symphony-gpu-ready
resource:gpu
stage:tester | stage:fixer | stage:integrator
```

The lane has concurrency `1`. Its Codex app-server holds a host `flock` for the
entire turn. Before enabling an experiment, the Issue must name its checkpoint,
dataset/split, expected duration, acceptance/retention gates and restoration
plan. Use the `Symphony GPU experiment` Issue Form.

GPU agents write candidate checkpoints, metrics, plots and journals under
`state/symphony_artifacts/` inside their workspace. Lifecycle hooks copy those
files to the persistent GPU artifact root after every attempt and before
workspace removal. No candidate checkpoint is promoted into live Heiwa state
automatically.

A CPU Symphony agent must never opportunistically start ROCm work. A manual GPU
job does not automatically honor Symphony's lock; do not schedule one at the
same time as a GPU-ready Issue.

## Failure recovery

- Port 4040 occupied: do not start a second service; stop the known Symphony
  process with `SIGTERM`.
- Agent repeatedly retries: remove `symphony-ready`, inspect structured logs,
  fix the infrastructure cause, then restore the label.
- Candidate cannot merge cleanly: mark `needs-human`; never force-update main.
- GitHub transport failure after a mutating request: query current Issue/ref
  state before retrying. Never blindly retry POST creation.
- Workspace and remote branch disagree: the remote commit and Issue handoff are
  durable truth; local workspace state is diagnostic only.

Official reference: <https://github.com/openai/symphony/blob/main/SPEC.md>

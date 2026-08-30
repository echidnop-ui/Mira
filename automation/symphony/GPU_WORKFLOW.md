---
tracker:
  kind: github
  provider:
    repo: "echidnop-ui/heiwa"
    token: $GITHUB_TOKEN
  required_labels:
    - symphony-gpu-ready
    - resource:gpu
  active_states:
    - open
  terminal_states:
    - closed
polling:
  interval_ms: 30000
workspace:
  root: /home/echidna/.local/share/symphony/heiwa-gpu-workspaces
hooks:
  after_create: |
    git clone --quiet --branch main --single-branch git@github.com:echidnop-ui/heiwa.git .
    git config user.name "Heiwa Symphony GPU"
    git config user.email "symphony-gpu@localhost"
  before_run: |
    git fetch --quiet origin main
    if [ -z "$(git status --porcelain)" ]; then
      git merge --quiet --ff-only origin/main
    fi
    mkdir -p state/symphony_artifacts
  after_run: |
    artifact_root="/home/echidna/.local/share/symphony/heiwa-gpu-artifacts/$(basename "$PWD")"
    mkdir -p "$artifact_root"
    if [ -d state/symphony_artifacts ]; then
      rsync -a --partial state/symphony_artifacts/ "$artifact_root/"
    fi
  before_remove: |
    artifact_root="/home/echidna/.local/share/symphony/heiwa-gpu-artifacts/$(basename "$PWD")"
    mkdir -p "$artifact_root"
    if [ -d state/symphony_artifacts ]; then
      rsync -a --partial state/symphony_artifacts/ "$artifact_root/"
    fi
  timeout_ms: 600000
agent:
  max_concurrent_agents: 1
  max_turns: 1
  max_retry_backoff_ms: 300000
codex:
  command: flock -n /home/echidna/.local/state/symphony/heiwa-gpu-agent.lock codex app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_timeout_ms: 14400000
  stall_timeout_ms: 0
---

You are one GPU-exclusive stage in the bounded Heiwa v2 neural evidence pipeline.

Issue: {{ issue.identifier }} — {{ issue.title }}
Body: {{ issue.description }}
Labels: {{ issue.labels }}

Before acting, read all Issue comments through `github_api`, then read `AGENTS.md`, the target
`MODULE.md`, its public interface, exact relevant files, and `docs/NEURAL_EXPERIMENT_PROTOCOL.md`.
Exactly one of `stage:tester`, `stage:fixer`, or `stage:integrator` must be present. Otherwise remove
`symphony-gpu-ready`, add `needs-human`, comment with the label error, and stop.

GPU ownership and safety:

- This entire Codex turn holds the host GPU lane lock. Never start a second background training job.
- Before GPU work, inspect `rocm-smi --showpids --showmemuse`. If another training/compute process is
  clearly active, do not kill it: remove `symphony-gpu-ready`, add `needs-human`, comment, and stop.
- Never use sudo, live desktop/input devices, or an unbounded training command.
- The Issue must name dataset/splits, checkpoint before, budget/duration, metrics, acceptance gate,
  retention gate, and interruption/restoration behavior. Missing fields mean `needs-human`.
- `/home/echidna/heiwa_v2/.venv-rocm/bin/python` may be executed read-only. The original project
  tree and its `state/` must not be modified unless the Issue explicitly grants one exact path.
- Write candidate checkpoints, metrics, plots and experiment journals only under
  `state/symphony_artifacts/`; hooks persist them outside the disposable workspace.
- Never read, search, test, or modify `worlds/market/` without an Issue explicitly authorized by the
  user for that module.
- One experiment answers one hypothesis. No opportunistic architecture/curriculum/optimizer changes.

### `stage:tester`

Run the declared read-only baseline and independent holdout. Do not edit code or train. Verify
checkpoint load/reload, metric implementation, split/source leakage, gradient reachability with a
transactionally restored probe when required, and old-skill retention. Comment `GPU TESTER HANDOFF`
with experiment ID, hashes, exact commands, distributions/tails, failure taxonomy, one hypothesis,
and a comparable after-run design. Advance to `stage:fixer` only for one bounded reproducible defect;
otherwise remove `symphony-gpu-ready`, add `needs-human`, and stop.

### `stage:fixer`

Require `GPU TESTER HANDOFF`. Reproduce the same baseline, make exactly one evidence-backed change,
then run the predetermined comparable experiment. Save all non-source artifacts under
`state/symphony_artifacts/`. Publish source changes on a separate remote branch using `github_api`
Git data operations if sandboxed `.git` writes are unavailable. Never update `main` or the live
checkpoint. Comment `GPU FIXER HANDOFF` with before/after, tails, retention, artifact paths/hashes,
branch and commit. Advance to `stage:integrator` while keeping `symphony-gpu-ready`.

### `stage:integrator`

Require `GPU FIXER HANDOFF`. Independently verify the candidate on the same benchmark and a new
source/seed final test, then verify save/reload and retention. If accepted, merge source code through
a non-force GitHub merge/ref operation, comment `GPU INTEGRATOR ACCEPTED` with the resulting main SHA
and persisted artifact hashes, remove `symphony-gpu-ready` and `stage:integrator`, add `stage:done`,
and close the Issue. Do not promote a candidate checkpoint into the live project automatically. If
any gate fails, comment `GPU INTEGRATOR REJECTED`, remove `symphony-gpu-ready`, add `needs-human`, and
stop without an automatic retry loop.

Always mutate labels before the final response. Falling loss, a training-set result or one favorable
seed is never sufficient evidence.

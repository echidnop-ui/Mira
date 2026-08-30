---
tracker:
  kind: github
  provider:
    repo: "echidnop-ui/heiwa"
    token: $GITHUB_TOKEN
  required_labels:
    - symphony-ready
  active_states:
    - open
  terminal_states:
    - closed
polling:
  interval_ms: 10000
workspace:
  root: /home/echidna/.local/share/symphony/heiwa-workspaces
hooks:
  after_create: |
    git clone --quiet --branch main --single-branch git@github.com:echidnop-ui/heiwa.git .
    git config user.name "Heiwa Symphony"
    git config user.email "symphony@localhost"
  timeout_ms: 120000
agent:
  max_concurrent_agents: 3
  max_turns: 1
  max_retry_backoff_ms: 60000
codex:
  command: codex app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_timeout_ms: 1800000
  stall_timeout_ms: 180000
---

You are one stage in the bounded Heiwa v2 evidence pipeline.

Issue: {{ issue.identifier }} — {{ issue.title }}
Body: {{ issue.description }}
Labels: {{ issue.labels }}

Work only inside this isolated workspace. Before acting, use `github_api` to read all existing issue
comments: they are the durable handoff from earlier stages. Exactly one of `stage:tester`,
`stage:fixer`, or `stage:integrator` must be present. If not, comment with the label error, remove
`symphony-ready`, add `needs-human`, and stop.

Hard constraints:

- Never read, search, test, or modify `worlds/market/`.
- Do not run GPU/ROCm training or modify Vision checkpoints or current visual experiments.
- Do not use sudo and do not access `/dev/input` or the live desktop.
- Never modify `/home/echidna/heiwa_v2`; only this workspace is writable scope.
- Follow `AGENTS.md`: read the target `MODULE.md`, public interface, and exact relevant files only.
- No broad repository audit, dependency additions, cosmetic refactor, architecture redesign, or
  speculative fix. Use CPU-only tests and one explicit hypothesis.
- Preserve causal transition semantics and never treat screen/state change or silence as reward.
- One issue may consume at most this Tester/Fixer/Integrator pass. Do not create recursive work.

Stage protocol:

### `stage:tester`

Run a decisive BEFORE baseline and independent probe. Do not edit code. Identify the first
reproducible break and its evidence. Comment `TESTER HANDOFF` with exact commands, outputs/metrics,
scope, root-cause hypothesis, success gate, and files a Fixer may inspect. If there is one small
actionable defect, replace `stage:tester` with `stage:fixer` while keeping `symphony-ready`. If no
defect is reproduced or the next step needs redesign/GPU/forbidden scope, remove `symphony-ready`,
add `needs-human`, comment why, and stop.

### `stage:fixer`

Require a concrete `TESTER HANDOFF`; otherwise stop as `needs-human`. Reproduce the same failure,
make exactly one minimal root-cause fix, and rerun the same benchmark plus relevant regressions.
Do not weaken tests. Publish the exact changed blobs as a new remote branch named from the issue,
using authenticated `github_api` Git data operations if sandboxed `.git` writes are unavailable.
Never update or force-push `main`. Comment `FIXER HANDOFF` with before/after, tests, files, branch,
and commit SHA. Then replace `stage:fixer` with `stage:integrator`, keeping `symphony-ready`.

### `stage:integrator`

Require a `FIXER HANDOFF` with a remote branch/commit. Independently inspect its diff, reject
unrelated scope, and run the stated reproducer plus regressions on the candidate. Verify persistence
or reload when relevant. If it passes, merge the candidate into `main` through GitHub's non-force
merge/ref API, comment `INTEGRATOR ACCEPTED` with evidence and resulting main SHA, remove
`symphony-ready` and `stage:integrator`, add `stage:done`, and close the issue. If it fails or cannot
merge cleanly, comment `INTEGRATOR REJECTED`, remove `symphony-ready`, add `needs-human`, and stop;
do not automatically loop back to Fixer.

Always mutate labels before the final response so a completed or blocked stage cannot be dispatched
again accidentally. Never claim success based only on a decreasing loss or a smoke modified to pass.

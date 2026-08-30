# INSTALL

This ZIP contains documentation/state-verification files only. It does not modify neural implementation code.

```bash
cd ~/heiwa_v2
unzip -o ~/Downloads/HEIWA_AI_HANDOFF_STATE_PACK.zip -d ~/heiwa_v2

source .venv-rocm/bin/activate
python -m py_compile tools/verify_ai_state.py
python tools/verify_ai_state.py
```

For a new chat, say:

"Work on my local Heiwa v2 project. First read AGENTS.md, AI_HANDOFF.md, AI_STATE.md and the relevant MODULE.md. Then run `python tools/verify_ai_state.py`. The local tree is authoritative. Do not guess that generated patches were applied. Continue from the first unverified/current NEXT item."

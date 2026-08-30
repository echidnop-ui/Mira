from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def yn(v: bool) -> str:
    return "YES" if v else "NO"

def main() -> int:
    print("HEIWA AI STATE VERIFY")
    print(f"root: {ROOT}")

    print("\nDOCS")
    for rel in (
        "AGENTS.md",
        "AI_HANDOFF.md",
        "AI_STATE.md",
        "language/MODULE.md",
        "runtime/MODULE.md",
        "worlds/desktop/MODULE.md",
    ):
        print(f"{rel}: {yn((ROOT / rel).is_file())}")

    print("\nLANGUAGE")
    pedagogy = ROOT / "language/compositional_pedagogy.py"
    runtime = ROOT / "runtime/language_compositional_school.py"
    curriculum = ROOT / "state/language_curriculum/compositional.json"
    print(f"language/compositional_pedagogy.py: {yn(pedagogy.is_file())}")
    print(f"runtime/language_compositional_school.py: {yn(runtime.is_file())}")
    print(f"curriculum: {yn(curriculum.is_file())}")

    if curriculum.is_file():
        try:
            obj = json.loads(curriculum.read_text(encoding="utf-8"))
            print("curriculum_json: VALID")
            for key in ("atoms", "train", "holdout"):
                val = obj.get(key)
                print(f"curriculum_{key}: {len(val) if isinstance(val, list) else 'INVALID'}")
        except Exception as exc:
            print(f"curriculum_json: INVALID {exc!r}")

    if pedagogy.is_file():
        text = pedagogy.read_text(encoding="utf-8", errors="replace").lower()
        print(f"pedagogy_mentions_slot: {yn('slot' in text)}")
        print(f"pedagogy_mentions_holdout: {yn('holdout' in text)}")
        print("NOTE: markers are hints only; inspect source before claiming V2 is installed.")

    print("\nCHECKPOINTS")
    for rel in (
        "state/language_bootstrap.pt",
        "state/language_compositional.pt",
        "state/language_compositional_v2.pt",
        "state/desktop_policy_weights.pt",
    ):
        p = ROOT / rel
        print(f"{rel}: {yn(p.is_file())} size={p.stat().st_size if p.is_file() else 0}")

    print("\nMENTOR/DESKTOP")
    for rel in (
        "worlds/desktop/perception.py",
        "runtime/mentor.py",
    ):
        print(f"{rel}: {yn((ROOT / rel).is_file())}")

    print("\nINTERPRETATION")
    v2 = ROOT / "state/language_compositional_v2.pt"
    if v2.is_file():
        print("V2 checkpoint exists, but this alone does NOT prove semantic success. Read run metrics.")
    else:
        print("No V2 checkpoint. Do NOT claim a successful V2 run.")

    print("Fresh local source/output overrides AI_HANDOFF.md if they disagree.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

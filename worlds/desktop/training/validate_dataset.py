from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(manifest: Path) -> dict[str, object]:
    root = manifest.parent; records = 0; trainable = 0; bytes_total = manifest.stat().st_size
    actions: dict[str, int] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        item = json.loads(line); records += 1
        before, action, after = int(item["frame_before_ns"]), int(item["action_timestamp_ns"]), int(item["frame_after_ns"])
        if not before < action < after: raise ValueError(f"line {line_number}: frame/action alignment is invalid")
        for key in ("before", "after"):
            path = root / item[key]
            if not path.is_file(): raise ValueError(f"line {line_number}: missing {key} frame {path}")
        target = item.get("target_xy")
        if target is not None and (len(target) != 2 or any(not 0 <= float(value) <= 1 for value in target)):
            raise ValueError(f"line {line_number}: target_xy is outside [0,1]")
        teacher = item.get("teacher")
        if isinstance(teacher, dict):
            bbox = teacher.get("region_bbox")
            visual_text = teacher.get("visual_text")
            if ((bbox is not None or visual_text is not None)
                    and not (teacher.get("language_teacher", False) or teacher.get("pixel_supervision", False))):
                raise ValueError(f"line {line_number}: visual text/bbox label is not explicit pixel supervision")
            if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4
                                     or any(not 0 <= float(value) <= 1 for value in bbox)):
                raise ValueError(f"line {line_number}: teacher region_bbox is invalid")
        action_name = str(item["action"]); actions[action_name] = actions.get(action_name, 0) + 1
        trainable += int(item.get("trainable", True))
    unique_frames = {str(item) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()
                     for payload in (json.loads(line),) for item in (root / payload["before"], root / payload["after"])}
    bytes_total += sum(Path(path).stat().st_size for path in unique_frames)
    return {"ok": True, "records": records, "trainable": trainable, "actions": actions,
            "disk_bytes": bytes_total, "approx_bytes_per_transition": round(bytes_total / max(records, 1))}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", type=Path); args = parser.parse_args()
    print(json.dumps(validate(args.manifest), ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())

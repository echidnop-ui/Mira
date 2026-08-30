from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from worlds.desktop.inference import NeuralDesktopInference
from worlds.desktop.neural import ACTION_KINDS, NeuralDesktopBody, NeuralDesktopConfig
from worlds.desktop.training.train import save_checkpoint


class _Portal:
    def close(self) -> None: pass


def main() -> int:
    torch.manual_seed(71)
    config = NeuralDesktopConfig(
        image_height=64, image_width=96, hidden_size=64, token_size=64,
        region_count=4, transformer_layers=1, transformer_heads=4,
        action_embed_size=16, task_context_size=32, max_read_bytes=12,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); checkpoint = root / "desktop.pt"; replay = root / "replay.pt"
        seed = NeuralDesktopBody(config); optimizer = torch.optim.AdamW(seed.parameters())
        save_checkpoint(checkpoint, seed, optimizer, step=0)
        inference = NeuralDesktopInference(checkpoint, portal=_Portal(), device="cpu", replay_path=replay)
        inference.learner.slow_every = 1
        before = torch.rand(1, 3, 64, 96); after = before.clone()
        after[:, :, 20:44, 38:70] = (after[:, :, 20:44, 38:70] + .35).clamp(0, 1)
        action = ACTION_KINDS.index("left_click"); parameters = (.56, .50, 0.0, 0.0)
        first = inference.model.observe(before).detached()
        second = inference.model.observe(
            after, first, torch.tensor([action]), torch.tensor([parameters])
        ).detached()
        with torch.inference_mode(): initial = inference.model.propose(first).target_point.clone()
        initial_weight = inference.model.stem[0].weight.detach().clone()
        initial_pointer = {name: value.detach().clone() for name, value in inference.model.state_dict().items()
                           if name.startswith(("target_query.", "target_key.", "offset_head."))}
        initial_predictor = inference.model.next_latent[0].weight.detach().clone()
        metrics = inference.learn_transition(
            before_frame=before, after_frame=after, before_state=first, after_state=second,
            action_kind=action, parameters=parameters, ownership="heiwa", completion=True,
        )
        with torch.inference_mode(): changed = inference.model.propose(first).target_point.clone()
        assert metrics["fast_loss"] >= 0 and metrics["work_loss"] >= 0 and metrics["slow_loss"] >= 0
        assert metrics["fast_latency_ms"] >= 0 and metrics["work_latency_ms"] >= 0
        assert torch.equal(initial_weight, inference.model.stem[0].weight)
        assert torch.equal(initial, changed)
        assert all(torch.equal(value, inference.model.state_dict()[name]) for name, value in initial_pointer.items())
        assert not torch.equal(initial_predictor, inference.model.next_latent[0].weight)
        # External life enters replay/world learning but never becomes approval/rejection.
        external = inference.learn_transition(
            before_frame=after, after_frame=before, before_state=second, after_state=first,
            action_kind=ACTION_KINDS.index("noop"), parameters=(0, 0, 0, 0),
            ownership="user", completion=None,
        )
        assert external["ownership"] == "user" and external["completion"] is None
        inference.save_checkpoint(); saved = {
            name: value.detach().clone() for name, value in inference.model.state_dict().items()
        }; counters = inference.learner.learning_state(); inference.portal.close()
        restored = NeuralDesktopInference(checkpoint, portal=_Portal(), device="cpu", replay_path=replay)
        assert all(torch.equal(saved[name], value) for name, value in restored.model.state_dict().items())
        assert restored.learner.learning_state() == counters
        assert counters["slow_updates"] == 1
        assert len(restored.learner.replay) == 2
        print({"ok": True, "predictor_changed_online": True,
               "perception_frozen_online": True, "pointer_head_frozen_online": True,
               "replay_size": len(restored.learner.replay),
               "persistence_after_reload": True, "metrics": metrics})
    return 0


if __name__ == "__main__": raise SystemExit(main())

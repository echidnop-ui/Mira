from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from worlds.market.perception.fusion import AdaptiveMarketFusion
from worlds.market.perception.organ import HeterogeneousMarketOrgan
from worlds.market.perception.router import AdaptiveMarketRouter
from worlds.market.perception.experts.attention import AttentionMarketExpert
from worlds.market.perception.experts.deeplob import DeepLOBMarketExpert
from worlds.market.perception.experts.recurrent import RecurrentMarketExpert
from worlds.market.perception.experts.ssm import SSMMarketExpert
from worlds.market.perception.experts.tcn import TCNMarketExpert


@dataclass(frozen=True, slots=True)
class MarketPerceptionConfig:
    """
    Default construction config for the heterogeneous Market Neural Organ.

    This file owns practical first-run defaults only.

    It does NOT define shared Heiwa cognition and does NOT make trading
    decisions.

    The target Market horizon around 240 minutes is kept here as
    Market-specific training metadata and passed into specialists that may
    use it for auxiliary/downstream training.

    It is not a forced holding duration.
    """

    # Shared expert/fusion representation sizes.
    expert_representation_size: int = 64
    market_representation_size: int = 96

    # Market-specific prediction/training horizon.
    training_horizon_minutes: int = 240

    # Shared causal trace budget.
    expert_max_traces: int = 512
    router_max_traces: int = 4096
    fusion_max_traces: int = 4096

    # General optimizer defaults.
    expert_learning_rate: float = 1e-3
    router_learning_rate: float = 1e-3
    fusion_learning_rate: float = 1e-3

    gradient_clip: float = 5.0

    # Device string passed to PyTorch experts.
    #
    # None:
    #   expert selects CUDA/ROCm when available, otherwise CPU.
    device: str | None = None

    # --------------------------------------------------------
    # TCN
    # --------------------------------------------------------

    tcn_channels: int = 64
    tcn_branch_size: int = 48
    tcn_depth: int = 5
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.05
    tcn_min_steps: int = 8

    # --------------------------------------------------------
    # SSM
    # --------------------------------------------------------

    ssm_hidden_size: int = 64
    ssm_branch_size: int = 48
    ssm_depth: int = 3
    ssm_dropout: float = 0.05
    ssm_min_steps: int = 8

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    attention_hidden_size: int = 64
    attention_branch_size: int = 48
    attention_depth: int = 3
    attention_heads: int = 4
    attention_feedforward_multiplier: int = 4
    attention_dropout: float = 0.05
    attention_relative_buckets: int = 32
    attention_min_steps: int = 8
    attention_max_steps_per_window: int = 256

    # --------------------------------------------------------
    # Recurrent / xLSTM-style
    # --------------------------------------------------------

    recurrent_hidden_size: int = 64
    recurrent_branch_size: int = 48
    recurrent_depth: int = 3
    recurrent_dropout: float = 0.05
    recurrent_min_steps: int = 8
    recurrent_max_steps_per_window: int = 512

    # --------------------------------------------------------
    # DeepLOB
    # --------------------------------------------------------

    deeplob_levels: int = 10
    deeplob_conv_channels: int = 48
    deeplob_inception_channels: int = 24
    deeplob_recurrent_size: int = 64
    deeplob_recurrent_layers: int = 1
    deeplob_dropout: float = 0.05
    deeplob_min_snapshots: int = 16
    deeplob_max_snapshots: int = 256
    deeplob_max_traces: int = 256

    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    router_hidden_size: int = 32
    router_temperature: float = 1.0

    # None = dense routing over all available experts.
    #
    # Later we can experimentally move to sparse top-k routing without
    # changing expert or shared Heiwa contracts.
    router_top_k: int | None = None

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    tcn_seed: int = 20260827
    ssm_seed: int = 20260828
    attention_seed: int = 20260829
    recurrent_seed: int = 20260830
    deeplob_seed: int = 20260831
    router_seed: int = 20260901
    fusion_seed: int = 20260902

    def __post_init__(self) -> None:
        if self.expert_representation_size <= 0:
            raise ValueError(
                "expert_representation_size must be > 0"
            )

        if self.market_representation_size <= 0:
            raise ValueError(
                "market_representation_size must be > 0"
            )

        if self.training_horizon_minutes <= 0:
            raise ValueError(
                "training_horizon_minutes must be > 0"
            )

        if self.expert_max_traces <= 0:
            raise ValueError(
                "expert_max_traces must be > 0"
            )

        if self.router_max_traces <= 0:
            raise ValueError(
                "router_max_traces must be > 0"
            )

        if self.fusion_max_traces <= 0:
            raise ValueError(
                "fusion_max_traces must be > 0"
            )

        if self.expert_learning_rate <= 0.0:
            raise ValueError(
                "expert_learning_rate must be > 0"
            )

        if self.router_learning_rate <= 0.0:
            raise ValueError(
                "router_learning_rate must be > 0"
            )

        if self.fusion_learning_rate <= 0.0:
            raise ValueError(
                "fusion_learning_rate must be > 0"
            )

        if self.gradient_clip <= 0.0:
            raise ValueError(
                "gradient_clip must be > 0"
            )

        if self.router_hidden_size <= 0:
            raise ValueError(
                "router_hidden_size must be > 0"
            )

        if self.router_temperature <= 0.0:
            raise ValueError(
                "router_temperature must be > 0"
            )

        if (
            self.router_top_k is not None
            and self.router_top_k <= 0
        ):
            raise ValueError(
                "router_top_k must be > 0 or None"
            )


def build_default_market_organ(
    config: MarketPerceptionConfig | None = None,
) -> HeterogeneousMarketOrgan:
    """
    Construct the first full heterogeneous Market Neural Organ.

    Current expert set:

        TCN
        SSM
        Attention
        recurrent/xLSTM-style
        DeepLOB

    Forward:

        MarketInput
            ↓
        experts
            ↓
        AdaptiveMarketRouter
            ↓
        AdaptiveMarketFusion
            ↓
        MarketRepresentation

    DeepLOB may be unavailable on a transition when L2 history is missing.
    The remaining experts continue normally.

    This factory only constructs perception. It does not construct:
        - shared Heiwa Core;
        - Policy;
        - T-Bank backend;
        - execution;
        - OutcomePredictor.
    """

    cfg = (
        config
        if config is not None
        else MarketPerceptionConfig()
    )

    experts = (
        _build_tcn(cfg),
        _build_ssm(cfg),
        _build_attention(cfg),
        _build_recurrent(cfg),
        _build_deeplob(cfg),
    )

    router = AdaptiveMarketRouter(
        hidden_size=cfg.router_hidden_size,
        learning_rate=cfg.router_learning_rate,
        temperature=cfg.router_temperature,
        top_k=cfg.router_top_k,
        gradient_clip=cfg.gradient_clip,
        max_traces=cfg.router_max_traces,
        seed=cfg.router_seed,
    )

    fusion = AdaptiveMarketFusion(
        output_size=cfg.market_representation_size,
        learning_rate=cfg.fusion_learning_rate,
        gradient_clip=cfg.gradient_clip,
        max_traces=cfg.fusion_max_traces,
        seed=cfg.fusion_seed,
    )

    return HeterogeneousMarketOrgan(
        experts=experts,
        router=router,
        fusion=fusion,
    )


def build_market_organ_with_experts(
    experts: Iterable,
    *,
    market_representation_size: int = 96,
    router_hidden_size: int = 32,
    router_top_k: int | None = None,
    router_temperature: float = 1.0,
    router_learning_rate: float = 1e-3,
    fusion_learning_rate: float = 1e-3,
    gradient_clip: float = 5.0,
    router_max_traces: int = 4096,
    fusion_max_traces: int = 4096,
    router_seed: int = 20260901,
    fusion_seed: int = 20260902,
) -> HeterogeneousMarketOrgan:
    """
    Experimental construction helper.

    Allows ablation experiments such as:

        TCN only
        TCN + SSM
        TCN + SSM + Attention
        all experts except DeepLOB

    without changing Router/Fusion/Organ code.

    The caller supplies already-created expert objects that obey the
    MarketExpert contract.
    """

    router = AdaptiveMarketRouter(
        hidden_size=router_hidden_size,
        learning_rate=router_learning_rate,
        temperature=router_temperature,
        top_k=router_top_k,
        gradient_clip=gradient_clip,
        max_traces=router_max_traces,
        seed=router_seed,
    )

    fusion = AdaptiveMarketFusion(
        output_size=market_representation_size,
        learning_rate=fusion_learning_rate,
        gradient_clip=gradient_clip,
        max_traces=fusion_max_traces,
        seed=fusion_seed,
    )

    return HeterogeneousMarketOrgan(
        experts=tuple(experts),
        router=router,
        fusion=fusion,
    )


def _build_tcn(
    cfg: MarketPerceptionConfig,
) -> TCNMarketExpert:
    return TCNMarketExpert(
        representation_size=(
            cfg.expert_representation_size
        ),
        channels=cfg.tcn_channels,
        branch_size=cfg.tcn_branch_size,
        depth=cfg.tcn_depth,
        kernel_size=cfg.tcn_kernel_size,
        dropout=cfg.tcn_dropout,
        learning_rate=cfg.expert_learning_rate,
        gradient_clip=cfg.gradient_clip,
        min_steps=cfg.tcn_min_steps,
        max_traces=cfg.expert_max_traces,
        training_horizon_minutes=(
            cfg.training_horizon_minutes
        ),
        device=cfg.device,
        seed=cfg.tcn_seed,
    )


def _build_ssm(
    cfg: MarketPerceptionConfig,
) -> SSMMarketExpert:
    return SSMMarketExpert(
        representation_size=(
            cfg.expert_representation_size
        ),
        hidden_size=cfg.ssm_hidden_size,
        branch_size=cfg.ssm_branch_size,
        depth=cfg.ssm_depth,
        dropout=cfg.ssm_dropout,
        learning_rate=cfg.expert_learning_rate,
        gradient_clip=cfg.gradient_clip,
        min_steps=cfg.ssm_min_steps,
        max_traces=cfg.expert_max_traces,
        training_horizon_minutes=(
            cfg.training_horizon_minutes
        ),
        device=cfg.device,
        seed=cfg.ssm_seed,
    )


def _build_attention(
    cfg: MarketPerceptionConfig,
) -> AttentionMarketExpert:
    return AttentionMarketExpert(
        representation_size=(
            cfg.expert_representation_size
        ),
        hidden_size=cfg.attention_hidden_size,
        branch_size=cfg.attention_branch_size,
        depth=cfg.attention_depth,
        heads=cfg.attention_heads,
        feedforward_multiplier=(
            cfg.attention_feedforward_multiplier
        ),
        dropout=cfg.attention_dropout,
        relative_buckets=(
            cfg.attention_relative_buckets
        ),
        learning_rate=cfg.expert_learning_rate,
        gradient_clip=cfg.gradient_clip,
        min_steps=cfg.attention_min_steps,
        max_steps_per_window=(
            cfg.attention_max_steps_per_window
        ),
        max_traces=cfg.expert_max_traces,
        training_horizon_minutes=(
            cfg.training_horizon_minutes
        ),
        device=cfg.device,
        seed=cfg.attention_seed,
    )


def _build_recurrent(
    cfg: MarketPerceptionConfig,
) -> RecurrentMarketExpert:
    return RecurrentMarketExpert(
        representation_size=(
            cfg.expert_representation_size
        ),
        hidden_size=cfg.recurrent_hidden_size,
        branch_size=cfg.recurrent_branch_size,
        depth=cfg.recurrent_depth,
        dropout=cfg.recurrent_dropout,
        learning_rate=cfg.expert_learning_rate,
        gradient_clip=cfg.gradient_clip,
        min_steps=cfg.recurrent_min_steps,
        max_steps_per_window=(
            cfg.recurrent_max_steps_per_window
        ),
        max_traces=cfg.expert_max_traces,
        training_horizon_minutes=(
            cfg.training_horizon_minutes
        ),
        device=cfg.device,
        seed=cfg.recurrent_seed,
    )


def _build_deeplob(
    cfg: MarketPerceptionConfig,
) -> DeepLOBMarketExpert:
    return DeepLOBMarketExpert(
        representation_size=(
            cfg.expert_representation_size
        ),
        levels=cfg.deeplob_levels,
        conv_channels=cfg.deeplob_conv_channels,
        inception_channels=(
            cfg.deeplob_inception_channels
        ),
        recurrent_size=(
            cfg.deeplob_recurrent_size
        ),
        recurrent_layers=(
            cfg.deeplob_recurrent_layers
        ),
        dropout=cfg.deeplob_dropout,
        learning_rate=cfg.expert_learning_rate,
        gradient_clip=cfg.gradient_clip,
        min_snapshots=cfg.deeplob_min_snapshots,
        max_snapshots=cfg.deeplob_max_snapshots,
        max_traces=cfg.deeplob_max_traces,
        training_horizon_minutes=(
            cfg.training_horizon_minutes
        ),
        device=cfg.device,
        seed=cfg.deeplob_seed,
    )

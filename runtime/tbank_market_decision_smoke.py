from __future__ import annotations

import os
import sys
import time

import torch

from core.personal import PersonalCognitiveCore
from policy.dynamic import DynamicAffordancePolicy
from runtime.cycle import MarketCognitionCycle
from runtime.market_decision import MarketDecisionCycle
from worlds.market.affordances import MarketAffordanceProvider
from worlds.market.input_builder import MarketInputBuilder
from worlds.market.perception.defaults import build_default_market_organ
from worlds.market.tbank.sandbox import (
    TBankRestError,
    TBankSandboxObservationBackend,
    TBankSandboxObservationConfig,
)
from worlds.market.tbank.sandbox_capabilities import (
    TBankSandboxCapabilityError,
    TBankSandboxCapabilityReader,
)


def _instrument_ids() -> tuple[str, ...]:
    raw = (
        os.environ.get("TBANK_INSTRUMENT_IDS", "").strip()
        or os.environ.get("TBANK_INSTRUMENT_ID", "").strip()
    )
    result = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not result:
        raise RuntimeError("set TBANK_INSTRUMENT_ID, for example SBER_TQBR")
    return result


def _resolve_instrument(backend: TBankSandboxObservationBackend, requested_id: str):
    by_id = {item.instrument_id: item for item in backend.instruments}
    if requested_id in by_id:
        return by_id[requested_id]
    if len(backend.instruments) == 1:
        return backend.instruments[0]
    raise RuntimeError("requested T-Bank instrument was not resolved uniquely")


def _latest_trading_status(events, *, requested_id: str, canonical_id: str):
    matches = [
        event
        for event in events
        if event.kind == "trading_status"
        and event.instrument_id in {requested_id, canonical_id}
    ]
    return max(matches, key=lambda event: event.observed_at) if matches else None


def run() -> None:
    if not os.environ.get("TBANK_INVEST_TOKEN", "").strip():
        raise RuntimeError("TBANK_INVEST_TOKEN is missing")

    instrument_ids = _instrument_ids()
    requested_id = instrument_ids[0]

    backend = TBankSandboxObservationBackend(
        TBankSandboxObservationConfig(instrument_ids=instrument_ids)
    )
    capabilities = TBankSandboxCapabilityReader()
    builder = MarketInputBuilder()
    organ = build_default_market_organ()
    core = PersonalCognitiveCore()
    policy = DynamicAffordancePolicy()
    provider = MarketAffordanceProvider()
    decision_cycle = MarketDecisionCycle(MarketCognitionCycle(core), policy)

    print("TBANK -> MARKET -> CORE -> AFFORDANCES -> POLICY SMOKE")
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("accelerator_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("accelerator:", torch.cuda.get_device_name(0))

    try:
        backend.connect()
        capabilities.connect()
        instrument = _resolve_instrument(backend, requested_id)

        latest_status = None
        last_frame = None

        for poll_index in range(14):
            frame = backend.poll(time.time())
            builder.ingest_frame(frame)
            last_frame = frame

            status = _latest_trading_status(
                frame.events,
                requested_id=requested_id,
                canonical_id=instrument.instrument_id,
            )
            if status is not None:
                latest_status = status

            if poll_index + 1 < 14:
                time.sleep(1.1)

        if last_frame is None or latest_status is None:
            raise RuntimeError("no complete real T-Bank decision reached ActionIntent")

        market_input = builder.build(
            requested_id,
            as_of=last_frame.observed_at,
            metadata={
                "source": "tbank:sandbox",
                "real_market_data": True,
                "runtime_policy_smoke": True,
            },
        )

        steps = {
            "micro": len(market_input.micro.rows) if market_input.micro else 0,
            "short": len(market_input.short.rows) if market_input.short else 0,
            "medium": len(market_input.medium.rows) if market_input.medium else 0,
            "long": len(market_input.long.rows) if market_input.long else 0,
        }
        builder_telemetry = builder.telemetry()

        print(
            "windows:",
            " ".join(
                f"{name}={count}"
                for name, count in steps.items()
            ),
        )
        print("l2_snapshots:", len(market_input.order_book_history))
        print(
            "builder:",
            f"seen={builder_telemetry.events_seen}",
            f"used={builder_telemetry.events_used}",
            f"ignored={builder_telemetry.events_ignored}",
            f"stale={builder_telemetry.stale_temporal_events}",
        )

        transition_id = (
            "tbank:sandbox:market-policy:"
            f"{requested_id}:{last_frame.observed_at:.6f}:14"
        )

        try:
            representation = organ.encode(
                market_input,
                transition_id=transition_id,
                shared_context=None,
            )
        except Exception as exc:
            if type(exc).__name__ != "MarketPerceptionUnavailable":
                raise
            print("PERCEPTION_UNAVAILABLE_REAL_DATA")
            raise

        if len(representation.vector) != 96:
            raise RuntimeError("MarketRepresentation size is not 96")

        available_experts = {
            name
            for name, is_available in representation.expert_available.items()
            if is_available
        }
        if not available_experts:
            raise RuntimeError("no Market expert is available")

        payload = latest_status.payload
        api_trade_available = bool(payload.get("api_trade_available", False))
        market_order_available = bool(
            payload.get("market_order_available", False)
        )
        limit_order_available = bool(
            payload.get("limit_order_available", False)
        )
        trading_open = api_trade_available and (
            market_order_available or limit_order_available
        )

        limits = capabilities.get_trade_limits(requested_id)
        affordances = provider.build_from_limits(
            instrument,
            limits,
            world_id="market:tbank:sandbox",
            now=last_frame.observed_at,
            trading_open=trading_open,
        )
        decision = decision_cycle.process(
            representation,
            affordances,
            now=last_frame.observed_at,
            remember_core_for_learning=True,
        )

        if decision.transition_id != transition_id:
            raise RuntimeError("MarketDecision lost transition_id")
        if decision.cognitive_state.transition_id != transition_id:
            raise RuntimeError("Core lost transition_id")
        if decision.intent.transition_id != transition_id:
            raise RuntimeError("Policy lost transition_id")
        if len(decision.cognitive_state.vector) != 96:
            raise RuntimeError("IntegratedCognitiveState size is not 96")

        offered = {item.id: item for item in affordances}
        chosen = offered.get(decision.intent.affordance_id)
        if chosen is None:
            raise RuntimeError(
                "Policy chose an Affordance that was not offered"
            )
        if chosen.action != decision.intent.action or chosen.world != decision.intent.world:
            raise RuntimeError("ActionIntent does not match chosen Affordance")
        if decision.intent.parameters:
            raise RuntimeError(
                "Policy fabricated action parameters before parameterization"
            )

        core_telemetry = core.telemetry()
        policy_telemetry = policy.telemetry()

        if core_telemetry.learning_updates != 0:
            raise RuntimeError("Core learned without a real Consequence")
        if policy_telemetry.learning_updates != 0:
            raise RuntimeError("Policy learned without a real Consequence")
        if core_telemetry.traces_created < 1:
            raise RuntimeError("Core did not preserve a causal trace")
        if policy_telemetry.traces_created < 1:
            raise RuntimeError("Policy did not preserve a causal trace")

        actions = tuple(item.action for item in affordances)
        status_payload = latest_status.payload

        print("instrument_requested:", requested_id)
        print("instrument_canonical:", instrument.instrument_id)
        print("market_representation_size:", len(representation.vector))
        print("available_experts:", ", ".join(sorted(available_experts)))
        print("core_state_size:", len(decision.cognitive_state.vector))
        print("trading_status:", status_payload.get("trading_status", ""))
        print("api_trade_available:", bool(status_payload.get("api_trade_available", False)))
        print(
            "market_order_available:",
            bool(status_payload.get("market_order_available", False)),
        )
        print(
            "limit_order_available:",
            bool(status_payload.get("limit_order_available", False)),
        )
        print(
            "broker_limits:",
            f"buy={limits.buy_max_lots}",
            f"buy_market={limits.buy_max_market_lots}",
            f"sell={limits.sell_max_lots}",
        )
        print("affordances:", ", ".join(actions))
        print("chosen_action:", decision.intent.action)
        print("intent_parameters:", dict(decision.intent.parameters))
        print("transition_preserved: OK")
        print("real_affordances_to_policy: OK")
        print("causal_traces_created: OK")
        print("no_fake_learning: OK")
        print("NO EXECUTION ATTEMPTED")
        print(
            "CHAIN: T-Bank observation -> MarketInput -> MarketRepresentation[96] "
            "-> Core[96] -> real T-Bank Affordances -> trainable Policy -> ActionIntent"
        )
        print("TBANK -> MARKET -> CORE -> AFFORDANCES -> POLICY SMOKE: OK")

        policy.discard_trace(decision.transition_id)
        core.discard_trace(decision.transition_id)

    finally:
        try:
            organ.reset()
        finally:
            try:
                capabilities.close()
            finally:
                backend.close()


def main() -> int:
    try:
        run()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (TBankRestError, TBankSandboxCapabilityError) as exc:
        print("TBANK ERROR:", str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "TBANK -> MARKET -> CORE -> AFFORDANCES -> POLICY SMOKE FAILED:",
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

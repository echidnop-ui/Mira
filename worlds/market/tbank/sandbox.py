from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import socket
import ssl
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worlds.market.interface import (
    MarketBackendFrame,
    MarketCancelRequest,
    MarketCancelSubmission,
    MarketInstrument,
    MarketMode,
    MarketOrderRequest,
    MarketOrderSubmission,
)
from worlds.market.tbank.normalize import (
    normalize_candles,
    normalize_instrument,
    normalize_orderbook,
    normalize_trades,
    normalize_trading_status,
)


_RUSSIAN_TRUSTED_ROOT_CA = Path(__file__).with_name(
    "russian_trusted_root_ca.pem"
)


# ============================================================
# Public configuration / errors
# ============================================================


class TBankRestError(RuntimeError):
    """
    Transport/API failure without exposing the bearer token.
    """


@dataclass(frozen=True, slots=True)
class TBankSandboxObservationConfig:
    """
    Read-only first T-Bank vertical slice.

    `instrument_ids` accepts the same universal identifiers documented by
    T-Invest API market-data methods, for example:

        instrument UID
        FIGI
        ticker + "_" + class_code

    Example for a MOEX share when its class code is known:
        "SBER_TQBR"

    This class intentionally contains no account id and no execution settings.
    """

    instrument_ids: tuple[str, ...]

    base_url: str = (
        "https://sandbox-invest-public-api.tbank.ru/rest"
    )

    token_env: str = "TBANK_INVEST_TOKEN"

    request_timeout_seconds: float = 20.0

    orderbook_depth: int = 10

    # The backend is polling in this first slice. Streaming can replace this
    # transport later without changing MarketBackendEvent / MarketInputBuilder.
    minimum_poll_interval_seconds: float = 1.0

    # Warmup windows.
    trade_warmup_seconds: float = 3600.0
    short_candle_warmup_seconds: float = 3600.0
    minute_candle_warmup_seconds: float = 4.0 * 3600.0

    # Incremental refresh cadences.
    short_candle_refresh_seconds: float = 5.0
    minute_candle_refresh_seconds: float = 60.0
    trading_status_refresh_seconds: float = 30.0

    max_seen_event_ids: int = 100_000

    def __post_init__(self) -> None:
        cleaned = tuple(
            item.strip()
            for item in self.instrument_ids
            if item and item.strip()
        )

        if cleaned != self.instrument_ids:
            object.__setattr__(
                self,
                "instrument_ids",
                cleaned,
            )

        if not self.instrument_ids:
            raise ValueError(
                "TBankSandboxObservationConfig.instrument_ids "
                "must not be empty"
            )

        if len(set(self.instrument_ids)) != len(
            self.instrument_ids
        ):
            raise ValueError(
                "T-Bank instrument_ids must be unique"
            )

        if self.request_timeout_seconds <= 0.0:
            raise ValueError(
                "request_timeout_seconds must be > 0"
            )

        if self.orderbook_depth <= 0:
            raise ValueError(
                "orderbook_depth must be > 0"
            )

        if self.minimum_poll_interval_seconds < 0.0:
            raise ValueError(
                "minimum_poll_interval_seconds must be >= 0"
            )

        if self.trade_warmup_seconds <= 0.0:
            raise ValueError(
                "trade_warmup_seconds must be > 0"
            )

        if not (
            0.0
            < self.short_candle_warmup_seconds
            <= 200.0 * 60.0
        ):
            # Current T-Invest GetCandles documentation limits 5-second
            # candles to about 200 minutes.
            raise ValueError(
                "short_candle_warmup_seconds must be in "
                "(0, 12000]"
            )

        if self.minute_candle_warmup_seconds <= 0.0:
            raise ValueError(
                "minute_candle_warmup_seconds must be > 0"
            )

        if self.max_seen_event_ids <= 0:
            raise ValueError(
                "max_seen_event_ids must be > 0"
            )


@dataclass(frozen=True, slots=True)
class TBankSandboxObservationTelemetry:
    connected: bool
    polls: int
    requests: int
    emitted_events: int
    duplicate_events: int
    instruments: int
    last_poll_at: float | None
    last_error: str | None


# ============================================================
# Read-only sandbox backend
# ============================================================


class TBankSandboxObservationBackend:
    """
    First real T-Bank input adapter for Heiwa v2.

    Responsibility:

        T-Bank sandbox REST
        -> broker-neutral MarketBackendFrame/Event

    It deliberately does NOT:

        decide BUY/SELL
        create ActionIntent
        execute an order
        access Core internals
        access neural expert internals
        build MarketInput directly

    MarketInputBuilder remains the next boundary.

    Why REST here
    -------------
    T-Invest exposes an official REST proxy in addition to gRPC. Using the
    Python standard library keeps this first observation slice independent of
    a broker SDK build and, importantly for the current Heiwa environment,
    independent of Python/ROCm package compatibility.

    Streaming can replace polling later. That should be a transport change,
    not a cognitive architecture change.
    """

    _BACKEND_ID = "tbank:sandbox:observation"

    def __init__(
        self,
        config: TBankSandboxObservationConfig,
        *,
        token: str | None = None,
    ) -> None:
        self.config = config

        self._ssl_context = ssl.create_default_context()

        try:
            self._ssl_context.load_verify_locations(
                cafile=str(
                    _RUSSIAN_TRUSTED_ROOT_CA
                )
            )
        except (
            OSError,
            ssl.SSLError,
        ) as exc:
            raise RuntimeError(
                "T-Bank trusted root CA could not be loaded"
            ) from exc

        self._explicit_token = (
            token.strip()
            if token is not None
            else None
        )

        self._token: str | None = None

        self._connected = False
        self._closed = False

        self._instruments: tuple[
            MarketInstrument,
            ...,
        ] = ()

        self._emit_instruments_next_poll = False
        self._warmed_up = False

        self._last_poll_at: float | None = None
        self._last_short_candle_fetch_at: float | None = None
        self._last_minute_candle_fetch_at: float | None = None
        self._last_status_fetch_at: float | None = None

        self._seen_ids: set[
            str
        ] = set()

        self._seen_order: deque[
            str
        ] = deque()

        self._polls = 0
        self._requests = 0
        self._emitted_events = 0
        self._duplicate_events = 0
        self._last_error: str | None = None

    # --------------------------------------------------------
    # Market backend identity
    # --------------------------------------------------------

    @property
    def backend_id(
        self,
    ) -> str:
        return self._BACKEND_ID

    @property
    def mode(
        self,
    ) -> MarketMode:
        return MarketMode.SANDBOX

    @property
    def connected(
        self,
    ) -> bool:
        return self._connected

    @property
    def instruments(
        self,
    ) -> tuple[
        MarketInstrument,
        ...,
    ]:
        return self._instruments

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def connect(
        self,
    ) -> None:
        if self._closed:
            raise RuntimeError(
                "T-Bank observation backend is closed"
            )

        if self._connected:
            return

        token = self._resolve_token()

        # Fetch full normalized instrument descriptions. This doubles as an
        # authenticated connectivity check before runtime polling starts.
        instruments: list[
            MarketInstrument
        ] = []

        try:
            for instrument_id in (
                self.config.instrument_ids
            ):
                response = self._post(
                    (
                        "tinkoff.public.invest.api.contract.v1."
                        "InstrumentsService/GetInstrumentBy"
                    ),
                    {
                        "idType": (
                            "INSTRUMENT_ID_TYPE_ID"
                        ),
                        "id": instrument_id,
                    },
                    token=token,
                )

                instruments.append(
                    normalize_instrument(
                        response,
                        requested_id=instrument_id,
                    )
                )

        except Exception as exc:
            self._last_error = str(exc)
            raise

        self._token = token
        self._instruments = tuple(
            instruments
        )
        self._emit_instruments_next_poll = True
        self._connected = True
        self._last_error = None

    def close(
        self,
    ) -> None:
        self._connected = False
        self._closed = True
        self._token = None

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    def poll(
        self,
        now: float,
    ) -> MarketBackendFrame:
        """
        Read new sandbox market observations.

        `now` is supplied by Runtime so wall-clock ownership stays outside the
        backend. API requests themselves use UTC RFC3339 derived from it.
        """

        if not self._connected:
            self.connect()

        if not isfinite(
            now
        ):
            raise ValueError(
                "poll now must be finite"
            )

        now = float(
            now
        )

        if (
            self._last_poll_at is not None
            and now
            < self._last_poll_at
        ):
            raise ValueError(
                "T-Bank backend poll time moved backwards"
            )

        instruments_for_frame = (
            self._instruments
            if self._emit_instruments_next_poll
            else ()
        )

        if (
            self._last_poll_at is not None
            and (
                now - self._last_poll_at
                < self.config.minimum_poll_interval_seconds
            )
        ):
            self._emit_instruments_next_poll = False

            return MarketBackendFrame(
                backend_id=self.backend_id,
                mode=self.mode,
                observed_at=now,
                events=(),
                instruments=instruments_for_frame,
            )

        try:
            if not self._warmed_up:
                events = self._warmup(
                    now
                )
                self._warmed_up = True
            else:
                events = self._incremental_poll(
                    now
                )

            unique = self._deduplicate(
                events
            )

        except Exception as exc:
            self._last_error = str(exc)
            raise

        self._polls += 1
        self._last_poll_at = now
        self._emit_instruments_next_poll = False
        self._emitted_events += len(
            unique
        )
        self._last_error = None

        return MarketBackendFrame(
            backend_id=self.backend_id,
            mode=self.mode,
            observed_at=now,
            events=tuple(
                sorted(
                    unique,
                    key=lambda event: (
                        event.observed_at,
                        event.id,
                    ),
                )
            ),
            instruments=instruments_for_frame,
        )

    # --------------------------------------------------------
    # Execution is intentionally absent in this slice
    # --------------------------------------------------------

    def submit_order(
        self,
        request: MarketOrderRequest,
    ) -> MarketOrderSubmission:
        raise RuntimeError(
            "TBankSandboxObservationBackend is read-only. "
            "Sandbox execution must be added through the explicit "
            "ActionIntent -> MarketOrderRequest -> submission path."
        )

    def cancel_order(
        self,
        request: MarketCancelRequest,
    ) -> MarketCancelSubmission:
        raise RuntimeError(
            "TBankSandboxObservationBackend is read-only. "
            "Order cancellation belongs to the later sandbox "
            "execution slice."
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def telemetry(
        self,
    ) -> TBankSandboxObservationTelemetry:
        return TBankSandboxObservationTelemetry(
            connected=self._connected,
            polls=self._polls,
            requests=self._requests,
            emitted_events=self._emitted_events,
            duplicate_events=self._duplicate_events,
            instruments=len(
                self._instruments
            ),
            last_poll_at=self._last_poll_at,
            last_error=self._last_error,
        )

    # ========================================================
    # Warmup
    # ========================================================

    def _warmup(
        self,
        now: float,
    ) -> list:
        events = []

        for instrument_id in (
            self.config.instrument_ids
        ):
            # Individual trades give genuinely fine timestamps and may seed
            # micro/short/medium/long without inventing finer candle data.
            events.extend(
                self._get_trades(
                    instrument_id,
                    start=max(
                        0.0,
                        now
                        - self.config.trade_warmup_seconds,
                    ),
                    end=now,
                )
            )

            # 5-second candles may seed short and coarser views but NOT the
            # 1-second micro view. MarketInputBuilder enforces that invariant.
            events.extend(
                self._get_candles(
                    instrument_id,
                    start=max(
                        0.0,
                        now
                        - self.config.short_candle_warmup_seconds,
                    ),
                    end=now,
                    interval=(
                        "CANDLE_INTERVAL_5_SEC"
                    ),
                    interval_seconds=5.0,
                )
            )

            # Minute candles cheaply establish enough medium/long history.
            events.extend(
                self._get_candles(
                    instrument_id,
                    start=max(
                        0.0,
                        now
                        - self.config.minute_candle_warmup_seconds,
                    ),
                    end=now,
                    interval=(
                        "CANDLE_INTERVAL_1_MIN"
                    ),
                    interval_seconds=60.0,
                )
            )

            orderbook = self._get_orderbook(
                instrument_id,
                now=now,
            )

            if orderbook is not None:
                events.append(
                    orderbook
                )

            events.append(
                self._get_trading_status(
                    instrument_id,
                    now=now,
                )
            )

        self._last_short_candle_fetch_at = now
        self._last_minute_candle_fetch_at = now
        self._last_status_fetch_at = now

        return events

    # ========================================================
    # Incremental polling
    # ========================================================

    def _incremental_poll(
        self,
        now: float,
    ) -> list:
        events = []

        previous = (
            self._last_poll_at
            if self._last_poll_at is not None
            else now
        )

        # Small overlap protects against boundary timestamp races. Stable event
        # ids remove duplicates before the frame crosses the boundary.
        trade_start = max(
            0.0,
            now - 3600.0,
            previous - 2.0,
        )

        refresh_short = self._due(
            now,
            self._last_short_candle_fetch_at,
            self.config.short_candle_refresh_seconds,
        )

        refresh_minute = self._due(
            now,
            self._last_minute_candle_fetch_at,
            self.config.minute_candle_refresh_seconds,
        )

        refresh_status = self._due(
            now,
            self._last_status_fetch_at,
            self.config.trading_status_refresh_seconds,
        )

        for instrument_id in (
            self.config.instrument_ids
        ):
            events.extend(
                self._get_trades(
                    instrument_id,
                    start=trade_start,
                    end=now,
                )
            )

            orderbook = self._get_orderbook(
                instrument_id,
                now=now,
            )

            if orderbook is not None:
                events.append(
                    orderbook
                )

            if refresh_short:
                events.extend(
                    self._get_candles(
                        instrument_id,
                        start=max(
                            0.0,
                            now - 20.0,
                        ),
                        end=now,
                        interval=(
                            "CANDLE_INTERVAL_5_SEC"
                        ),
                        interval_seconds=5.0,
                    )
                )

            if refresh_minute:
                events.extend(
                    self._get_candles(
                        instrument_id,
                        start=max(
                            0.0,
                            now - 180.0,
                        ),
                        end=now,
                        interval=(
                            "CANDLE_INTERVAL_1_MIN"
                        ),
                        interval_seconds=60.0,
                    )
                )

            if refresh_status:
                events.append(
                    self._get_trading_status(
                        instrument_id,
                        now=now,
                    )
                )

        if refresh_short:
            self._last_short_candle_fetch_at = now

        if refresh_minute:
            self._last_minute_candle_fetch_at = now

        if refresh_status:
            self._last_status_fetch_at = now

        return events

    # ========================================================
    # T-Invest REST methods
    # ========================================================

    def _get_candles(
        self,
        instrument_id: str,
        *,
        start: float,
        end: float,
        interval: str,
        interval_seconds: float,
    ):
        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "MarketDataService/GetCandles"
            ),
            {
                "from": self._rfc3339(
                    start
                ),
                "to": self._rfc3339(
                    end
                ),
                "interval": interval,
                "instrumentId": instrument_id,
                "candleSourceType": (
                    "CANDLE_SOURCE_EXCHANGE"
                ),
            },
        )

        return normalize_candles(
            response,
            instrument_id=instrument_id,
            interval_seconds=interval_seconds,
        )

    def _get_trades(
        self,
        instrument_id: str,
        *,
        start: float,
        end: float,
    ):
        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "MarketDataService/GetLastTrades"
            ),
            {
                "from": self._rfc3339(
                    start
                ),
                "to": self._rfc3339(
                    end
                ),
                "instrumentId": instrument_id,
                "tradeSource": (
                    "TRADE_SOURCE_EXCHANGE"
                ),
            },
        )

        return normalize_trades(
            response,
            instrument_id=instrument_id,
        )

    def _get_orderbook(
        self,
        instrument_id: str,
        *,
        now: float,
    ):
        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "MarketDataService/GetOrderBook"
            ),
            {
                "depth": int(
                    self.config.orderbook_depth
                ),
                "instrumentId": instrument_id,
            },
        )

        return normalize_orderbook(
            response,
            instrument_id=instrument_id,
            fallback_observed_at=now,
        )

    def _get_trading_status(
        self,
        instrument_id: str,
        *,
        now: float,
    ):
        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "MarketDataService/GetTradingStatus"
            ),
            {
                "instrumentId": instrument_id,
            },
        )

        return normalize_trading_status(
            response,
            instrument_id=instrument_id,
            observed_at=now,
        )

    # ========================================================
    # HTTP
    # ========================================================

    def _post(
        self,
        method_path: str,
        body: Mapping[str, object],
        *,
        token: str | None = None,
    ) -> Mapping[str, object]:
        resolved_token = (
            token
            if token is not None
            else self._token
        )

        if not resolved_token:
            raise RuntimeError(
                "T-Bank token is not initialized"
            )

        url = (
            self.config.base_url.rstrip("/")
            + "/"
            + method_path.lstrip("/")
        )

        encoded = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        request = Request(
            url,
            data=encoded,
            method="POST",
            headers={
                "Authorization": (
                    f"Bearer {resolved_token}"
                ),
                "Content-Type": (
                    "application/json"
                ),
                "Accept": (
                    "application/json"
                ),
                "User-Agent": (
                    "HeiwaV2-Market/0.1"
                ),
            },
        )

        self._requests += 1

        try:
            with urlopen(
                request,
                timeout=(
                    self.config.request_timeout_seconds
                ),
                context=self._ssl_context,
            ) as response:
                raw = response.read()

        except HTTPError as exc:
            try:
                error_body = (
                    exc.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except Exception:
                error_body = ""

            message = (
                f"T-Bank HTTP {exc.code} for "
                f"{method_path}"
            )

            if error_body:
                # Body may contain request diagnostics but never our
                # Authorization header. Limit length for terminal readability.
                message += (
                    ": "
                    + error_body[:1000]
                )

            raise TBankRestError(
                message
            ) from exc

        except (
            URLError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise TBankRestError(
                f"T-Bank transport failure for "
                f"{method_path}: {exc}"
            ) from exc

        if not raw:
            return {}

        try:
            decoded = json.loads(
                raw.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise TBankRestError(
                f"T-Bank returned invalid JSON for "
                f"{method_path}"
            ) from exc

        if not isinstance(
            decoded,
            Mapping,
        ):
            raise TBankRestError(
                f"T-Bank returned non-object JSON for "
                f"{method_path}"
            )

        return decoded

    # ========================================================
    # Event dedupe
    # ========================================================

    def _deduplicate(
        self,
        events: Sequence,
    ) -> list:
        unique = []

        for event in events:
            if event.id in self._seen_ids:
                self._duplicate_events += 1
                continue

            self._seen_ids.add(
                event.id
            )
            self._seen_order.append(
                event.id
            )
            unique.append(
                event
            )

            while (
                len(
                    self._seen_order
                )
                > self.config.max_seen_event_ids
            ):
                expired = (
                    self._seen_order.popleft()
                )
                self._seen_ids.discard(
                    expired
                )

        return unique

    # ========================================================
    # Helpers
    # ========================================================

    def _resolve_token(
        self,
    ) -> str:
        token = (
            self._explicit_token
            or os.environ.get(
                self.config.token_env,
                "",
            ).strip()
        )

        if not token:
            raise RuntimeError(
                "T-Bank API token is missing. "
                f"Set {self.config.token_env} in the shell "
                "or pass token= explicitly."
            )

        return token

    @staticmethod
    def _rfc3339(
        timestamp: float,
    ) -> str:
        dt = datetime.fromtimestamp(
            float(timestamp),
            tz=timezone.utc,
        )

        return (
            dt.isoformat(
                timespec="milliseconds"
            )
            .replace(
                "+00:00",
                "Z",
            )
        )

    @staticmethod
    def _due(
        now: float,
        previous: float | None,
        interval: float,
    ) -> bool:
        return (
            previous is None
            or now - previous >= interval
        )

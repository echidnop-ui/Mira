from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import ssl
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from worlds.market.affordances import MarketTradeLimits
from worlds.market.tbank.normalize import (
    int_field,
    mapping_field,
    sequence_field,
    string_field,
)


_RUSSIAN_TRUSTED_ROOT_CA = (
    Path(__file__).with_name("certs")
    / "russian_trusted_root_ca.crt"
)


class TBankSandboxCapabilityError(RuntimeError):
    """Read-only sandbox capability/account-state failure."""


@dataclass(frozen=True, slots=True)
class TBankSandboxCapabilityConfig:
    """
    Read-only account/capability configuration.

    No account is opened, funded, closed or mutated here.
    """

    base_url: str = (
        "https://invest-public-api.tbank.ru/rest"
    )
    token_env: str = "TBANK_INVEST_TOKEN"
    account_id_env: str = (
        "TBANK_SANDBOX_ACCOUNT_ID"
    )
    request_timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0.0:
            raise ValueError(
                "request_timeout_seconds must be > 0"
            )


class TBankSandboxCapabilityReader:
    """
    Read current sandbox account trade limits without executing anything.

    Responsibilities:
        token/account resolution
        -> GetSandboxMaxLots
        -> broker-neutral MarketTradeLimits

    It deliberately does NOT:
        create ActionIntent
        choose BUY/SELL/HOLD
        submit/cancel orders
        create/fund/close sandbox accounts
    """

    def __init__(
        self,
        config: TBankSandboxCapabilityConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else TBankSandboxCapabilityConfig()
        )

        self._ssl_context = ssl.create_default_context()
        self._ssl_context.load_verify_locations(
            cafile=str(_RUSSIAN_TRUSTED_ROOT_CA)
        )

        self._token: str | None = None
        self._account_id: str | None = None
        self._connected = False
        self._requests = 0

    @property
    def connected(
        self,
    ) -> bool:
        return self._connected

    @property
    def account_resolved(
        self,
    ) -> bool:
        return bool(
            self._account_id
        )

    @property
    def requests(
        self,
    ) -> int:
        return self._requests

    def connect(
        self,
    ) -> None:
        if self._connected:
            return

        token = os.environ.get(
            self.config.token_env,
            "",
        ).strip()

        if not token:
            raise TBankSandboxCapabilityError(
                f"Missing T-Bank token in "
                f"{self.config.token_env}"
            )

        configured_account = os.environ.get(
            self.config.account_id_env,
            "",
        ).strip()

        self._token = token

        try:
            if configured_account:
                account_id = configured_account
            else:
                account_id = (
                    self._discover_single_open_account()
                )
        except Exception:
            self._token = None
            raise

        self._account_id = account_id
        self._connected = True

    def close(
        self,
    ) -> None:
        self._connected = False
        self._token = None
        self._account_id = None

    def get_trade_limits(
        self,
        instrument_id: str,
    ) -> MarketTradeLimits:
        if not self._connected:
            self.connect()

        instrument_id = instrument_id.strip()
        if not instrument_id:
            raise ValueError(
                "instrument_id must not be empty"
            )

        assert self._account_id is not None

        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "SandboxService/GetSandboxMaxLots"
            ),
            {
                "accountId": self._account_id,
                "instrumentId": instrument_id,
                # Deliberately omit `price`: T-Bank documents that the
                # calculation then uses current order-book prices.
            },
        )

        buy_limits = mapping_field(
            response,
            "buyLimits",
            "buy_limits",
        )
        sell_limits = mapping_field(
            response,
            "sellLimits",
            "sell_limits",
        )

        return MarketTradeLimits(
            currency=string_field(
                response,
                "currency",
                default="",
            ),
            buy_max_lots=max(
                0,
                int_field(
                    buy_limits,
                    "buyMaxLots",
                    "buy_max_lots",
                    default=0,
                ),
            ),
            buy_max_market_lots=max(
                0,
                int_field(
                    buy_limits,
                    "buyMaxMarketLots",
                    "buy_max_market_lots",
                    default=0,
                ),
            ),
            sell_max_lots=max(
                0,
                int_field(
                    sell_limits,
                    "sellMaxLots",
                    "sell_max_lots",
                    default=0,
                ),
            ),
        )

    # ========================================================
    # Account resolution
    # ========================================================

    def _discover_single_open_account(
        self,
    ) -> str:
        response = self._post(
            (
                "tinkoff.public.invest.api.contract.v1."
                "SandboxService/GetSandboxAccounts"
            ),
            {
                "status": (
                    "ACCOUNT_STATUS_OPEN"
                ),
            },
        )

        open_ids: list[str] = []

        for raw in sequence_field(
            response,
            "accounts",
        ):
            if not isinstance(
                raw,
                Mapping,
            ):
                continue

            account_id = string_field(
                raw,
                "id",
                default="",
            ).strip()
            status = string_field(
                raw,
                "status",
                default="",
            ).strip()

            if (
                account_id
                and (
                    not status
                    or status
                    == "ACCOUNT_STATUS_OPEN"
                )
            ):
                open_ids.append(
                    account_id
                )

        # Defensive de-duplication while preserving response order.
        open_ids = list(
            dict.fromkeys(
                open_ids
            )
        )

        if not open_ids:
            raise TBankSandboxCapabilityError(
                "No open T-Bank sandbox account found. "
                "This reader will not create or fund an account automatically."
            )

        if len(open_ids) > 1:
            raise TBankSandboxCapabilityError(
                "Multiple open T-Bank sandbox accounts found. "
                f"Set {self.config.account_id_env} explicitly so Heiwa does "
                "not silently choose the wrong sandbox account."
            )

        return open_ids[0]

    # ========================================================
    # HTTP
    # ========================================================

    def _post(
        self,
        method_path: str,
        body: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not self._token:
            raise TBankSandboxCapabilityError(
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
                    f"Bearer {self._token}"
                ),
                "Content-Type": "application/json",
                "Accept": "application/json",
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
                message += (
                    ": "
                    + error_body[:1000]
                )

            raise TBankSandboxCapabilityError(
                message
            ) from exc
        except (
            URLError,
            TimeoutError,
            socket.timeout,
        ) as exc:
            raise TBankSandboxCapabilityError(
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
            raise TBankSandboxCapabilityError(
                f"T-Bank returned invalid JSON for "
                f"{method_path}"
            ) from exc

        if not isinstance(
            decoded,
            Mapping,
        ):
            raise TBankSandboxCapabilityError(
                f"T-Bank returned non-object JSON for "
                f"{method_path}"
            )

        return decoded

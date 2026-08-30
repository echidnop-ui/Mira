from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import http.client
import json
import os
import socket
import ssl
from typing import Iterable, Mapping
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from worlds.market.interface import (
    MarketBackendEvent,
    MarketBackendFrame,
    MarketMode,
)


_PUBLIC_ISS = "https://iss.moex.com/iss"
_AUTH_ISS = "https://apim.moex.com/iss"
_MOSCOW = ZoneInfo("Europe/Moscow")
_VIRTUAL_INTERFACE_PREFIXES = (
    "tun",
    "tap",
    "vpn",
    "wg",
    "tailscale",
    "ppp",
)


class _InterfaceBoundHTTPSConnection(
    http.client.HTTPSConnection
):
    def __init__(
        self,
        host: str,
        *,
        interface: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(
            host,
            timeout=timeout,
            context=context,
        )
        self._interface = interface
        self._create_connection = (
            self._create_interface_bound_connection
        )

    def _create_interface_bound_connection(
        self,
        address: tuple[str, int],
        timeout: float,
        source_address=None,
    ) -> socket.socket:
        host, port = address
        last_error: OSError | None = None

        for family, socktype, proto, _, socket_address in (
            socket.getaddrinfo(
                host,
                port,
                socket.AF_INET,
                socket.SOCK_STREAM,
            )
        ):
            connection = socket.socket(
                family,
                socktype,
                proto,
            )

            try:
                connection.settimeout(timeout)
                connection.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_BINDTODEVICE,
                    self._interface.encode("utf-8") + b"\0",
                )

                if source_address:
                    connection.bind(source_address)

                connection.connect(socket_address)
                return connection
            except OSError as exc:
                last_error = exc
                connection.close()

        if last_error is not None:
            raise last_error

        raise OSError(
            f"no IPv4 address resolved for {host!r}"
        )


def _physical_default_interface() -> str | None:
    explicit = os.environ.get(
        "MOEX_NETWORK_INTERFACE",
        "",
    ).strip()
    if explicit:
        return explicit

    try:
        with open(
            "/proc/net/route",
            "r",
            encoding="ascii",
        ) as routes:
            lines = routes.readlines()[1:]
    except OSError:
        return None

    candidates: list[tuple[int, str]] = []

    for line in lines:
        fields = line.split()
        if len(fields) < 8:
            continue

        interface = fields[0]
        destination = fields[1]

        if destination != "00000000":
            continue

        if interface.casefold().startswith(
            _VIRTUAL_INTERFACE_PREFIXES
        ):
            continue

        try:
            flags = int(fields[3], 16)
            metric = int(fields[6])
        except ValueError:
            continue

        if flags & 0x3 != 0x3:
            continue

        candidates.append((metric, interface))

    if not candidates:
        return None

    return min(candidates)[1]


def _read_bound_https(
    request: Request,
    *,
    timeout: float,
    interface: str,
) -> bytes:
    parsed = urlsplit(request.full_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(
            "interface-bound MOEX transport requires HTTPS"
        )

    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query

    connection = _InterfaceBoundHTTPSConnection(
        parsed.hostname,
        interface=interface,
        timeout=timeout,
        context=ssl.create_default_context(),
    )

    try:
        connection.request(
            request.get_method(),
            target,
            headers=dict(request.header_items()),
        )
        response = connection.getresponse()
        raw = response.read()

        if not 200 <= response.status < 300:
            raise MoexIssError(
                "MOEX ISS returned HTTP "
                f"{response.status} {response.reason}"
            )

        return raw
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class MoexFuturesContract:
    secid: str
    board_id: str
    short_name: str
    asset_code: str
    last_trade_date: str
    min_step: float
    step_price: float
    initial_margin: float
    lot_volume: int
    exchange_buy_sell_fee: float

    @property
    def rubles_per_price_unit(self) -> float:
        if self.min_step <= 0.0:
            raise ValueError("min_step must be > 0")
        return self.step_price / self.min_step


@dataclass(frozen=True, slots=True)
class MoexCandle:
    secid: str
    begin: str
    end: str
    open: float
    high: float
    low: float
    close: float
    value: float
    volume: float

    @property
    def observed_at(self) -> float:
        """
        A completed OHLC candle is causal only at its END.

        Using `begin` as the event timestamp would leak the candle's future
        high/low/close into cognition.
        """
        return _parse_moex_datetime(self.end)


class MoexIssError(RuntimeError):
    pass


class MoexFuturesHistoryClient:
    """
    Narrow MOEX ISS reader for historical futures candles and current
    machine-readable contract metadata.

    It performs no trading and makes no cognitive decisions.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be > 0")

        explicit = (
            api_key.strip()
            if api_key is not None
            else ""
        )
        env_key = os.environ.get(
            "MOEX_API_KEY",
            "",
        ).strip()

        self.api_key = explicit or env_key or None
        self.timeout_seconds = float(timeout_seconds)
        self.base_url = (
            _AUTH_ISS
            if self.api_key
            else _PUBLIC_ISS
        )
        self._network_interface = (
            _physical_default_interface()
        )

    # --------------------------------------------------------
    # Contract discovery
    # --------------------------------------------------------

    def futures_contracts(
        self,
        *,
        asset_code: str | None = None,
    ) -> tuple[MoexFuturesContract, ...]:
        payload = self._get_json(
            "/engines/futures/markets/forts/"
            "securities.json",
            {
                "iss.meta": "off",
                "iss.only": "securities",
                "securities.columns": (
                    "SECID,BOARDID,SHORTNAME,ASSETCODE,"
                    "LASTTRADEDATE,MINSTEP,STEPPRICE,"
                    "INITIALMARGIN,LOTVOLUME,BUYSELLFEE"
                ),
            },
        )

        columns, rows = _block(
            payload,
            "securities",
        )
        index = _column_index(columns)

        wanted = (
            asset_code.strip().upper()
            if asset_code
            else None
        )

        result: list[MoexFuturesContract] = []

        for row in rows:
            row_asset = str(
                row[index["ASSETCODE"]]
                or ""
            ).upper()

            if wanted and row_asset != wanted:
                continue

            result.append(
                MoexFuturesContract(
                    secid=str(
                        row[index["SECID"]]
                    ),
                    board_id=str(
                        row[index["BOARDID"]]
                    ),
                    short_name=str(
                        row[index["SHORTNAME"]]
                    ),
                    asset_code=row_asset,
                    last_trade_date=str(
                        row[index["LASTTRADEDATE"]]
                        or ""
                    ),
                    min_step=float(
                        row[index["MINSTEP"]]
                    ),
                    step_price=float(
                        row[index["STEPPRICE"]]
                    ),
                    initial_margin=float(
                        row[index["INITIALMARGIN"]]
                    ),
                    lot_volume=int(
                        row[index["LOTVOLUME"]]
                    ),
                    exchange_buy_sell_fee=float(
                        row[index["BUYSELLFEE"]]
                    ),
                )
            )

        result.sort(
            key=lambda item: (
                item.last_trade_date,
                item.secid,
            )
        )

        return tuple(result)

    def contract(
        self,
        secid: str,
    ) -> MoexFuturesContract:
        wanted = secid.strip().upper()
        if not wanted:
            raise ValueError("secid must not be empty")

        for item in self.futures_contracts():
            if item.secid.upper() == wanted:
                return item

        raise MoexIssError(
            f"MOEX futures contract {secid!r} "
            "was not found in the current FORTS list"
        )

    # --------------------------------------------------------
    # Candles
    # --------------------------------------------------------

    def candles(
        self,
        secid: str,
        *,
        from_date: str,
        till_date: str,
        interval_minutes: int = 1,
        board_id: str = "RFUD",
    ) -> tuple[MoexCandle, ...]:
        clean_secid = secid.strip().upper()
        clean_board = board_id.strip().upper()

        if not clean_secid:
            raise ValueError("secid must not be empty")
        if not clean_board:
            raise ValueError("board_id must not be empty")
        if interval_minutes <= 0:
            raise ValueError(
                "interval_minutes must be > 0"
            )

        result: list[MoexCandle] = []
        seen: set[tuple[str, str]] = set()
        start = 0

        while True:
            payload = self._get_json(
                (
                    "/engines/futures/markets/forts/"
                    f"boards/{clean_board}/"
                    f"securities/{clean_secid}/"
                    "candles.json"
                ),
                {
                    "iss.meta": "off",
                    "iss.only": "candles",
                    "from": from_date,
                    "till": till_date,
                    "interval": interval_minutes,
                    "start": start,
                },
            )

            columns, rows = _block(
                payload,
                "candles",
            )

            if not rows:
                break

            index = _column_index(columns)

            required = {
                "open",
                "close",
                "high",
                "low",
                "value",
                "volume",
                "begin",
                "end",
            }
            missing = required.difference(
                name.casefold()
                for name in columns
            )
            if missing:
                raise MoexIssError(
                    "MOEX candle response is missing "
                    f"columns: {sorted(missing)!r}"
                )

            # MOEX candle column names are usually lowercase.
            lower_index = {
                name.casefold(): offset
                for name, offset
                in index.items()
            }

            for row in rows:
                begin = str(
                    row[lower_index["begin"]]
                )
                end = str(
                    row[lower_index["end"]]
                )

                key = (begin, end)
                if key in seen:
                    continue
                seen.add(key)

                candle = MoexCandle(
                    secid=clean_secid,
                    begin=begin,
                    end=end,
                    open=float(
                        row[lower_index["open"]]
                    ),
                    high=float(
                        row[lower_index["high"]]
                    ),
                    low=float(
                        row[lower_index["low"]]
                    ),
                    close=float(
                        row[lower_index["close"]]
                    ),
                    value=float(
                        row[lower_index["value"]]
                        or 0.0
                    ),
                    volume=float(
                        row[lower_index["volume"]]
                        or 0.0
                    ),
                )

                # Parse immediately so malformed times fail before
                # they enter MarketInputBuilder.
                _ = candle.observed_at
                result.append(candle)

            start += len(rows)

        result.sort(
            key=lambda item: (
                item.observed_at,
                item.begin,
            )
        )

        return tuple(result)

    def candle_events(
        self,
        secid: str,
        *,
        from_date: str,
        till_date: str,
        interval_minutes: int = 1,
        board_id: str = "RFUD",
    ) -> tuple[MarketBackendEvent, ...]:
        candles = self.candles(
            secid,
            from_date=from_date,
            till_date=till_date,
            interval_minutes=interval_minutes,
            board_id=board_id,
        )

        interval_seconds = float(
            interval_minutes * 60
        )

        return tuple(
            MarketBackendEvent(
                id=(
                    f"moex:candle:{item.secid}:"
                    f"{item.begin}:{item.end}:"
                    f"{interval_minutes}"
                ),
                kind="candle",
                # CRITICAL: completed candle appears at its END,
                # never at its begin timestamp.
                observed_at=item.observed_at,
                instrument_id=item.secid,
                payload={
                    "open": item.open,
                    "high": item.high,
                    "low": item.low,
                    "close": item.close,
                    "volume": item.volume,
                    "value": item.value,
                    "interval_seconds": (
                        interval_seconds
                    ),
                    "candle_begin": item.begin,
                    "candle_end": item.end,
                    "is_complete": True,
                    "source": "moex:iss",
                    "historical": True,
                },
            )
            for item in candles
        )

    def candle_frame(
        self,
        secid: str,
        *,
        from_date: str,
        till_date: str,
        interval_minutes: int = 1,
        board_id: str = "RFUD",
    ) -> MarketBackendFrame:
        events = self.candle_events(
            secid,
            from_date=from_date,
            till_date=till_date,
            interval_minutes=interval_minutes,
            board_id=board_id,
        )

        if not events:
            raise MoexIssError(
                "MOEX returned no candles for "
                f"{secid!r} in {from_date}..{till_date}"
            )

        return MarketBackendFrame(
            backend_id="moex:iss:history",
            # Current shared enum has SANDBOX/REAL only.
            # Historical replay is marked explicitly in backend_id
            # and event payload rather than pretending it is live.
            mode=MarketMode.SANDBOX,
            observed_at=events[-1].observed_at,
            events=events,
            instruments=(),
        )

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    def _get_json(
        self,
        path: str,
        params: Mapping[str, object],
    ) -> Mapping[str, object]:
        query = urlencode(
            {
                key: str(value)
                for key, value
                in params.items()
            }
        )
        url = (
            f"{self.base_url}{path}?{query}"
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "Heiwa-v2/market-history",
        }
        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        request = Request(
            url,
            headers=headers,
            method="GET",
        )

        try:
            if self._network_interface:
                raw = _read_bound_https(
                    request,
                    timeout=self.timeout_seconds,
                    interface=self._network_interface,
                )
            else:
                with urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    raw = response.read()
        except Exception as exc:
            raise MoexIssError(
                f"MOEX ISS request failed: {url}"
            ) from exc

        try:
            parsed = json.loads(
                raw.decode("utf-8")
            )
        except Exception as exc:
            raise MoexIssError(
                "MOEX ISS returned invalid JSON"
            ) from exc

        if not isinstance(parsed, Mapping):
            raise MoexIssError(
                "MOEX ISS JSON root is not a mapping"
            )

        return parsed


def _block(
    payload: Mapping[str, object],
    name: str,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[object, ...], ...],
]:
    raw = payload.get(name)
    if not isinstance(raw, Mapping):
        raise MoexIssError(
            f"MOEX ISS response has no {name!r} block"
        )

    columns_raw = raw.get("columns")
    data_raw = raw.get("data")

    if not isinstance(columns_raw, list):
        raise MoexIssError(
            f"{name!r}.columns is invalid"
        )
    if not isinstance(data_raw, list):
        raise MoexIssError(
            f"{name!r}.data is invalid"
        )

    columns = tuple(
        str(item)
        for item in columns_raw
    )

    rows: list[tuple[object, ...]] = []
    for item in data_raw:
        if not isinstance(item, list):
            raise MoexIssError(
                f"{name!r}.data row is invalid"
            )
        if len(item) != len(columns):
            raise MoexIssError(
                f"{name!r}.data row width mismatch"
            )
        rows.append(tuple(item))

    return columns, tuple(rows)


def _column_index(
    columns: Iterable[str],
) -> dict[str, int]:
    return {
        str(name): index
        for index, name
        in enumerate(columns)
    }


def _parse_moex_datetime(
    value: str,
) -> float:
    text = value.strip()
    if not text:
        raise MoexIssError(
            "MOEX candle datetime is empty"
        )

    # ISS candle timestamps are Moscow-local datetimes without an
    # explicit offset. Attach Europe/Moscow explicitly so replay epochs
    # are machine-independent and remain compatible with future
    # session/clearing simulation.
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MoexIssError(
            f"invalid MOEX datetime: {value!r}"
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=_MOSCOW
        )

    return parsed.timestamp()

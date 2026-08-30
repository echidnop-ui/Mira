from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Iterable, Mapping


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

_RUSSIAN_LOWER = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_WHITESPACE = " \n\t"
_PUNCTUATION = "?!. ,:;-\"'()[]{}_/\\@#%&*+=<>|~`^"


def _unique_characters(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for character in value:
            if character not in seen:
                seen.add(character)
                result.append(character)
    return tuple(result)


DEFAULT_CHARACTERS = _unique_characters(
    (
        _RUSSIAN_LOWER,
        _RUSSIAN_LOWER.upper(),
        _LATIN_LOWER,
        _LATIN_LOWER.upper(),
        _DIGITS,
        _WHITESPACE,
        _PUNCTUATION,
    )
)


@dataclass(frozen=True, slots=True)
class TokenizerState:
    version: int
    characters: tuple[str, ...]
    normalization: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "characters": list(self.characters),
            "normalization": self.normalization,
        }


class UnicodeCharTokenizer:
    """Deterministic, serializable Unicode character tokenizer.

    The default vocabulary explicitly models Russian and basic Latin text.
    Unknown Unicode maps to UNK and decodes visibly as ``<UNK>`` rather than
    introducing an invalid UTF-8 replacement character.
    """

    VERSION = 1

    def __init__(self, characters: Iterable[str] = DEFAULT_CHARACTERS) -> None:
        normalized = _unique_characters(characters)
        if any(len(character) != 1 for character in normalized):
            raise ValueError("tokenizer vocabulary entries must be characters")
        self.characters = normalized
        self.tokens = (*SPECIAL_TOKENS, *normalized)
        self._token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_characters: int | None = None,
    ) -> list[int]:
        value = self.normalize(text)
        if max_characters is not None:
            if max_characters <= 0:
                raise ValueError("max_characters must be > 0")
            value = value[:max_characters]
        result = [self.bos_id] if add_bos else []
        result.extend(self._token_to_id.get(character, self.unk_id) for character in value)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        parts: list[str] = []
        for raw in token_ids:
            token_id = int(raw)
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id < 0 or token_id >= self.vocab_size:
                token_id = self.unk_id
            token = self.tokens[token_id]
            if token_id < len(SPECIAL_TOKENS):
                if token_id == self.unk_id:
                    parts.append(UNK_TOKEN)
                elif not skip_special:
                    parts.append(token)
                continue
            parts.append(token)
        return "".join(parts)

    def state(self) -> TokenizerState:
        return TokenizerState(
            version=self.VERSION,
            characters=self.characters,
            normalization="NFC+CRLF_TO_LF",
        )

    def state_dict(self) -> dict[str, object]:
        return self.state().as_dict()

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "UnicodeCharTokenizer":
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError("unsupported tokenizer version")
        raw = payload.get("characters")
        if not isinstance(raw, (list, tuple)) or not all(isinstance(x, str) for x in raw):
            raise ValueError("tokenizer checkpoint has invalid characters")
        tokenizer = cls(raw)
        if payload.get("normalization") != "NFC+CRLF_TO_LF":
            raise ValueError("tokenizer normalization is incompatible")
        return tokenizer


@dataclass(frozen=True, slots=True)
class ByteTokenizerState:
    version: int
    encoding: str

    def as_dict(self) -> dict[str, object]:
        return {"kind": "utf8_bytes", "version": self.version, "encoding": self.encoding}


class Utf8ByteTokenizer:
    """Exact UTF-8 byte tokenizer with four reserved control tokens.

    IDs 0..3 are PAD/BOS/EOS/UNK for API compatibility. Raw byte values
    0..255 are represented by IDs 4..259. Decoding is exact for valid UTF-8;
    invalid partial generations are ignored rather than injected as teacher data.
    """

    VERSION = 1
    BYTE_OFFSET = len(SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return self.BYTE_OFFSET + 256

    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        max_characters: int | None = None,
    ) -> list[int]:
        # max_characters is retained for tokenizer API compatibility; for this
        # tokenizer it is intentionally interpreted as a byte budget.
        raw = self.normalize(text).encode("utf-8")
        if max_characters is not None:
            if max_characters <= 0:
                raise ValueError("max_characters must be > 0")
            raw = raw[:max_characters]
        result = [self.bos_id] if add_bos else []
        result.extend(self.BYTE_OFFSET + byte for byte in raw)
        if add_eos:
            result.append(self.eos_id)
        return result

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        raw = bytearray()
        for value in token_ids:
            token_id = int(value)
            if token_id == self.eos_id and stop_at_eos:
                break
            if token_id < self.BYTE_OFFSET:
                continue
            byte = token_id - self.BYTE_OFFSET
            if 0 <= byte <= 255:
                raw.append(byte)
        return bytes(raw).decode("utf-8", errors="ignore")

    def state_dict(self) -> dict[str, object]:
        return ByteTokenizerState(self.VERSION, "utf-8").as_dict()

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "Utf8ByteTokenizer":
        if payload.get("kind") != "utf8_bytes":
            raise ValueError("tokenizer checkpoint is not utf8_bytes")
        if int(payload.get("version", -1)) != cls.VERSION:
            raise ValueError("unsupported byte tokenizer version")
        if payload.get("encoding") != "utf-8":
            raise ValueError("unsupported byte tokenizer encoding")
        return cls()


def tokenizer_from_state_dict(payload: Mapping[str, object]):
    if payload.get("kind") == "utf8_bytes":
        return Utf8ByteTokenizer.from_state_dict(payload)
    return UnicodeCharTokenizer.from_state_dict(payload)

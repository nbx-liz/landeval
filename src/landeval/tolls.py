"""The toll request shape, and the price grammar it makes readable.

The Routes API answers these two requests the same way:

    a route that genuinely owes nothing        -> HTTP 200, travelAdvisory: {}
    a request that forgot extraComputations    -> HTTP 200, travelAdvisory: {}

The two bodies are identical. **Nothing on the response side separates "free"
from "did not ask"**, and neither is an error, so a reader that sees an empty
advisory and answers zero is wrong half the time with no way to notice.

The separation is therefore made on the request side. A :class:`RequestShape`
records what was asked; it rides in the cache key and in the cache body, and
:func:`toll_yen` refuses to read a price out of a body whose shape does not say
the question was put. The default shape is the body sent without any toll
fields, and it contributes nothing to the cache key -- entries written in the
default shape stay valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Only the fields actually used, kept identical to the legacy mask so the
# default path is byte-for-byte the request that has always been sent.
BASE_FIELD_MASK = "routes.duration,routes.distanceMeters"

# Both halves, always together. A request that drops the computation, or one
# that drops the mask, still returns 200 and looks like a free route, so these
# two are never assembled apart.
TOLL_FIELDS = "routes.travelAdvisory.tollInfo,routes.legs.travelAdvisory.tollInfo"
TOLL_COMPUTATION = "TOLLS"

# The two values the API documents for Japan. A pass outside them is refused
# rather than sent: a fixture written from a guess inherits the guess.
KNOWN_JP_TOLL_PASSES = frozenset({"JP_ETC", "JP_ETC2"})

# The documented `VehicleEmissionType` enum. Its spellings are accepted because
# they are documented, and anything else is refused so a typo cannot reach the
# wire as a silently ignored key.
KNOWN_EMISSION_TYPES = frozenset({"GASOLINE", "ELECTRIC", "HYBRID", "DIESEL"})

# The spelling the grammar accepts: ASCII digits, no sign, no separators, no
# leading zero beyond `0` itself.
_CANONICAL_UNITS = re.compile(r"0|[1-9][0-9]*")


def _described(value: Any) -> str:
    """A value from a response or a cache body, described without its text.

    An error message never quotes such a value. A response can echo the API
    key, and once a message quotes a string -- through ``repr`` or
    ``json.dumps``, around an echo that may already be escaped any number of
    times -- no finite list of the key's spellings can find every copy. So a
    message names the field and says what kind of value it holds: its type
    and, for text and containers, its length. A number, a boolean or ``None``
    is shown, because once parsed as one it cannot hold the key.
    """
    if value is None or isinstance(value, bool | int | float):
        return f"{type(value).__name__} {value!r}"
    kind = type(value).__name__
    if isinstance(value, str):
        return f"{kind}, {len(value)} chars"
    if isinstance(value, list | dict):
        return f"{kind}, {len(value)} items"
    return kind


class TollError(Exception):
    """Raised for a shape that cannot be sent, or a price that cannot be read."""


@dataclass(frozen=True)
class RequestShape:
    """What a route request asked for, beyond origin, destination and time.

    Frozen and validated on construction: an unsendable shape must not exist,
    because the thing it would produce is a 200 that reads like a free route.
    """

    tolls: bool = False
    toll_passes: tuple[str, ...] = ()
    emission_type: str | None = None

    def __post_init__(self) -> None:
        # `from_cache` requires a real boolean, so anything else here produces a
        # shape that writes an entry its own reader refuses. The two ends of one
        # round trip have to agree about what a value is.
        if not isinstance(self.tolls, bool):
            raise TollError(f"tolls is not a boolean: {self.tolls!r}")
        if not isinstance(self.toll_passes, tuple):
            raise TollError(f"toll_passes is not a tuple: {self.toll_passes!r}")
        for value in self.toll_passes:
            if value not in KNOWN_JP_TOLL_PASSES:
                raise TollError(
                    f"unsupported toll pass {value!r}; "
                    f"known: {sorted(KNOWN_JP_TOLL_PASSES)}. "
                    "Verify a new spelling against the API before declaring it"
                )
        if len(set(self.toll_passes)) != len(self.toll_passes):
            raise TollError(
                f"duplicate toll pass in {self.toll_passes!r}; a pass declared twice is "
                "one declaration written twice, and the duplicate reaches the wire"
            )
        if self.emission_type is not None and self.emission_type not in KNOWN_EMISSION_TYPES:
            raise TollError(
                f"unknown emission type {self.emission_type!r}; "
                f"known: {sorted(KNOWN_EMISSION_TYPES)}"
            )
        if not self.tolls and (self.toll_passes or self.emission_type is not None):
            # Either modifier without the computation produces a request that
            # carries `routeModifiers` and no `extraComputations` -- a body
            # that returns 200 with an empty advisory. The
            # emission half is worse than useless: `body_extras` drops it, so
            # the wire, the key and the cache body would describe three
            # different questions.
            raise TollError(
                "toll modifiers were given without asking for tolls. The request "
                "would carry no extraComputations, and the modifiers would not "
                "reach the wire at all while still changing the cache key"
            )
        # Canonical order, because `token()` sorts and therefore declares the two
        # orderings the same question. Leaving the field unsorted made one digest
        # hold two unequal shapes, and the reader then refused its own entry.
        object.__setattr__(self, "toll_passes", tuple(sorted(self.toll_passes)))

    # -- what the request carries -----------------------------------------

    @property
    def is_default(self) -> bool:
        """True for the body this pipeline sent before tolls existed."""
        return self == DEFAULT_SHAPE

    def body_extras(self) -> dict[str, Any]:
        """The keys to merge into the request body.

        Empty for the default shape, and empty means *absent*: the default
        body carries no ``routeModifiers`` key at all rather than an empty one.
        """
        if not self.tolls:
            return {}
        extras: dict[str, Any] = {"extraComputations": [TOLL_COMPUTATION]}
        modifiers: dict[str, Any] = {}
        if self.toll_passes:
            modifiers["tollPasses"] = list(self.toll_passes)
        if self.emission_type is not None:
            modifiers["vehicleInfo"] = {"emissionType": self.emission_type}
        if modifiers:
            extras["routeModifiers"] = modifiers
        return extras

    def field_mask(self) -> str:
        """The ``X-Goog-FieldMask`` for this shape, base fields included."""
        if not self.tolls:
            return BASE_FIELD_MASK
        return f"{BASE_FIELD_MASK},{TOLL_FIELDS}"

    # -- what the cache records -------------------------------------------

    def token(self) -> str:
        """The shape's contribution to the cache key.

        Empty for the default shape, and the caller appends nothing when it is
        empty, so the digest of a default request is the digest it has always
        been and entries written before shapes existed keep resolving.

        Passes are sorted at construction, not here: the same two passes written
        in either order are the same question, and two spellings of one question
        would buy the answer twice. Sorting only in the token would be worse than
        not sorting -- one digest would then hold two shapes that compare
        unequal, and the cache reader would refuse an entry it had written
        itself.
        """
        if self.is_default:
            return ""
        parts = [f"tolls={int(self.tolls)}"]
        if self.toll_passes:
            parts.append("passes=" + "+".join(sorted(self.toll_passes)))
        if self.emission_type is not None:
            parts.append(f"emission={self.emission_type}")
        return ";".join(parts)

    def to_cache(self) -> dict[str, Any]:
        """The shape as it is written into a cache body.

        The default shape writes nothing, so a default entry written today is
        the same bytes as one written a year ago.
        """
        if self.is_default:
            return {}
        return {
            "shape": {
                "tolls": self.tolls,
                "toll_passes": list(self.toll_passes),
                "emission_type": self.emission_type,
            }
        }

    @classmethod
    def from_cache(cls, body: dict[str, Any]) -> RequestShape:
        """The shape a cache body was written under.

        A body with no ``shape`` key is an entry written without one, and
        those are all in the default shape -- which does not ask about
        tolls, so :func:`toll_yen` will report the price missing rather than
        zero. A body whose ``shape`` is malformed is an error instead: absent
        means "written without the field", and answering the same way
        for "written wrong" hides a writer that is getting it wrong now.
        """
        # `in` rather than `get() is None`, because JSON `null` and an absent
        # key are different statements and only one of them means "written
        # without the field". Treating them alike would let a hand-edited
        # `"shape": null` pass as a legacy entry -- a malformed body wearing
        # an old body's clothes, which is the false-clean shape this reader
        # refuses.
        if "shape" not in body:
            return DEFAULT_SHAPE
        raw = body["shape"]
        if not isinstance(raw, dict):
            raise TollError(f"cached shape is not an object: {type(raw).__name__}")

        tolls = raw.get("tolls")
        if not isinstance(tolls, bool):
            raise TollError(f"cached shape.tolls is not a boolean ({_described(tolls)})")

        passes = raw.get("toll_passes", [])
        if not isinstance(passes, list) or not all(isinstance(item, str) for item in passes):
            raise TollError(
                f"cached shape.toll_passes is not a list of strings ({_described(passes)})"
            )

        emission = raw.get("emission_type")
        if emission is not None and not isinstance(emission, str):
            raise TollError(f"cached shape.emission_type is not a string ({_described(emission)})")

        # The constructor's own refusals quote the value they refuse, which is
        # right for a shape a caller writes and wrong for one read from a cache
        # body. Raised after the handler, so that refusal is not chained either.
        try:
            return cls(tolls=tolls, toll_passes=tuple(passes), emission_type=emission)
        except TollError:
            failure = TollError(
                "cached shape is not one a request can carry: a toll pass or an emission "
                "type outside the known set, a pass named twice, or modifiers without "
                f"tolls (toll_passes: {_described(passes)}; "
                f"emission_type: {_described(emission)})"
            )
        raise failure


DEFAULT_SHAPE = RequestShape()


def toll_yen(route: dict[str, Any], shape: RequestShape) -> int | None:
    """The toll for one route in yen, or ``None`` when the question was not put.

    ``None`` and ``0`` are different answers and the difference is not visible
    in ``route`` alone. ``shape`` is what decides, and it has to be the shape the
    response was actually fetched under, not the shape the caller would like it
    to have been.
    """
    if not shape.tolls:
        return None

    # Membership, not `get() is None`, at both levels. An absent key is the
    # API declining to say; an explicit `null` is the API saying something else.
    # Reading them the same way turns a form outside the grammar into a
    # confident zero.
    if "travelAdvisory" not in route:
        raise TollError(
            "the request asked for tolls but the response carries no travelAdvisory; "
            "the field mask was probably incomplete (the API can answer this "
            "under HTTP 200)"
        )
    advisory = route["travelAdvisory"]
    if not isinstance(advisory, dict):
        raise TollError(f"travelAdvisory is not an object: {type(advisory).__name__}")

    if "tollInfo" not in advisory:
        # A short surface-road route, asked properly, returns the advisory
        # present and empty. Zero is only readable here because `shape` proved
        # the question was asked -- and only for an *absent* tollInfo, which is
        # the form the grammar admits.
        return 0
    info = advisory["tollInfo"]
    if not isinstance(info, dict):
        raise TollError(f"tollInfo is not an object: {type(info).__name__}")

    return _price_yen(info)


def _price_yen(info: dict[str, Any]) -> int:
    """The single yen amount in a ``tollInfo``, or an error.

    The grammar is closed: JPY, exactly one entry, integer
    ``units`` as a string, ``nanos`` absent or zero. Anything else is refused
    rather than coerced -- summing two currencies invents a number and taking
    the first drops one silently.
    """
    prices = info.get("estimatedPrice")
    if not isinstance(prices, list) or not prices:
        raise TollError(
            f"tollInfo has no estimatedPrice entries (tollInfo: {_described(info)}; "
            f"estimatedPrice: {_described(prices)})"
        )
    if len(prices) != 1:
        raise TollError(
            f"expected exactly one price, got {len(prices)} (estimatedPrice: list). "
            "Routes priced in more than one currency are not supported"
        )

    price = prices[0]
    if not isinstance(price, dict):
        raise TollError(f"price entry is not an object: {type(price).__name__}")

    currency = price.get("currencyCode")
    if currency != "JPY":
        raise TollError(f"unexpected currencyCode ({_described(currency)}); only JPY is supported")

    nanos = price.get("nanos", 0)
    if isinstance(nanos, bool) or not isinstance(nanos, int):
        raise TollError(f"nanos is not an integer ({_described(nanos)})")
    if nanos != 0:
        raise TollError(
            f"nanos is {nanos}, but yen is an integer currency and the field is "
            "expected to be absent or zero; a fractional yen is not rounded away here"
        )

    units = price.get("units")
    if not isinstance(units, str):
        raise TollError(f"units is not a string ({_described(units)})")
    # `int()` is far more permissive than the grammar: it accepts `"+1"`,
    # `" 1"`, `"1_0"`, `"-0"` and full-width digits, and answers 1, 1, 10, 0 and
    # 0. The API sends a canonical ASCII decimal string, so that is what is
    # accepted -- checking the type
    # and leaving the spelling open would be the same axis mistake as leaving
    # the numeric domain open.
    if not _CANONICAL_UNITS.fullmatch(units):
        raise TollError(f"units is not a canonical decimal string ({_described(units)})")
    try:
        yen = int(units)
    except ValueError:  # pragma: no cover - the pattern already refuses these
        raise TollError(f"units is not an integer string ({_described(units)})") from None
    # No non-negative check here: the pattern above admits no sign, so a
    # negative amount is refused as an unmeasured spelling before it becomes a
    # number. A second bound would be unreachable code pretending to be a guard.
    return yen

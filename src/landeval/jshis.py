"""The earthquake layer, read one point at a time.

J-SHIS answers two different questions about a 250 m mesh, and the reading needs
both: how much this ground amplifies a shake (`sstrct`), and how likely a strong
one is here (`pshm`). Neither is a designated zone in the sense the other hazard
layers use -- there is no line on a map that a parcel is inside or outside -- so
this module produces numbers and a landform name, never 区域内 / 区域外.

**There is no liquefaction layer.** 「液状化」 is the natural thing to ask
for, and J-SHIS publishes no designation for it. What exists is the microlandform
classification, which is a *predictor* of liquefaction rather than a statement
about any particular ground. It is reported under its own name, and the caller
prints it as what it is.

Fetching is at lookup time: every cell of a region at 250 m resolution would
be a very large number of requests against a public service that charges
nothing and owes us nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from landeval.basedate import JST

BASE_URL = "https://www.j-shis.bosai.go.jp/map/api"

# V4 is the 2020 national
# seismic hazard map's ground model; Y2024 the current probabilistic one.
DEFAULT_GROUND_EDITION = "V4"
DEFAULT_HAZARD_EDITION = "Y2024"

# 「今後 30 年間に計測震度 5.5 以上」, which is 震度 6 弱以上 -- the threshold
# ordinary readers recognise. AVR averages over the seismic activity models.
HAZARD_CASE = "AVR"
HAZARD_EQCODE = "TTL_MTTL"
HAZARD_ATTRIBUTE = "T30_I55_PS"

# WGS84. The API also accepts JGD2000 and Tokyo Datum, and the difference
# between them is a few hundred metres -- more than the mesh this reads.
EPSG = "4326"

DEFAULT_DELAY_S = 0.2
DEFAULT_TIMEOUT_S = 30.0


class JshisError(Exception):
    """Raised for a failed request, an empty answer, or a value that will not parse."""


@dataclass(frozen=True)
class JshisReading:
    """One 250 m mesh's earthquake numbers, and where they came from."""

    meshcode: str
    avs30_ms: float
    amplification: float
    landform: str
    probability_6lower_30y: float
    ground_edition: str
    hazard_edition: str
    fetched_at: datetime

    @property
    def probability_percent(self) -> float:
        return self.probability_6lower_30y * 100.0


def _number(properties: dict[str, Any], key: str) -> float:
    """Parse one of the API's string-encoded numbers, or say which one failed.

    Every value arrives as a string, so a change of format shows up as a string
    that does not parse. Substituting a default here would put a made-up number
    beside measured ones with nothing to tell them apart.
    """
    raw = properties.get(key)
    if raw is None:
        raise JshisError(f"the response has no {key}; it has {sorted(properties)}")
    try:
        return float(raw)
    except (TypeError, ValueError) as failure:
        raise JshisError(f"{key} is {raw!r}, which is not a number") from failure


class JshisClient:
    """Reads J-SHIS for a coordinate, caching each mesh's answer on disk."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        transport: httpx.BaseTransport | None = None,
        delay_s: float = DEFAULT_DELAY_S,
        ground_edition: str = DEFAULT_GROUND_EDITION,
        hazard_edition: str = DEFAULT_HAZARD_EDITION,
    ) -> None:
        self._cache_dir = cache_dir
        self._delay_s = delay_s
        self.ground_edition = ground_edition
        self.hazard_edition = hazard_edition
        self.requests = 0
        self._client = httpx.Client(
            timeout=DEFAULT_TIMEOUT_S,
            transport=transport,
            headers={"User-Agent": "landeval/0.1"},
        )

    def __enter__(self) -> JshisClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- fetching ----------------------------------------------------------

    def _cache_path(self, url: str, position: str) -> Path:
        # The edition is in the URL, so it is in the key: raising the edition
        # asks a different question and must not be answered from the old
        # model's cache.
        digest = hashlib.sha256(f"{url}|{position}".encode()).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _get(self, url: str, params: dict[str, str]) -> tuple[dict[str, Any], datetime]:
        path = self._cache_path(url, params["position"])
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            return stored["body"], datetime.fromisoformat(stored["fetched_at"])

        if self._delay_s:
            time.sleep(self._delay_s)
        response = self._client.get(url, params=params)
        self.requests += 1
        if response.status_code != 200:
            raise JshisError(f"{url} answered HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as failure:
            raise JshisError(f"{url} answered a body that is not JSON") from failure
        if not isinstance(body, dict):
            raise JshisError(f"{url} answered a {type(body).__name__}, not an object")

        fetched_at = datetime.now(tz=JST)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": fetched_at.isoformat(), "body": body}, ensure_ascii=False),
            encoding="utf-8",
        )
        return body, fetched_at

    def _properties(self, body: dict[str, Any], what: str) -> dict[str, Any]:
        features = body.get("features")
        if not features:
            raise JshisError(
                f"J-SHIS returned no mesh for the {what} query; every point on land in "
                "Japan is in some 250 m mesh, so an empty answer is the service declining "
                "to say rather than ground with no data"
            )
        properties = features[0].get("properties")
        if not isinstance(properties, dict):
            raise JshisError(f"the {what} feature has no properties object")
        return properties

    # -- the reading -------------------------------------------------------

    def reading_at(self, lat: float, lon: float) -> JshisReading:
        """The earthquake numbers for the 250 m mesh containing this point.

        Note the coordinate order: J-SHIS takes ``lon,lat``, the opposite of
        every other call in this repository. Transposed, the query asks about
        some other point, which may well be answered rather than refused.
        """
        position = f"{lon},{lat}"

        ground_url = f"{BASE_URL}/sstrct/{self.ground_edition}/meshinfo.geojson"
        ground_body, ground_time = self._get(ground_url, {"position": position, "epsg": EPSG})
        ground = self._properties(ground_body, "surface ground")

        hazard_url = (
            f"{BASE_URL}/pshm/{self.hazard_edition}/{HAZARD_CASE}/{HAZARD_EQCODE}/meshinfo.geojson"
        )
        hazard_body, hazard_time = self._get(
            hazard_url,
            {"position": position, "epsg": EPSG, "attr": HAZARD_ATTRIBUTE},
        )
        hazard = self._properties(hazard_body, "seismic hazard")

        landform = ground.get("JNAME")
        if not isinstance(landform, str) or not landform:
            raise JshisError(f"the surface-ground answer has no JNAME; it has {sorted(ground)}")

        return JshisReading(
            meshcode=str(ground.get("meshcode", "")),
            avs30_ms=_number(ground, "AVS"),
            amplification=_number(ground, "ARV"),
            landform=landform,
            probability_6lower_30y=_number(hazard, HAZARD_ATTRIBUTE),
            ground_edition=self.ground_edition,
            hazard_edition=self.hazard_edition,
            # The older of the two: the reading is only as fresh as its stalest
            # half, and saying the newer time would overstate it.
            fetched_at=min(ground_time, hazard_time),
        )

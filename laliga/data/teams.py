"""Football-data.co.uk names and the SportMonks names they refer to.

The canonical key is stable. A team that also appears in ``matches.csv``
keeps that file's team id. A team that only appears in the older history
gets a synthetic id in the 8000000 range, so it cannot collide with a
SportMonks id and stays the same team across seasons.
"""

from __future__ import annotations

import unicodedata

# football-data spelling and common SportMonks spellings, after
# accent-stripping and lower-casing. Values are canonical keys.
_ALIAS_PAIRS = {
    "ath madrid": "atletico madrid",
    "atletico madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "club atletico de madrid": "atletico madrid",
    "ath bilbao": "athletic club",
    "athletic club": "athletic club",
    "athletic bilbao": "athletic club",
    "athletic": "athletic club",
    "real madrid": "real madrid",
    "barcelona": "barcelona",
    "fc barcelona": "barcelona",
    "betis": "real betis",
    "real betis": "real betis",
    "real betis balompie": "real betis",
    "celta": "celta vigo",
    "celta vigo": "celta vigo",
    "celta de vigo": "celta vigo",
    "rc celta": "celta vigo",
    "rc celta de vigo": "celta vigo",
    "espanol": "espanyol",
    "espanyol": "espanyol",
    "rcd espanyol": "espanyol",
    "rcd espanyol de barcelona": "espanyol",
    "espanyol de barcelona": "espanyol",
    "getafe": "getafe",
    "getafe cf": "getafe",
    "sociedad": "real sociedad",
    "real sociedad": "real sociedad",
    "valencia": "valencia",
    "valencia cf": "valencia",
    "sevilla": "sevilla",
    "sevilla fc": "sevilla",
    "villarreal": "villarreal",
    "villarreal cf": "villarreal",
    "vallecano": "rayo vallecano",
    "rayo vallecano": "rayo vallecano",
    "osasuna": "osasuna",
    "ca osasuna": "osasuna",
    "mallorca": "mallorca",
    "rcd mallorca": "mallorca",
    "alaves": "alaves",
    "deportivo alaves": "alaves",
    "girona": "girona",
    "girona fc": "girona",
    "las palmas": "las palmas",
    "ud las palmas": "las palmas",
    "leganes": "leganes",
    "cd leganes": "leganes",
    "valladolid": "valladolid",
    "real valladolid": "valladolid",
    "granada": "granada",
    "granada cf": "granada",
    "levante": "levante",
    "levante ud": "levante",
    "malaga": "malaga",
    "malaga cf": "malaga",
    "eibar": "eibar",
    "sd eibar": "eibar",
    "elche": "elche",
    "elche cf": "elche",
    "cadiz": "cadiz",
    "cadiz cf": "cadiz",
    "huesca": "huesca",
    "sd huesca": "huesca",
    "almeria": "almeria",
    "ud almeria": "almeria",
    "cordoba": "cordoba",
    "cordoba cf": "cordoba",
    "zaragoza": "zaragoza",
    "real zaragoza": "zaragoza",
    "sp gijon": "sporting gijon",
    "sporting gijon": "sporting gijon",
    "sporting de gijon": "sporting gijon",
    "la coruna": "deportivo la coruna",
    "deportivo la coruna": "deportivo la coruna",
    "deportivo a coruna": "deportivo la coruna",
    "deportivo de la coruna": "deportivo la coruna",
    "rc deportivo": "deportivo la coruna",
    "rc deportivo a coruna": "deportivo la coruna",
    "rc deportivo de la coruna": "deportivo la coruna",
    "deportivo": "deportivo la coruna",
    "oviedo": "real oviedo",
    "real oviedo": "real oviedo",
    "santander": "racing santander",
    "racing santander": "racing santander",
    "racing de santander": "racing santander",
}

# Every HomeTeam / AwayTeam in the SP1 files from 2012/13 through 2026/27.
FOOTBALL_DATA_TEAM_NAMES = (
    "Alaves",
    "Almeria",
    "Ath Bilbao",
    "Ath Madrid",
    "Barcelona",
    "Betis",
    "Cadiz",
    "Celta",
    "Cordoba",
    "Eibar",
    "Elche",
    "Espanol",
    "Getafe",
    "Girona",
    "Granada",
    "Huesca",
    "La Coruna",
    "Las Palmas",
    "Leganes",
    "Levante",
    "Malaga",
    "Mallorca",
    "Osasuna",
    "Oviedo",
    "Real Madrid",
    "Santander",
    "Sevilla",
    "Sociedad",
    "Sp Gijon",
    "Valencia",
    "Valladolid",
    "Vallecano",
    "Villarreal",
    "Zaragoza",
)

# Teams that appear in 2024/25, 2025/26, or the started 2026/27 file.
OVERLAP_FOOTBALL_DATA_NAMES = (
    "Alaves",
    "Ath Bilbao",
    "Ath Madrid",
    "Barcelona",
    "Betis",
    "Celta",
    "Elche",
    "Espanol",
    "Getafe",
    "Girona",
    "La Coruna",
    "Las Palmas",
    "Leganes",
    "Levante",
    "Malaga",
    "Mallorca",
    "Osasuna",
    "Oviedo",
    "Real Madrid",
    "Santander",
    "Sevilla",
    "Sociedad",
    "Valencia",
    "Valladolid",
    "Vallecano",
    "Villarreal",
)

_CANONICALS = tuple(sorted(set(_ALIAS_PAIRS.values())))
SYNTHETIC_TEAM_IDS = {name: 8_000_001 + index for index, name in enumerate(_CANONICALS)}
# Dropped only after an exact alias miss. "a" stays, so "Deportivo A Coruña"
# is not reduced to a different club. Exact keys such as "deportivo alaves"
# and "deportivo" are resolved before this list is used.
_NAME_PARTICLES = frozenset(
    {"de", "del", "la", "el", "cf", "fc", "ud", "cd", "rcd", "ca", "sd", "rc", "club", "balompie", "sad"}
)


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def canonical_team(name: str) -> str:
    """Map a football-data or SportMonks name onto one canonical key."""

    key = normalize_team_name(name)
    found = _ALIAS_PAIRS.get(key)
    if found is None:
        stripped = " ".join(token for token in key.split() if token not in _NAME_PARTICLES)
        found = _ALIAS_PAIRS.get(stripped)
    if found is None:
        raise KeyError(f"没有球队对照：{name}")
    return found


def team_id_for(name: str, sportmonks_ids: dict[str, int] | None = None) -> int:
    """Prefer the id already used in ``matches.csv``. Otherwise use the synthetic id."""

    canonical = canonical_team(name)
    if sportmonks_ids and canonical in sportmonks_ids:
        return int(sportmonks_ids[canonical])
    return SYNTHETIC_TEAM_IDS[canonical]

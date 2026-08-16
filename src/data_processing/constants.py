#!/usr/bin/env python3
"""
Common constants and station metadata for the data processing package.
"""

from typing import Any, Dict

# Central station metadata definition
STATION_METADATA: Dict[str, Dict[str, Any]] = {
    "ZSPD": {
        "station_id": "ZSPD",
        "name": "Shanghai Pudong International Airport",
        "city": "shanghai",
        "country": "cn",
        "latitude": 31.15,
        "longitude": 121.80,
        "elevation": 4.0,
        "timezone": "Asia/Shanghai",
        "temperature_unit": "C",
        "polymarket_id": "shanghai",
    },
    "shanghai": {
        "station_id": "ZSPD",
        "name": "Shanghai Pudong International Airport",
        "city": "shanghai",
        "country": "cn",
        "latitude": 31.15,
        "longitude": 121.80,
        "elevation": 4.0,
        "timezone": "Asia/Shanghai",
        "temperature_unit": "C",
        "polymarket_id": "shanghai",
    },
    "KDEN": {
        "station_id": "KDEN",
        "name": "Denver International Airport",
        "city": "denver",
        "country": "us",
        "latitude": 39.86,
        "longitude": -104.67,
        "elevation": 1655.0,
        "timezone": "America/Denver",
        "temperature_unit": "F",
        "polymarket_id": "denver",
    },
    "denver": {
        "station_id": "KDEN",
        "name": "Denver International Airport",
        "city": "denver",
        "country": "us",
        "latitude": 39.86,
        "longitude": -104.67,
        "elevation": 1655.0,
        "timezone": "America/Denver",
        "temperature_unit": "F",
        "polymarket_id": "denver",
    },
}

# Aliases for backward compatibility and specialized lookups
STATION_COORDINATES: Dict[str, Dict[str, Any]] = {
    k: {
        "latitude": v["latitude"],
        "longitude": v["longitude"],
        "elevation": v["elevation"],
        "name": v["name"],
        "timezone": v["timezone"],
    }
    for k, v in STATION_METADATA.items()
}

STATION_TIMEZONES: Dict[str, str] = {
    k: v["timezone"] for k, v in STATION_METADATA.items()
}

STATION_DEFAULT_UNITS: Dict[str, str] = {
    k: v["temperature_unit"] for k, v in STATION_METADATA.items()
}

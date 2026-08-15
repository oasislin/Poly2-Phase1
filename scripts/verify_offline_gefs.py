#!/usr/bin/env python3
"""
Offline verification of Task 1.2 real-data tests (NO network access).

Replays the key GEFS download→decode→crop paths against the REAL GRIB subsets
already cached under data/raw/gefs_probe/, and asserts the recorded golden
values (window counts, valid_time, grid, dtype). If the pipeline or the cached
data ever looks wrong, run this FIRST — it needs no network.

Usage:
  python scripts/verify_offline_gefs.py

See docs/handoffs/2026-08-15-handoff-T12-real-data-checks.md for the full record
of each case and how the golden values were obtained.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
import cfgrib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_acquisition.gefs_fetcher import GEFSFetcher

CACHE = Path("data/raw/gefs_probe")

# Golden values recorded 2026-08-15 from real AWS downloads (single member c00).
# valid_time is UTC, ISO format (seconds precision).
GOLDEN_CASES = [
    {
        "name": "shanghai reforecast summer (fxx 24/30/36)",
        "tmax": CACHE / "gefs_reforecast/20190701/subset_19efc1fa__tmax_2m_2019070100_c00.grib2",
        "tmin": CACHE / "gefs_reforecast/20190701/subset_19efc1fa__tmin_2m_2019070100_c00.grib2",
        "region": {"lat": (25, 35), "lon": (115, 125)},
        "step": 3,
        "valid_time": [
            "2019-07-02T00:00:00",
            "2019-07-02T06:00:00",
            "2019-07-02T12:00:00",
        ],
    },
    {
        "name": "denver reforecast summer/DST (fxx 36/42/48/54)",
        "tmax": CACHE / "gefs_reforecast/20190701/subset_19efcf91__tmax_2m_2019070100_c00.grib2",
        "tmin": CACHE / "gefs_reforecast/20190701/subset_19efcf91__tmin_2m_2019070100_c00.grib2",
        "region": {"lat": (35, 45), "lon": (-110, -100)},
        "step": 4,
        "valid_time": [
            "2019-07-02T12:00:00",
            "2019-07-02T18:00:00",
            "2019-07-03T00:00:00",
            "2019-07-03T06:00:00",
        ],
    },
    {
        "name": "denver reforecast winter (fxx 42/48/54)",
        "tmax": CACHE / "gefs_reforecast/20190101/subset_97ef3872__tmax_2m_2019010100_c00.grib2",
        "tmin": CACHE / "gefs_reforecast/20190101/subset_97ef3872__tmin_2m_2019010100_c00.grib2",
        "region": {"lat": (35, 45), "lon": (-110, -100)},
        "step": 3,
        "valid_time": [
            "2019-01-02T18:00:00",
            "2019-01-03T00:00:00",
            "2019-01-03T06:00:00",
        ],
    },
    {
        "name": "shanghai realtime f006 (atmos.25)",
        "tmax": CACHE / "gefs/20240101/subset_6bb21aeb__gec00.t00z.pgrb2s.0p25.f006",
        "tmin": CACHE / "gefs/20240101/subset_6bb2cf9d__gec00.t00z.pgrb2s.0p25.f006",
        "region": {"lat": (25, 35), "lon": (115, 125)},
        "step": 1,
        "valid_time": ["2024-01-01T06:00:00"],
    },
]


def _decode_and_crop(grib_path, region):
    """Decode one cached GRIB subset (global) and crop to the region.

    cfgrib reads the local file only — no network. `indexpath=""` stops cfgrib
    from touching Herbie's `.idx` sidecars.
    """
    dss = cfgrib.open_datasets(
        grib_path,
        backend_kwargs={"indexpath": ""},
        decode_timedelta=True,
    )
    return GEFSFetcher.extract_region(dss[0], region["lat"], region["lon"])


def _valid_time_iso(ds):
    """valid_time as a list of ISO strings (scalar -> single-element list)."""
    vt = ds.valid_time.values
    if vt.ndim == 0:
        vt = np.array([vt])
    return [np.datetime_as_string(v, unit="s") for v in vt]


def check(case):
    tmax_ds = _decode_and_crop(case["tmax"], case["region"])
    tmin_ds = _decode_and_crop(case["tmin"], case["region"])
    ds = xr.merge([tmax_ds, tmin_ds], compat="override")

    errors = []

    # both variables present, float32
    for var in ("tmax", "tmin"):
        if var not in ds.data_vars:
            errors.append(f"missing data_var {var}")
        elif str(ds[var].dtype) != "float32":
            errors.append(f"{var} dtype {ds[var].dtype} != float32")

    # forecast windows: reforecast decodes them onto a `step` dim; realtime
    # single-message leaves `step` as a scalar coord. In both cases the
    # valid_time length equals the window count.
    vt = _valid_time_iso(ds)
    if len(vt) != case["step"]:
        errors.append(f"window count {len(vt)} != {case['step']}")

    # valid_time (window end, UTC)
    if vt != case["valid_time"]:
        errors.append(f"valid_time {vt} != {case['valid_time']}")

    # cropped grid: 41x41, bounds = region, 0.25 deg
    if ds.sizes.get("latitude") != 41 or ds.sizes.get("longitude") != 41:
        errors.append(
            f"grid {ds.sizes.get('latitude')}x{ds.sizes.get('longitude')} != 41x41"
        )
    if float(ds.latitude.min()) != case["region"]["lat"][0] or float(
        ds.latitude.max()
    ) != case["region"]["lat"][1]:
        errors.append(f"lat bounds wrong: {float(ds.latitude.min())}..{float(ds.latitude.max())}")

    if errors:
        print(f"FAIL  {case['name']}")
        for e in errors:
            print(f"      - {e}")
        return False
    print(f"PASS  {case['name']}  (step={case['step']}, vt={case['valid_time'][0]}..)")
    return True


def main():
    missing = [c["name"] for c in GOLDEN_CASES if not (c["tmax"].exists() and c["tmin"].exists())]
    if missing:
        print("ERROR: cached GRIB subsets missing — run scripts/probe_gefs.py first:")
        for n in missing:
            print(f"  - {n}")
        sys.exit(2)

    results = [check(c) for c in GOLDEN_CASES]
    n_fail = results.count(False)
    print(f"\n{len(results) - n_fail}/{len(results)} cases passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

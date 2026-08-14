#!/usr/bin/env python3
"""
GEFS data fetcher (Task 1.2, T01 slice).

Downloads GEFS reforecast data via Herbie (PyPI package `herbie-data`),
crops to a requested region, and returns an xarray Dataset.

T01 scope: minimal reforecast path (tmax_2m, basic cropping, cycle loop).
Caching / retry / realtime / longitude-wrapping belong to T02-T06.
"""

from datetime import date, datetime, timedelta

import xarray as xr

from herbie import Herbie

DEFAULT_REGIONS = {
    "shanghai": {"lat": (25, 35), "lon": (115, 125)},
    "denver": {"lat": (35, 45), "lon": (-110, -100)},
}

REFORECAST_CYCLES = (0, 6, 12, 18)

DEFAULT_VARIABLE = "tmax_2m"

# variable_level (file naming) -> regex matching wgrib2-style idx content
VARIABLE_SEARCH = {
    "tmax_2m": "TMAX:2 m above ground",
    "tmin_2m": "TMIN:2 m above ground",
}


class GEFSValidationError(Exception):
    """Raised when user-supplied arguments are invalid."""


class GEFSDownloadError(Exception):
    """Raised when a download/read from the data source fails."""


class GEFSFetcher:
    """Fetch GEFS data and return regional xarray Datasets."""

    def __init__(self, cache_dir="/tmp/gefs_cache", verbose=False):
        self.cache_dir = cache_dir
        self.verbose = verbose

    def download_reforecast(
        self,
        region_bounds,
        date_range,
        members,
        cycles=None,
        variable=DEFAULT_VARIABLE,
        forecast_hours=None,
    ):
        """Download GEFS reforecast for every (day, cycle, member) and
        concatenate the region-cropped Datasets along the time dimension."""
        self._validate(date_range, region_bounds)
        start, end = date_range
        if cycles is None:
            cycles = REFORECAST_CYCLES

        days = [
            start + timedelta(days=i) for i in range((end - start).days + 1)
        ]
        datasets = []
        for day in days:
            for cycle in cycles:
                for member in members:
                    init_time = datetime(
                        day.year, day.month, day.day, cycle
                    )
                    h = Herbie(
                        init_time,
                        model="gefs_reforecast",
                        member=member,
                        fxx=0,
                        variable_level=variable,
                        save_dir=self.cache_dir,
                        verbose=self.verbose,
                    )
                    search = self._build_search(variable, forecast_hours)
                    h.download(search=search)
                    ds = h.xarray(search=search)
                    ds = self.extract_region(
                        ds, region_bounds["lat"], region_bounds["lon"]
                    )
                    ds = ds.expand_dims(member=[member])
                    datasets.append(ds)
        return xr.concat(datasets, dim="time")

    @staticmethod
    def _build_search(variable, forecast_hours):
        """Build a regex matching idx lines (wgrib2 style) for the requested
        variable and forecast hours; falls back to the variable name itself."""
        base = VARIABLE_SEARCH.get(variable, variable)
        if not forecast_hours:
            return base
        windows = "|".join(
            (f"{h - 6}-{h} hour max fcst" if h % 6 == 0 else f"{h - 3}-{h} hour max fcst")
            for h in forecast_hours
        )
        return rf"{base}:(?:{windows})"

    @staticmethod
    def extract_region(ds, lat_bounds, lon_bounds):
        """Crop a Dataset to the given lat/lon window (basic slicing).
        GEFS global grids have descending latitudes, so sort ascending first."""
        ds = ds.sortby("latitude")
        return ds.sel(
            latitude=slice(*lat_bounds), longitude=slice(*lon_bounds)
        )

    @staticmethod
    def _validate(date_range, region_bounds):
        start, end = date_range
        if start > end:
            raise GEFSValidationError(
                f"date_range start {start} is after end {end}"
            )
        lat_lo, lat_hi = region_bounds["lat"]
        if not (-90 <= lat_lo < lat_hi <= 90):
            raise GEFSValidationError(
                f"invalid latitude bounds {region_bounds['lat']}"
            )

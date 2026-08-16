#!/usr/bin/env python3
"""
Time alignment, window selection, and astronomical sunrise module (Task 1.3 T1.3-03).

1. Aligns forecast model timelines to station local calendar days (00:00 - 24:00 LT)
   and selects 6h TMAX/TMIN forecast windows that are COMPLETELY CONTAINED (subseteq)
   within the target local day.
2. Implements NOAA astronomical sunrise algorithm and verifies that selected 6h
   windows safely cover the minimum temperature sensitive window [Sunrise - 1h, Sunrise + 0.5h].
"""

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from src.data_processing.constants import STATION_COORDINATES, STATION_TIMEZONES

logger = logging.getLogger(__name__)


class TimeAlignmentError(Exception):
    """Raised when time alignment or timezone resolution fails."""


@dataclass(frozen=True)
class ForecastWindow:
    """Represents a 6-hour forecast window."""

    fxx: int
    start_utc: datetime
    end_utc: datetime
    start_lt: datetime
    end_lt: datetime


def _resolve_timezone(tz: str) -> ZoneInfo:
    """Resolve timezone string or alias to a ZoneInfo object."""
    tz_str = STATION_TIMEZONES.get(tz, tz)
    try:
        return ZoneInfo(tz_str)
    except Exception as exc:
        raise TimeAlignmentError(f"Invalid timezone: {tz}") from exc


def get_local_day_bounds_utc(
    target_date: date, tz_or_station: str
) -> Tuple[datetime, datetime]:
    """Calculate UTC start and end datetimes for a local calendar day (00:00 to 24:00 LT).

    Handles daylight saving time (DST) shifts automatically.
    Returns (start_utc, end_utc) datetimes in UTC.
    """
    tz = _resolve_timezone(tz_or_station)

    # Local midnight start (00:00:00 LT)
    start_lt = datetime.combine(target_date, time(0, 0, 0), tzinfo=tz)
    # Next day midnight end (24:00:00 LT)
    end_lt = datetime.combine(
        target_date + timedelta(days=1), time(0, 0, 0), tzinfo=tz
    )

    # Convert to UTC
    start_utc = start_lt.astimezone(timezone.utc)
    end_utc = end_lt.astimezone(timezone.utc)

    return start_utc, end_utc


def select_contained_6h_windows(
    init_time_utc: datetime,
    target_date: date,
    tz_or_station: str,
    max_lead_hours: int = 120,
) -> List[int]:
    """Select forecast lead step hours (fxx in {6, 12, 18, ...}) where the 6h
    forecast interval [init + (fxx - 6)h, init + fxx h] is COMPLETELY CONTAINED
    (⊆) within the target local day [00:00 LT, 24:00 LT].
    """
    if init_time_utc.tzinfo is None:
        init_utc = init_time_utc.replace(tzinfo=timezone.utc)
    else:
        init_utc = init_time_utc.astimezone(timezone.utc)

    local_start_utc, local_end_utc = get_local_day_bounds_utc(
        target_date, tz_or_station
    )

    contained_fxx = []
    for fxx in range(6, max_lead_hours + 1, 6):
        win_start_utc = init_utc + timedelta(hours=fxx - 6)
        win_end_utc = init_utc + timedelta(hours=fxx)

        # Completely contained rule: start >= local_start and end <= local_end
        if win_start_utc >= local_start_utc and win_end_utc <= local_end_utc:
            contained_fxx.append(fxx)

    return contained_fxx


def select_contained_window_objects(
    init_time_utc: datetime,
    target_date: date,
    tz_or_station: str,
    max_lead_hours: int = 120,
) -> List[ForecastWindow]:
    """Select completely contained 6h forecast windows, returning rich ForecastWindow objects."""
    tz = _resolve_timezone(tz_or_station)
    if init_time_utc.tzinfo is None:
        init_utc = init_time_utc.replace(tzinfo=timezone.utc)
    else:
        init_utc = init_time_utc.astimezone(timezone.utc)

    fxx_list = select_contained_6h_windows(
        init_time_utc, target_date, tz_or_station, max_lead_hours=max_lead_hours
    )

    windows = []
    for fxx in fxx_list:
        win_start_utc = init_utc + timedelta(hours=fxx - 6)
        win_end_utc = init_utc + timedelta(hours=fxx)
        win_start_lt = win_start_utc.astimezone(tz)
        win_end_lt = win_end_utc.astimezone(tz)

        windows.append(
            ForecastWindow(
                fxx=fxx,
                start_utc=win_start_utc,
                end_utc=win_end_utc,
                start_lt=win_start_lt,
                end_lt=win_end_lt,
            )
        )
    return windows


def _calculate_solar_coordinates(t: float) -> Tuple[float, float, float]:
    """Calculate Sun's (sin_dec, cos_dec, RA_hour) for approximate time t in DOY."""
    M_rad = math.radians((0.9856 * t) - 3.289)
    L_deg = ((0.9856 * t) - 3.289 + 1.916 * math.sin(M_rad) + 0.020 * math.sin(2 * M_rad) + 282.634) % 360.0
    L_rad = math.radians(L_deg)

    RA = (math.degrees(math.atan(0.91764 * math.tan(L_rad)))) % 360.0
    L_quadrant = (math.floor(L_deg / 90.0)) * 90.0
    RA_quadrant = (math.floor(RA / 90.0)) * 90.0
    RA_hour = (RA + (L_quadrant - RA_quadrant)) / 15.0

    sin_dec = 0.39782 * math.sin(L_rad)
    cos_dec = math.cos(math.asin(sin_dec))
    return sin_dec, cos_dec, RA_hour


def _calculate_sunrise_hour_angle(lat_deg: float, sin_dec: float, cos_dec: float) -> Optional[float]:
    """Calculate sunrise local hour angle in hours, or None for polar conditions."""
    lat_rad = math.radians(lat_deg)
    zenith_rad = math.radians(90.833)  # Official sunrise zenith angle with atmospheric refraction
    cos_H = (math.cos(zenith_rad) - (sin_dec * math.sin(lat_rad))) / (cos_dec * math.cos(lat_rad))

    if cos_H > 1.0 or cos_H < -1.0:
        return None
    H_deg = 360.0 - math.degrees(math.acos(cos_H))
    return H_deg / 15.0


def _build_localized_sunrise_datetime(
    T: float, lon_hour: float, target_date: date, tz: ZoneInfo
) -> datetime:
    """Construct UTC and localized sunrise datetime matching target_date."""
    UT = (T - lon_hour) % 24.0
    ut_h = int(UT)
    ut_m = int((UT - ut_h) * 60)
    ut_s = int((((UT - ut_h) * 60) - ut_m) * 60)

    sunrise_utc = datetime(
        target_date.year, target_date.month, target_date.day,
        ut_h, ut_m, ut_s, tzinfo=timezone.utc
    )
    sunrise_lt = sunrise_utc.astimezone(tz)

    if sunrise_lt.date() < target_date:
        sunrise_lt = (sunrise_utc + timedelta(days=1)).astimezone(tz)
    elif sunrise_lt.date() > target_date:
        sunrise_lt = (sunrise_utc - timedelta(days=1)).astimezone(tz)

    return sunrise_lt


def calculate_sunrise_time(
    target_date: date,
    latitude: float,
    longitude: float,
    tz_str: str,
) -> datetime:
    """Calculate astronomical sunrise time using standard NOAA solar calculation algorithm."""
    tz = _resolve_timezone(tz_str)
    lon_deg = (longitude + 180.0) % 360.0 - 180.0
    lon_hour = lon_deg / 15.0

    N = target_date.timetuple().tm_yday
    t = N + ((6.0 - lon_hour) / 24.0)

    sin_dec, cos_dec, RA_hour = _calculate_solar_coordinates(t)
    H_hour = _calculate_sunrise_hour_angle(latitude, sin_dec, cos_dec)

    if H_hour is None:
        # Polar night / midnight sun defaults
        return datetime.combine(target_date, time(6, 0, 0), tzinfo=tz)

    T = H_hour + RA_hour - (0.06571 * t) - 6.622
    return _build_localized_sunrise_datetime(T, lon_hour, target_date, tz)


def _resolve_station_metadata(
    station_id: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    tz_str: Optional[str],
) -> Tuple[float, float, str]:
    """Resolve latitude, longitude, and timezone from station_id or direct arguments."""
    if station_id in STATION_COORDINATES:
        meta = STATION_COORDINATES[station_id]
        return meta["latitude"], meta["longitude"], meta["timezone"]
    if latitude is not None and longitude is not None and tz_str is not None:
        return latitude, longitude, tz_str
    raise ValueError("Either a known station_id or (latitude, longitude, tz_str) must be provided.")


def verify_sunrise_coverage(
    windows: List[ForecastWindow],
    target_date: date,
    station_id: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    tz_str: Optional[str] = None,
    margin_before_hours: float = 1.0,
    margin_after_hours: float = 0.5,
) -> Tuple[bool, str]:
    """Verify whether selected 6h forecast windows cover [Sunrise - 1h, Sunrise + 0.5h]."""
    lat, lon, tz = _resolve_station_metadata(station_id, latitude, longitude, tz_str)
    st_label = station_id or "station"

    if not windows:
        msg = f"WARNING: No forecast windows available for {st_label} on {target_date}."
        logger.warning(msg)
        return False, msg

    sunrise_lt = calculate_sunrise_time(target_date, lat, lon, tz)
    req_start = sunrise_lt - timedelta(hours=margin_before_hours)
    req_end = sunrise_lt + timedelta(hours=margin_after_hours)

    win_start_min = min(w.start_lt for w in windows)
    win_end_max = max(w.end_lt for w in windows)

    is_covered = (win_start_min <= req_start) and (win_end_max >= req_end)
    if is_covered:
        msg = f"Sunrise span [{req_start.strftime('%H:%M')}, {req_end.strftime('%H:%M')}] is fully covered."
        return True, msg

    msg = (
        f"WARNING: Insufficient window coverage for min temperature on {target_date} at {st_label}. "
        f"Required sunrise span: [{req_start.strftime('%H:%M')}, {req_end.strftime('%H:%M')}], "
        f"Available window span: [{win_start_min.strftime('%H:%M')}, {win_end_max.strftime('%H:%M')}]."
    )
    logger.warning(msg)
    return False, msg


class TimeAligner:
    """Helper class providing time alignment, window selection, and sunrise validation utilities."""

    def __init__(self, default_station: Optional[str] = None):
        self.default_station = default_station

    def get_contained_windows(
        self,
        init_time_utc: datetime,
        target_date: date,
        station_or_tz: Optional[str] = None,
        max_lead_hours: int = 120,
    ) -> List[int]:
        """Get list of contained 6h forecast fxx hours."""
        station = station_or_tz or self.default_station
        if not station:
            raise TimeAlignmentError("Station or timezone must be specified.")
        return select_contained_6h_windows(
            init_time_utc, target_date, station, max_lead_hours=max_lead_hours
        )

    def get_contained_window_objects(
        self,
        init_time_utc: datetime,
        target_date: date,
        station_or_tz: Optional[str] = None,
        max_lead_hours: int = 120,
    ) -> List[ForecastWindow]:
        """Get list of contained ForecastWindow objects."""
        station = station_or_tz or self.default_station
        if not station:
            raise TimeAlignmentError("Station or timezone must be specified.")
        return select_contained_window_objects(
            init_time_utc, target_date, station, max_lead_hours=max_lead_hours
        )

    def get_local_bounds(
        self, target_date: date, station_or_tz: Optional[str] = None
    ) -> Tuple[datetime, datetime]:
        """Get UTC bounds of target local calendar day."""
        station = station_or_tz or self.default_station
        if not station:
            raise TimeAlignmentError("Station or timezone must be specified.")
        return get_local_day_bounds_utc(target_date, station)

    def check_sunrise_coverage(
        self,
        init_time_utc: datetime,
        target_date: date,
        station_id: Optional[str] = None,
        max_lead_hours: int = 120,
    ) -> Tuple[bool, str]:
        """Verify sunrise sensitivity coverage for a station and init forecast."""
        st = station_id or self.default_station
        if not st:
            raise TimeAlignmentError("station_id must be specified.")

        windows = self.get_contained_window_objects(
            init_time_utc, target_date, station_or_tz=st, max_lead_hours=max_lead_hours
        )
        return verify_sunrise_coverage(windows, target_date, station_id=st)

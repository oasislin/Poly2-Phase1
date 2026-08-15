#!/usr/bin/env python3
"""
Time alignment and forecast window selection module (Task 1.2 T05 / Task 1.3).

Aligns forecast model timelines to station local calendar days (00:00 - 24:00 LT)
and selects 6h TMAX/TMIN forecast windows that are COMPLETELY CONTAINED (subseteq)
within the target local day.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

STATION_TIMEZONES = {
    "shanghai": "Asia/Shanghai",
    "ZSPD": "Asia/Shanghai",
    "denver": "America/Denver",
    "KDEN": "America/Denver",
}


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
) -> tuple[datetime, datetime]:
    """Calculate UTC start and end datetimes for a local calendar day (00:00 to 24:00 LT).

    Handles daylight saving time (DST) shifts automatically.
    Returns naive or UTC-aware (start_utc, end_utc) datetimes in UTC.
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
) -> list[int]:
    """Select forecast lead step hours (fxx in {6, 12, 18, ...}) where the 6h
    forecast interval [init + (fxx - 6)h, init + fxx h] is COMPLETELY CONTAINED
    (⊆) within the target local day [00:00 LT, 24:00 LT].

    Partial overlapping windows (offset mod 6 != 0 at head/tail) are strictly discarded
    to eliminate cross-day boundary ambiguity (v5.7 §2.2 / v5.9.1 §1).

    Parameters
    ----------
    init_time_utc : datetime
        GEFS initialization time in UTC (e.g. 2019-07-01 00:00 UTC).
    target_date : date
        Target local calendar day.
    tz_or_station : str
        Station ID ('ZSPD', 'KDEN') or timezone name ('Asia/Shanghai', 'America/Denver').
    max_lead_hours : int, default 120
        Maximum forecast hour to search.

    Returns
    -------
    list[int]
        List of matching fxx step hours (e.g. [24, 30, 36]).
    """
    if init_time_utc.tzinfo is None:
        init_utc = init_time_utc.replace(tzinfo=timezone.utc)
    else:
        init_utc = init_time_utc.astimezone(timezone.utc)

    local_start_utc, local_end_utc = get_local_day_bounds_utc(
        target_date, tz_or_station
    )

    contained_fxx = []
    # 6h forecast intervals: fxx=6 is [0, 6], fxx=12 is [6, 12], etc.
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
) -> list[ForecastWindow]:
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


class TimeAligner:
    """Helper class providing time alignment and window selection utilities."""

    def __init__(self, default_station: str = None):
        self.default_station = default_station

    def get_contained_windows(
        self,
        init_time_utc: datetime,
        target_date: date,
        station_or_tz: str = None,
        max_lead_hours: int = 120,
    ) -> list[int]:
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
        station_or_tz: str = None,
        max_lead_hours: int = 120,
    ) -> list[ForecastWindow]:
        """Get list of contained ForecastWindow objects."""
        station = station_or_tz or self.default_station
        if not station:
            raise TimeAlignmentError("Station or timezone must be specified.")
        return select_contained_window_objects(
            init_time_utc, target_date, station, max_lead_hours=max_lead_hours
        )

    def get_local_bounds(
        self, target_date: date, station_or_tz: str = None
    ) -> tuple[datetime, datetime]:
        """Get UTC bounds of target local calendar day."""
        station = station_or_tz or self.default_station
        if not station:
            raise TimeAlignmentError("Station or timezone must be specified.")
        return get_local_day_bounds_utc(target_date, station)

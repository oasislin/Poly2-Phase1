"""Unit tests for time alignment and 6h window selection (Task 1.2 T05 / Task 1.3).

Tests the 'completely contained' (subseteq) window selection rule across:
- Shanghai (UTC+8, offset mod 6 != 0, 3 windows)
- Denver Summer (UTC-6, offset mod 6 == 0, 4 windows, 0 discarded)
- Denver Winter (UTC-7, offset mod 6 != 0, 3 windows)
- Denver DST transition dates
"""

from datetime import date, datetime, timezone

import pytest

from src.data_processing.time_aligner import (
    ForecastWindow,
    TimeAlignmentError,
    TimeAligner,
    get_local_day_bounds_utc,
    select_contained_6h_windows,
    select_contained_window_objects,
)


class TestShanghaiWindowSelection:
    """Shanghai: Asia/Shanghai (UTC+8). offset mod 6 = 2 != 0 -> 3 windows."""

    def test_shanghai_d_plus_1_target(self):
        # Init: 2019-07-01 00:00 UTC, Target: 2019-07-02 (D+1)
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        windows = select_contained_6h_windows(
            init_time_utc=init_time,
            target_date=target_date,
            tz_or_station="ZSPD",
        )

        # In UTC, 2019-07-02 LT [00:00, 24:00] is [2019-07-01 16:00, 2019-07-02 16:00] UTC.
        # Contained 6h intervals:
        # fxx=24: [18Z, 24Z] (LT 02:00-08:00)
        # fxx=30: [24Z, 30Z] (LT 08:00-14:00)
        # fxx=36: [30Z, 36Z] (LT 14:00-20:00)
        assert windows == [24, 30, 36]
        assert len(windows) == 3

    def test_shanghai_window_objects_spans(self):
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        objs = select_contained_window_objects(
            init_time_utc=init_time,
            target_date=target_date,
            tz_or_station="shanghai",
        )

        assert len(objs) == 3
        # First window LT span: 02:00 - 08:00
        assert objs[0].start_lt.hour == 2
        assert objs[0].end_lt.hour == 8
        # Third window LT span: 14:00 - 20:00
        assert objs[2].start_lt.hour == 14
        assert objs[2].end_lt.hour == 20
        # Overall coverage span is 02:00 - 20:00 LT (18 hours)
        assert (objs[2].end_lt - objs[0].start_lt).total_seconds() == 18 * 3600

    def test_shanghai_d_plus_2_target(self):
        # Init: 2019-07-01 00:00 UTC, Target: 2019-07-03 (D+2)
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 3)

        windows = select_contained_6h_windows(
            init_time_utc=init_time,
            target_date=target_date,
            tz_or_station="ZSPD",
        )
        assert windows == [48, 54, 60]
        assert len(windows) == 3


class TestDenverWindowSelection:
    """Denver: America/Denver.

    Summer (MDT, UTC-6): offset mod 6 == 0 -> 4 windows (0 discarded).
    Winter (MST, UTC-7): offset mod 6 = 5 != 0 -> 3 windows.
    """

    def test_denver_summer_yields_4_windows(self):
        # Init: 2019-07-01 00:00 UTC, Target: 2019-07-02 (Summer MDT, UTC-6)
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        windows = select_contained_6h_windows(
            init_time_utc=init_time,
            target_date=target_date,
            tz_or_station="KDEN",
        )

        # In UTC, 2019-07-02 MDT is [2019-07-02 06:00, 2019-07-03 06:00] UTC.
        # Intervals relative to 2019-07-01 00:00:
        # fxx=36: [30h, 36h] = [06Z, 12Z] (LT 00:00-06:00)
        # fxx=42: [36h, 42h] = [12Z, 18Z] (LT 06:00-12:00)
        # fxx=48: [42h, 48h] = [18Z, 24Z] (LT 12:00-18:00)
        # fxx=54: [48h, 54h] = [24Z, 30Z] (LT 18:00-24:00)
        assert windows == [36, 42, 48, 54]
        assert len(windows) == 4

    def test_denver_winter_yields_3_windows(self):
        # Init: 2019-01-01 00:00 UTC, Target: 2019-01-02 (Winter MST, UTC-7)
        init_time = datetime(2019, 1, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 1, 2)

        windows = select_contained_6h_windows(
            init_time_utc=init_time,
            target_date=target_date,
            tz_or_station="KDEN",
        )

        # In UTC, 2019-01-02 MST is [2019-01-02 07:00, 2019-01-03 07:00] UTC.
        # fxx=36 [30h, 36h] = [06Z, 12Z] starts at 06Z < 07Z -> DISCARDED (partial)
        # fxx=42 [36h, 42h] = [12Z, 18Z] (LT 05:00-11:00) -> INCLUDED
        # fxx=48 [42h, 48h] = [18Z, 24Z] (LT 11:00-17:00) -> INCLUDED
        # fxx=54 [48h, 54h] = [24Z, 30Z] (LT 17:00-23:00) -> INCLUDED
        # fxx=60 [54h, 60h] = [30Z, 36Z] ends at 12Z > 07Z -> DISCARDED (partial)
        assert windows == [42, 48, 54]
        assert len(windows) == 3

    def test_denver_dst_spring_forward(self):
        # 2019-03-10: DST begins in US (23-hour day)
        start_utc, end_utc = get_local_day_bounds_utc(
            date(2019, 3, 10), "America/Denver"
        )
        assert (end_utc - start_utc).total_seconds() == 23 * 3600

    def test_denver_dst_fall_back(self):
        # 2019-11-03: DST ends in US (25-hour day)
        start_utc, end_utc = get_local_day_bounds_utc(
            date(2019, 11, 3), "America/Denver"
        )
        assert (end_utc - start_utc).total_seconds() == 25 * 3600


class TestTimeAlignerClass:
    def test_aligner_wrapper(self):
        aligner = TimeAligner(default_station="ZSPD")
        init_time = datetime(2019, 7, 1, 0, 0, tzinfo=timezone.utc)
        target_date = date(2019, 7, 2)

        windows = aligner.get_contained_windows(init_time, target_date)
        assert windows == [24, 30, 36]

        objs = aligner.get_contained_window_objects(init_time, target_date)
        assert len(objs) == 3
        assert isinstance(objs[0], ForecastWindow)

    def test_invalid_timezone_raises(self):
        with pytest.raises(TimeAlignmentError):
            get_local_day_bounds_utc(date(2019, 1, 1), "NonExistent/Timezone")

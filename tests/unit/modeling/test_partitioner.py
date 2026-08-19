#!/usr/bin/env python3
"""
Unit tests for DatasetPartitioner (Ticket 2.2-03 / Issue #16).

Verifies:
1. round_to_nearest_6h properly rounds continuous lead times.
2. get_season properly maps months (3-5: Spring, 6-8: Summer, 9-11: Autumn, 12-2: Winter).
3. Nominal target time calculation (15:00 LT for Max, 06:00 LT for Min) with station UTC offsets (ZSPD: UTC+8, KDEN: UTC-7).
4. Mapping to the 5 discrete matrix training nodes (Max: 54h, 30h, 6h; Min: 48h, 24h).
5. Dataset partitioning producing exact 40 matrix subsets (2 stations x 4 seasons x 5 nodes).
"""

from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from src.modeling.partitioner import DatasetPartitioner


class TestLeadTimeCalculations:
    """Test rounding and nominal local time lead calculations."""

    @pytest.mark.parametrize("input_hours,expected_bucket", [
        (5.8, 6),
        (6.0, 6),
        (8.9, 6),
        (9.1, 12),
        (29.5, 30),
        (32.9, 30),
        (53.2, 54),
        (47.8, 48),
        (24.1, 24),
    ])
    def test_round_to_nearest_6h(self, input_hours, expected_bucket):
        assert DatasetPartitioner.round_to_nearest_6h(input_hours) == expected_bucket

    def test_nominal_lead_time_zspd(self):
        """ZSPD is UTC+8.
        00Z init today to today 15:00 LT Max:
        Today 15:00 LT = Today 07:00 UTC -> Lead time = 7h -> round_to_nearest_6h = 6h.
        00Z init today to tomorrow 15:00 LT Max:
        Tomorrow 15:00 LT = Tomorrow 07:00 UTC -> Lead time = 31h -> round_to_nearest_6h = 30h.
        00Z init today to day-after-tomorrow 15:00 LT Max:
        Day+2 15:00 LT = Day+2 07:00 UTC -> Lead time = 55h -> round_to_nearest_6h = 54h.
        """
        partitioner = DatasetPartitioner()
        
        # Today max from today 00Z
        lead_today_max = partitioner.compute_nominal_lead_hours(
            station_id="ZSPD",
            target_type="max",
            init_datetime=datetime(2019, 1, 1, 0, 0),
            target_date=date(2019, 1, 1),
        )
        assert np.isclose(lead_today_max, 7.0)
        assert DatasetPartitioner.round_to_nearest_6h(lead_today_max) == 6

        # Tomorrow max from today 00Z
        lead_tomorrow_max = partitioner.compute_nominal_lead_hours(
            station_id="ZSPD",
            target_type="max",
            init_datetime=datetime(2019, 1, 1, 0, 0),
            target_date=date(2019, 1, 2),
        )
        assert np.isclose(lead_tomorrow_max, 31.0)
        assert DatasetPartitioner.round_to_nearest_6h(lead_tomorrow_max) == 30

        # Day+2 max from today 00Z
        lead_day2_max = partitioner.compute_nominal_lead_hours(
            station_id="ZSPD",
            target_type="max",
            init_datetime=datetime(2019, 1, 1, 0, 0),
            target_date=date(2019, 1, 3),
        )
        assert np.isclose(lead_day2_max, 55.0)
        assert DatasetPartitioner.round_to_nearest_6h(lead_day2_max) == 54

    def test_nominal_lead_time_min_temp_zspd(self):
        """ZSPD is UTC+8.
        00Z init today to tomorrow 06:00 LT Min:
        Tomorrow 06:00 LT = Today 22:00 UTC -> Lead time = 22h -> round_to_nearest_6h = 24h.
        00Z init today to day+2 06:00 LT Min:
        Day+2 06:00 LT = Tomorrow 22:00 UTC -> Lead time = 46h -> round_to_nearest_6h = 48h.
        """
        partitioner = DatasetPartitioner()
        lead_min_day1 = partitioner.compute_nominal_lead_hours(
            station_id="ZSPD",
            target_type="min",
            init_datetime=datetime(2019, 1, 1, 0, 0),
            target_date=date(2019, 1, 2),
        )
        assert np.isclose(lead_min_day1, 22.0)
        assert DatasetPartitioner.round_to_nearest_6h(lead_min_day1) == 24

        lead_min_day2 = partitioner.compute_nominal_lead_hours(
            station_id="ZSPD",
            target_type="min",
            init_datetime=datetime(2019, 1, 1, 0, 0),
            target_date=date(2019, 1, 3),
        )
        assert np.isclose(lead_min_day2, 46.0)
        assert DatasetPartitioner.round_to_nearest_6h(lead_min_day2) == 48


class TestSeasonalPartitioning:
    """Test 4-season grouping and slicing."""

    @pytest.mark.parametrize("month,expected_season", [
        (3, "Spring"), (4, "Spring"), (5, "Spring"),
        (6, "Summer"), (7, "Summer"), (8, "Summer"),
        (9, "Autumn"), (10, "Autumn"), (11, "Autumn"),
        (12, "Winter"), (1, "Winter"), (2, "Winter"),
    ])
    def test_get_season_from_month(self, month, expected_season):
        assert DatasetPartitioner.get_season(month) == expected_season
        assert DatasetPartitioner.get_season(date(2019, month, 15)) == expected_season

    def test_split_dataframe_by_season(self):
        dates = pd.date_range("2019-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({
            "target_date": dates.strftime("%Y-%m-%d"),
            "temp": np.random.normal(20, 5, len(dates)),
        })

        partitioner = DatasetPartitioner()
        seasonal_dfs = partitioner.split_by_season(df, date_col="target_date")

        assert set(seasonal_dfs.keys()) == {"Spring", "Summer", "Autumn", "Winter"}
        assert len(seasonal_dfs["Winter"]) == 31 + 28 + 31  # Jan(31) + Feb(28) + Dec(31) = 90
        assert len(seasonal_dfs["Spring"]) == 31 + 30 + 31  # Mar(31) + Apr(30) + May(31) = 92
        assert len(seasonal_dfs["Summer"]) == 30 + 31 + 31  # Jun(30) + Jul(31) + Aug(31) = 92
        assert len(seasonal_dfs["Autumn"]) == 30 + 31 + 30  # Sep(30) + Oct(31) + Nov(30) = 91
        assert sum(len(v) for v in seasonal_dfs.values()) == 365


class TestMatrixPartitionsGeneration:
    """Test generating the 40 standard training matrix partitions."""

    def test_get_all_matrix_keys(self):
        partitioner = DatasetPartitioner()
        keys = partitioner.get_all_matrix_keys()
        
        # 2 stations * 4 seasons * (3 max nodes + 2 min nodes = 5 nodes) = 40 keys
        assert len(keys) == 40
        
        # Check components
        stations = {k[0] for k in keys}
        seasons = {k[1] for k in keys}
        target_types = {k[2] for k in keys}
        lead_buckets = {k[3] for k in keys}

        assert stations == {"ZSPD", "KDEN"}
        assert seasons == {"Spring", "Summer", "Autumn", "Winter"}
        assert target_types == {"max", "min"}
        assert lead_buckets == {6, 24, 30, 48, 54}

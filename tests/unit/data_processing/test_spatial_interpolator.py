"""Unit tests for spatial bilinear interpolator (Task 1.3 T1.3-02)."""

import numpy as np
import pytest
import xarray as xr

from src.data_processing.spatial_interpolator import (
    STATION_COORDINATES,
    SpatialInterpolator,
    bilinear_interp_2d,
    find_surrounding_grid_indices,
    normalize_longitude,
)


class TestCoordinateUtils:
    """Test coordinate normalization and bounding box neighborhood search."""

    def test_normalize_longitude_to_180(self):
        # 255.33 -> -104.67
        assert normalize_longitude(255.33, target_system="-180_to_180") == pytest.approx(-104.67, abs=1e-4)
        assert normalize_longitude(-104.67, target_system="-180_to_180") == pytest.approx(-104.67, abs=1e-4)
        assert normalize_longitude(121.80, target_system="-180_to_180") == pytest.approx(121.80, abs=1e-4)

    def test_normalize_longitude_to_360(self):
        # -104.67 -> 255.33
        assert normalize_longitude(-104.67, target_system="0_to_360") == pytest.approx(255.33, abs=1e-4)
        assert normalize_longitude(255.33, target_system="0_to_360") == pytest.approx(255.33, abs=1e-4)
        assert normalize_longitude(121.80, target_system="0_to_360") == pytest.approx(121.80, abs=1e-4)

    def test_find_surrounding_grid_ascending_coordinates(self):
        lats = np.array([25.0, 25.25, 25.5, 25.75, 26.0])
        lons = np.array([115.0, 115.25, 115.5, 115.75, 116.0])

        # Target: lat=25.4, lon=115.6
        lat_idx, lon_idx, u, v = find_surrounding_grid_indices(
            target_lat=25.4,
            target_lon=115.6,
            grid_lats=lats,
            grid_lons=lons,
        )

        assert lat_idx == (1, 2)  # [25.25, 25.5]
        assert lon_idx == (2, 3)  # [115.5, 115.75]
        # u = (25.4 - 25.25) / 0.25 = 0.15 / 0.25 = 0.6
        assert u == pytest.approx(0.6, abs=1e-4)
        # v = (115.6 - 115.5) / 0.25 = 0.1 / 0.25 = 0.4
        assert v == pytest.approx(0.4, abs=1e-4)

    def test_find_surrounding_grid_descending_latitude(self):
        # GRIB datasets often store latitude from North to South (descending)
        lats = np.array([35.0, 34.75, 34.5, 34.25, 34.0])
        lons = np.array([120.0, 120.25, 120.5, 120.75, 121.0])

        lat_idx, lon_idx, u, v = find_surrounding_grid_indices(
            target_lat=34.6,
            target_lon=120.6,
            grid_lats=lats,
            grid_lons=lons,
        )

        # 34.6 is between 34.75 (idx 1) and 34.5 (idx 2)
        assert lat_idx == (1, 2)
        assert lon_idx == (2, 3)
        # Fraction between 34.75 and 34.5: (34.75 - 34.6) / 0.25 = 0.6
        assert u == pytest.approx(0.6, abs=1e-4)
        assert v == pytest.approx(0.4, abs=1e-4)

    def test_target_out_of_bounds_raises(self):
        lats = np.array([25.0, 26.0])
        lons = np.array([115.0, 116.0])

        with pytest.raises(ValueError, match="out of grid bounds"):
            find_surrounding_grid_indices(24.0, 115.5, lats, lons)

        with pytest.raises(ValueError, match="out of grid bounds"):
            find_surrounding_grid_indices(25.5, 117.0, lats, lons)


class TestBilinearInterpolationMath:
    """Test numerical precision and exact corner/edge cases of bilinear interpolation."""

    def test_exact_corner_matches(self):
        # 2x2 grid
        grid = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])
        lats = np.array([0.0, 1.0])
        lons = np.array([0.0, 1.0])

        # Bottom-left corner (0,0) -> 10.0
        val = bilinear_interp_2d(grid, 0.0, 0.0, lats, lons)
        assert val == pytest.approx(10.0)

        # Bottom-right corner (0,1) -> 20.0
        val = bilinear_interp_2d(grid, 0.0, 1.0, lats, lons)
        assert val == pytest.approx(20.0)

        # Top-left corner (1,0) -> 30.0
        val = bilinear_interp_2d(grid, 1.0, 0.0, lats, lons)
        assert val == pytest.approx(30.0)

        # Top-right corner (1,1) -> 40.0
        val = bilinear_interp_2d(grid, 1.0, 1.0, lats, lons)
        assert val == pytest.approx(40.0)

    def test_exact_center_matches_mean_of_four_corners(self):
        grid = np.array([
            [10.0, 20.0],
            [30.0, 40.0],
        ])
        lats = np.array([0.0, 1.0])
        lons = np.array([0.0, 1.0])

        # Center (0.5, 0.5) -> (10 + 20 + 30 + 40) / 4 = 25.0
        val = bilinear_interp_2d(grid, 0.5, 0.5, lats, lons)
        assert val == pytest.approx(25.0)

    def test_linear_gradient_reproduction(self):
        # f(x, y) = 2*x + 3*y + 5
        lats = np.linspace(30.0, 32.0, 9)  # 0.25 step
        lons = np.linspace(120.0, 122.0, 9)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        grid = 2.0 * LAT + 3.0 * LON + 5.0

        target_lat = 31.15
        target_lon = 121.80
        expected = 2.0 * target_lat + 3.0 * target_lon + 5.0

        val = bilinear_interp_2d(grid, target_lat, target_lon, lats, lons)
        assert val == pytest.approx(expected, abs=1e-5)


class TestSpatialInterpolatorXarray:
    """Test SpatialInterpolator integration on multi-dimensional xarray datasets."""

    @pytest.fixture
    def mock_shanghai_grid_dataset(self):
        """Create a mock 41x41 GEFS dataset covering Shanghai region [25-35N, 115-125E]."""
        lats = np.linspace(35.0, 25.0, 41)  # Descending latitude, 0.25 deg
        lons = np.linspace(115.0, 125.0, 41)  # Ascending longitude, 0.25 deg

        # 2 members, 3 windows, 41 lat, 41 lon
        shape = (2, 3, 41, 41)
        members = ["c00", "p01"]
        windows = [24, 30, 36]

        # Construct spatial pattern where T = 20.0 + 0.1*(lat - 25) + 0.2*(lon - 115)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        base_field = 20.0 + 0.1 * (LAT - 25.0) + 0.2 * (LON - 115.0)

        tmax_data = np.zeros(shape)
        tmin_data = np.zeros(shape)
        for m in range(2):
            for w in range(3):
                tmax_data[m, w, :, :] = base_field + m * 0.5 + w * 1.0
                tmin_data[m, w, :, :] = (base_field - 10.0) + m * 0.5 + w * 1.0

        ds = xr.Dataset(
            {
                "tmax": (["member", "window", "latitude", "longitude"], tmax_data),
                "tmin": (["member", "window", "latitude", "longitude"], tmin_data),
                "orography": (["latitude", "longitude"], base_field * 5.0),
            },
            coords={
                "member": members,
                "window": windows,
                "latitude": lats,
                "longitude": lons,
            },
        )
        return ds

    def test_interpolate_shanghai_station_coordinates(self, mock_shanghai_grid_dataset):
        interpolator = SpatialInterpolator()
        # ZSPD coordinates: lat=31.15, lon=121.80
        station_ds = interpolator.interpolate_dataset(
            mock_shanghai_grid_dataset,
            target_lat=31.15,
            target_lon=121.80,
        )

        assert "latitude" not in station_ds.dims
        assert "longitude" not in station_ds.dims
        assert station_ds["tmax"].shape == (2, 3)
        assert station_ds["tmin"].shape == (2, 3)

        # Expected base field at (31.15, 121.80) = 20.0 + 0.1*(31.15 - 25.0) + 0.2*(121.80 - 115.0)
        # = 20.0 + 0.615 + 1.36 = 21.975
        expected_base = 20.0 + 0.1 * (31.15 - 25.0) + 0.2 * (121.80 - 115.0)
        assert station_ds["tmax"].sel(member="c00", window=24).values == pytest.approx(expected_base, abs=1e-4)
        assert station_ds["tmax"].sel(member="p01", window=30).values == pytest.approx(expected_base + 0.5 + 1.0, abs=1e-4)

    def test_interpolate_by_station_name(self, mock_shanghai_grid_dataset):
        interpolator = SpatialInterpolator()
        station_ds = interpolator.interpolate_station(
            mock_shanghai_grid_dataset,
            station_id="ZSPD",
        )
        assert station_ds.attrs["station_id"] == "ZSPD"
        assert station_ds.attrs["target_latitude"] == 31.15
        assert station_ds.attrs["target_longitude"] == 121.80

    def test_interpolate_denver_with_negative_longitude(self):
        # KDEN: lat 39.86, lon -104.67. Grid stored in 0-360 [250, 260]
        lats = np.linspace(45.0, 35.0, 41)
        lons_360 = np.linspace(250.0, 260.0, 41)  # 255.33 is in this range

        LAT, LON = np.meshgrid(lats, lons_360, indexing="ij")
        grid = LAT + LON

        ds = xr.Dataset(
            {"tmax": (["latitude", "longitude"], grid)},
            coords={"latitude": lats, "longitude": lons_360},
        )

        interpolator = SpatialInterpolator()
        # Query with negative lon: -104.67
        res = interpolator.interpolate_dataset(ds, target_lat=39.86, target_lon=-104.67)
        # Expected: 39.86 + 255.33 = 295.19
        assert res["tmax"].values == pytest.approx(39.86 + 255.33, abs=1e-3)

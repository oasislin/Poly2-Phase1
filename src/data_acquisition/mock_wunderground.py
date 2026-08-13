#!/usr/bin/env python3
"""
Mock Wunderground data generator for development when real API is blocked.
Generates realistic synthetic weather data for Shanghai and Denver.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWundergroundData:
    """Generate mock weather data for development"""
    
    # Seasonal patterns for Shanghai (ZSPD) and Denver (KDEN)
    SEASONAL_PATTERNS = {
        'ZSPD': {  # Shanghai, China
            'DJF': {'temp_mean': 8.0, 'temp_std': 5.0, 'temp_range': (-5, 15)},  # Winter
            'MAM': {'temp_mean': 15.0, 'temp_std': 6.0, 'temp_range': (5, 25)},   # Spring
            'JJA': {'temp_mean': 28.0, 'temp_std': 4.0, 'temp_range': (20, 38)},  # Summer
            'SON': {'temp_mean': 20.0, 'temp_std': 5.0, 'temp_range': (10, 30)},  # Fall
        },
        'KDEN': {  # Denver, USA (temperatures in Celsius)
            'DJF': {'temp_mean': 0.0, 'temp_std': 8.0, 'temp_range': (-20, 15)},  # Winter
            'MAM': {'temp_mean': 10.0, 'temp_std': 7.0, 'temp_range': (-5, 25)},   # Spring
            'JJA': {'temp_mean': 25.0, 'temp_std': 5.0, 'temp_range': (10, 35)},  # Summer
            'SON': {'temp_mean': 12.0, 'temp_std': 6.0, 'temp_range': (0, 25)},    # Fall
        }
    }
    
    @staticmethod
    def get_season(month: int) -> str:
        """Get season code from month"""
        if month in [12, 1, 2]:
            return 'DJF'  # Winter
        elif month in [3, 4, 5]:
            return 'MAM'  # Spring
        elif month in [6, 7, 8]:
            return 'JJA'  # Summer
        else:
            return 'SON'  # Fall
    
    @staticmethod
    def generate_daily_temperature(station_id: str, current_date: date, 
                                  base_temp: Optional[float] = None) -> Dict[str, float]:
        """Generate realistic daily temperatures for a station and date"""
        month = current_date.month
        season = MockWundergroundData.get_season(month)
        
        # Get seasonal parameters
        params = MockWundergroundData.SEASONAL_PATTERNS[station_id][season]
        
        # Add some randomness to mean based on day of year
        day_of_year = current_date.timetuple().tm_yday
        seasonal_variation = np.sin(2 * np.pi * day_of_year / 365) * 3
        
        # Calculate mean temperature with seasonal variation
        mean_temp = params['temp_mean'] + seasonal_variation
        
        # Add some autocorrelation (temperature similar to previous day)
        if base_temp is not None:
            mean_temp = 0.7 * base_temp + 0.3 * mean_temp
        
        # Generate max and min temperatures
        # Max is typically 8-12 degrees higher than mean
        # Min is typically 8-12 degrees lower than mean
        daily_range = np.random.uniform(8, 12)
        
        temp_max = np.random.normal(mean_temp + daily_range/2, 2)
        temp_min = np.random.normal(mean_temp - daily_range/2, 2)
        
        # Ensure min < max
        if temp_min > temp_max:
            temp_max, temp_min = temp_min, temp_max
        
        # Ensure within reasonable bounds
        temp_min = max(params['temp_range'][0], temp_min)
        temp_max = min(params['temp_range'][1], temp_max)
        
        # Generate other weather parameters
        dew_point_max = temp_max - np.random.uniform(2, 8)  # Dew point lower than temp
        dew_point_min = temp_min - np.random.uniform(2, 8)
        
        # Humidity inversely related to temperature
        humidity_avg = np.random.uniform(40, 80) if station_id == 'KDEN' else np.random.uniform(60, 90)
        
        # Wind speed (km/h)
        wind_speed_max = np.random.uniform(5, 25)
        
        # Pressure (hPa)
        pressure_avg = np.random.normal(1013, 10)
        
        # Precipitation (mm) - more likely in certain seasons
        if season in ['JJA', 'SON'] and station_id == 'ZSPD':  # Shanghai rainy season
            precipitation_prob = 0.4
        elif season in ['MAM', 'JJA'] and station_id == 'KDEN':  # Denver spring/summer storms
            precipitation_prob = 0.3
        else:
            precipitation_prob = 0.2
        
        precipitation = np.random.exponential(5) if np.random.random() < precipitation_prob else 0.0
        
        return {
            'temp_max': round(temp_max, 1),
            'temp_min': round(temp_min, 1),
            'dew_point_max': round(dew_point_max, 1),
            'dew_point_min': round(dew_point_min, 1),
            'humidity_avg': round(humidity_avg, 1),
            'wind_speed_max': round(wind_speed_max, 1),
            'pressure_avg': round(pressure_avg, 1),
            'precipitation': round(precipitation, 1),
        }
    
    @staticmethod
    def generate_station_data(station_id: str, start_date: date, end_date: date) -> List[Dict]:
        """Generate mock data for a station over a date range"""
        observations = []
        current_date = start_date
        previous_temp = None
        
        logger.info(f"Generating mock data for {station_id} from {start_date} to {end_date}")
        
        while current_date <= end_date:
            # Generate daily data with autocorrelation
            daily_data = MockWundergroundData.generate_daily_temperature(
                station_id, current_date, previous_temp
            )
            
            # Update previous temperature for autocorrelation
            previous_temp = (daily_data['temp_max'] + daily_data['temp_min']) / 2
            
            # Create observation record
            observation = {
                'station_id': station_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'year': current_date.year,
                'month': current_date.month,
                'day': current_date.day,
                'day_of_year': current_date.timetuple().tm_yday,
                'season': MockWundergroundData.get_season(current_date.month),
                **daily_data,
                'data_quality': np.random.uniform(0.8, 1.0),  # High quality mock data
                'source': 'mock',
                'created_at': datetime.now().isoformat(),
            }
            
            observations.append(observation)
            
            # Move to next day
            current_date += timedelta(days=1)
            
            # Log progress
            if len(observations) % 100 == 0:
                logger.info(f"Generated {len(observations)} days for {station_id}")
        
        logger.info(f"Completed: {len(observations)} days for {station_id}")
        return observations
    
    @staticmethod
    def save_to_parquet(observations: List[Dict], output_path: Path):
        """Save observations to Parquet file"""
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save to Parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        return df
    
    @staticmethod
    def save_to_csv(observations: List[Dict], output_path: Path):
        """Save observations to CSV file"""
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        return df
    
    @staticmethod
    def validate_data(observations: List[Dict], station_id: str) -> bool:
        """Validate generated data for consistency"""
        if not observations:
            logger.warning("No observations to validate")
            return False
        
        # Check required fields
        required_fields = ['station_id', 'date', 'temp_max', 'temp_min']
        for obs in observations:
            for field in required_fields:
                if field not in obs or obs[field] is None:
                    logger.error(f"Missing required field: {field}")
                    return False
        
        # Check temperature consistency
        for obs in observations:
            if obs['temp_min'] > obs['temp_max']:
                logger.error(f"temp_min > temp_max: {obs}")
                return False
            
            # Check reasonable ranges
            if obs['temp_max'] < -50 or obs['temp_max'] > 60:
                logger.warning(f"Unusual temp_max: {obs['temp_max']}")
            
            if obs['temp_min'] < -50 or obs['temp_min'] > 60:
                logger.warning(f"Unusual temp_min: {obs['temp_min']}")
        
        # Check station ID consistency
        station_ids = set(obs['station_id'] for obs in observations)
        if len(station_ids) != 1 or list(station_ids)[0] != station_id:
            logger.error(f"Station ID mismatch: {station_ids}")
            return False
        
        logger.info(f"Data validation passed for {station_id}")
        return True


def main():
    """Generate mock data for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock Wunderground data')
    parser.add_argument('--station', type=str, default='ZSPD', 
                       choices=['ZSPD', 'KDEN'], help='Station ID')
    parser.add_argument('--start-year', type=int, default=2000, help='Start year')
    parser.add_argument('--end-year', type=int, default=2019, help='End year')
    parser.add_argument('--output-format', type=str, default='parquet',
                       choices=['parquet', 'csv', 'json'], help='Output format')
    parser.add_argument('--output-dir', type=str, default='./data/mock_wunderground',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    start_date = date(args.start_year, 1, 1)
    end_date = date(args.end_year, 12, 31)
    
    mock_gen = MockWundergroundData()
    observations = mock_gen.generate_station_data(
        station_id=args.station,
        start_date=start_date,
        end_date=end_date
    )
    
    # Validate data
    if not mock_gen.validate_data(observations, args.station):
        logger.warning("Data validation found issues, but continuing...")
    
    # Save data
    output_path = output_dir / f"{args.station}_{args.start_year}_{args.end_year}"
    
    if args.output_format == 'parquet':
        df = mock_gen.save_to_parquet(observations, Path(f"{output_path}.parquet"))
    elif args.output_format == 'csv':
        df = mock_gen.save_to_csv(observations, Path(f"{output_path}.csv"))
    else:  # json
        output_file = output_dir / f"{args.station}_{args.start_year}_{args.end_year}.json"
        with open(output_file, 'w') as f:
            json.dump(observations, f, indent=2, default=str)
        logger.info(f"Saved {len(observations)} records to {output_file}")
        df = pd.DataFrame(observations)
    
    # Print summary
    print(f"\nGenerated mock data for {args.station} ({args.start_year}-{args.end_year}):")
    print(f"  Total days: {len(df)}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Temperature range: {df['temp_min'].min():.1f}°C to {df['temp_max'].max():.1f}°C")
    print(f"  Average max temp: {df['temp_max'].mean():.1f}°C")
    print(f"  Average min temp: {df['temp_min'].mean():.1f}°C")
    
    # Show sample
    print(f"\nSample data (first 5 days):")
    print(df[['date', 'temp_min', 'temp_max', 'season']].head().to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()
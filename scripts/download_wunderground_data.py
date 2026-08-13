#!/usr/bin/env python3
"""
Example script for downloading Wunderground historical data
"""

import sys
import os
from datetime import date
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_acquisition.wunderground_scraper import WundergroundScraper


def main():
    parser = argparse.ArgumentParser(description='Download Wunderground historical temperature data')
    parser.add_argument('--station', type=str, required=True, 
                       choices=['ZSPD', 'KDEN'], help='Station ID (ZSPD for Shanghai, KDEN for Denver)')
    parser.add_argument('--start-year', type=int, default=2000, 
                       help='Start year (default: 2000)')
    parser.add_argument('--end-year', type=int, default=2019, 
                       help='End year (default: 2019)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: data/raw/wunderground/{station}_2000_2019.csv)')
    parser.add_argument('--config', type=str, default='configs/stations.yaml',
                       help='Path to station configuration file')
    parser.add_argument('--cache-dir', type=str, default='data/raw/wunderground',
                       help='Directory for caching downloaded data')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        os.makedirs('data/raw/wunderground', exist_ok=True)
        args.output = f'data/raw/wunderground/{args.station}_{args.start_year}_{args.end_year}.csv'
    
    print(f"=== Wunderground Historical Data Download ===")
    print(f"Station: {args.station}")
    print(f"Date range: {args.start_year}-01-01 to {args.end_year}-12-31")
    print(f"Output file: {args.output}")
    print(f"Cache directory: {args.cache_dir}")
    print("=" * 50)
    
    # Create scraper
    print("Initializing scraper...")
    scraper = WundergroundScraper(config_path=args.config, cache_dir=args.cache_dir)
    
    try:
        # Fetch data
        start_date = date(args.start_year, 1, 1)
        end_date = date(args.end_year, 12, 31)
        
        print(f"\nFetching data for {args.station} from {start_date} to {end_date}...")
        print("This may take several minutes depending on the date range.")
        print("Data will be cached to avoid redundant downloads.")
        
        daily_temps = scraper.fetch_station_range(args.station, start_date, end_date)
        
        if not daily_temps:
            print("\n❌ No data was fetched. Check your internet connection and station ID.")
            return 1
        
        # Export to DataFrame
        print(f"\n✓ Successfully fetched {len(daily_temps)} daily temperature records")
        
        df = scraper.export_to_dataframe(daily_temps)
        
        # Save to CSV
        df.to_csv(args.output, index=False)
        print(f"✓ Data saved to {args.output}")
        
        # Print summary statistics
        print("\n=== Data Summary ===")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Calculate completeness
        max_temp_completeness = (1 - df['temp_max_c'].isna().mean()) * 100
        min_temp_completeness = (1 - df['temp_min_c'].isna().mean()) * 100
        
        print(f"Max temperature completeness: {max_temp_completeness:.1f}%")
        print(f"Min temperature completeness: {min_temp_completeness:.1f}%")
        
        if max_temp_completeness > 0:
            print(f"Max temperature range: {df['temp_max_c'].min():.1f}°C to {df['temp_max_c'].max():.1f}°C")
            print(f"Average max temperature: {df['temp_max_c'].mean():.1f}°C")
        
        if min_temp_completeness > 0:
            print(f"Min temperature range: {df['temp_min_c'].min():.1f}°C to {df['temp_min_c'].max():.1f}°C")
            print(f"Average min temperature: {df['temp_min_c'].mean():.1f}°C")
        
        print(f"Average quality score: {df['quality_score'].mean():.3f}")
        
        # Check for any issues
        validation = scraper.validate_daily_data(daily_temps)
        if validation['issues']:
            print(f"\n⚠️  Validation issues found:")
            for issue in validation['issues'][:5]:  # Show first 5 issues
                print(f"  - {issue}")
            if len(validation['issues']) > 5:
                print(f"  ... and {len(validation['issues']) - 5} more issues")
        
        # Save a sample of the data
        sample_path = args.output.replace('.csv', '_sample.csv')
        df.head(10).to_csv(sample_path, index=False)
        print(f"\n✓ Sample data saved to {sample_path}")
        
        # Print sample
        print("\n=== First 5 Records ===")
        print(df[['date', 'temp_max_c', 'temp_min_c', 'quality_score']].head().to_string(index=False))
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        scraper.close()
        print("\n✓ Scraper closed successfully.")


if __name__ == "__main__":
    sys.exit(main())
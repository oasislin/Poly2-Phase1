#!/usr/bin/env python3
"""
Batch download script for Wunderground historical data (2000-2019)
Supports resume capability using SQLite progress tracking.
"""

import argparse
import sys
import os
from datetime import datetime, date, timedelta
import time
from typing import List, Tuple, Optional
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_acquisition.wunderground_scraper import WundergroundScraper


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('wunderground_download.log')
        ]
    )


def get_date_range_for_station(scraper: WundergroundScraper, station_id: str, 
                               start_year: int, end_year: int) -> List[Tuple[int, int]]:
    """
    Get list of (year, month) pairs to download, considering existing data
    
    Args:
        scraper: WundergroundScraper instance
        station_id: Station identifier
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        
    Returns:
        List of (year, month) pairs that need to be downloaded
    """
    months_to_download = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):  # 1 to 12
            # Check if already downloaded
            progress = scraper.get_download_progress(station_id, year, month)
            
            if not progress:
                # No progress record, need to download
                months_to_download.append((year, month))
                logging.info(f"  {year}-{month:02d}: Not started")
            elif progress['status'] == 'completed':
                # Already completed
                logging.debug(f"  {year}-{month:02d}: Already completed ({progress['records_downloaded']} records)")
            elif progress['status'] == 'failed':
                # Failed previously, retry
                months_to_download.append((year, month))
                logging.info(f"  {year}-{month:02d}: Previously failed - {progress['error_message']}")
            elif progress['status'] == 'in_progress':
                # Was in progress, resume
                months_to_download.append((year, month))
                logging.info(f"  {year}-{month:02d}: Resume from {progress['records_downloaded']} records")
            else:  # pending
                # Pending, need to download
                months_to_download.append((year, month))
                logging.info(f"  {year}-{month:02d}: Pending")
    
    return months_to_download


def download_month(scraper: WundergroundScraper, station_id: str, 
                   year: int, month: int, max_retries: int = 3) -> Tuple[bool, int, Optional[str]]:
    """
    Download data for a specific month
    
    Args:
        scraper: WundergroundScraper instance
        station_id: Station identifier
        year: Year to download
        month: Month to download (1-12)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (success, records_downloaded, error_message)
    """
    logging.info(f"Downloading {station_id} - {year}-{month:02d}")
    
    # Mark as in progress
    scraper.save_download_progress(station_id, year, month, 'in_progress', 0)
    
    for attempt in range(max_retries):
        try:
            # Fetch complete observation data for the month
            observations = scraper.fetch_station_month_full(station_id, year, month)
            
            if observations:
                # Save to database
                saved_count = scraper.save_observations(observations)
                
                # Mark as completed
                scraper.save_download_progress(
                    station_id, year, month, 'completed', 
                    records_downloaded=len(observations)
                )
                
                logging.info(f"  ✓ Downloaded {len(observations)} records with all 12 fields, saved {saved_count} observations")
                return True, len(observations), None
                
            else:
                error_msg = f"No data found for {year}-{month:02d}"
                logging.warning(f"  ⚠ {error_msg}")
                scraper.save_download_progress(
                    station_id, year, month, 'failed', 
                    records_downloaded=0, 
                    error_message=error_msg
                )
                return False, 0, error_msg
                
        except Exception as e:
            error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            logging.error(f"  ✗ {error_msg}")
            
            if attempt < max_retries - 1:
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt  # 1, 2, 4, 8... seconds
                logging.info(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                scraper.save_download_progress(
                    station_id, year, month, 'failed', 
                    records_downloaded=0, 
                    error_message=str(e)
                )
                return False, 0, str(e)
    
    return False, 0, "Max retries exceeded"


def download_station_data(scraper: WundergroundScraper, station_id: str,
                         start_year: int, end_year: int, 
                         force_redownload: bool = False) -> dict:
    """
    Download data for a station for a range of years
    
    Args:
        scraper: WundergroundScraper instance
        station_id: Station identifier
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        force_redownload: If True, re-download even if already completed
        
    Returns:
        Dictionary with download statistics
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"Downloading data for station: {station_id}")
    logging.info(f"Period: {start_year} to {end_year}")
    logging.info(f"{'='*60}")
    
    # Check if station exists
    if station_id not in scraper.stations:
        logging.error(f"Station {station_id} not found in configuration")
        return {
            'station_id': station_id,
            'success': False,
            'error': f"Station {station_id} not found"
        }
    
    # Get months to download
    if force_redownload:
        # Force re-download all months
        months_to_download = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                months_to_download.append((year, month))
        logging.info(f"Force re-download: {len(months_to_download)} months")
    else:
        # Check progress and resume
        months_to_download = get_date_range_for_station(scraper, station_id, start_year, end_year)
        logging.info(f"Resume download: {len(months_to_download)} months to download")
    
    if not months_to_download:
        logging.info(f"All data already downloaded for {station_id}")
        return {
            'station_id': station_id,
            'success': True,
            'total_months': 0,
            'completed_months': 0,
            'failed_months': 0,
            'total_records': 0,
            'message': 'All data already downloaded'
        }
    
    # Download each month
    total_months = len(months_to_download)
    completed_months = 0
    failed_months = 0
    total_records = 0
    
    logging.info(f"Starting download of {total_months} months...")
    
    for i, (year, month) in enumerate(months_to_download, 1):
        logging.info(f"\n[{i}/{total_months}] Processing {year}-{month:02d}")
        
        success, records, error_msg = download_month(scraper, station_id, year, month)
        
        if success:
            completed_months += 1
            total_records += records
            logging.info(f"  ✓ Completed ({records} records)")
        else:
            failed_months += 1
            logging.error(f"  ✗ Failed: {error_msg}")
        
        # Progress update
        progress_pct = (i / total_months) * 100
        logging.info(f"  Progress: {progress_pct:.1f}% ({completed_months} completed, {failed_months} failed)")
        
        # Rate limiting between months (already handled in scraper, but extra safety)
        if i < total_months:
            time.sleep(1)  # Small delay between months
    
    # Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"Download complete for {station_id}")
    logging.info(f"  Total months: {total_months}")
    logging.info(f"  Completed: {completed_months}")
    logging.info(f"  Failed: {failed_months}")
    logging.info(f"  Total records: {total_records}")
    logging.info(f"{'='*60}")
    
    return {
        'station_id': station_id,
        'success': failed_months == 0,
        'total_months': total_months,
        'completed_months': completed_months,
        'failed_months': failed_months,
        'total_records': total_records,
        'message': 'Download completed' if failed_months == 0 else f'{failed_months} months failed'
    }


def export_to_csv(scraper: WundergroundScraper, station_id: str, 
                  start_year: int, end_year: int, output_dir: str = 'data') -> str:
    """
    Export downloaded data to CSV
    
    Args:
        scraper: WundergroundScraper instance
        station_id: Station identifier
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
        output_dir: Output directory
        
    Returns:
        Path to the CSV file
    """
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate date range
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 31)
    
    # Load observations from database
    observations = scraper.load_observations(station_id, start_date, end_date)
    
    if not observations:
        logging.warning(f"No observations found for {station_id} in {start_year}-{end_year}")
        return ""
    
    # Export to DataFrame
    df = scraper.export_observations_to_dataframe(observations)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f'wunderground_{station_id}_{start_year}_{end_year}.csv')
    df.to_csv(csv_path, index=False)
    
    logging.info(f"Exported {len(observations)} records to {csv_path}")
    
    # Also save summary statistics
    summary_path = os.path.join(output_dir, f'wunderground_{station_id}_{start_year}_{end_year}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Wunderground Data Summary - {station_id}\n")
        f.write(f"Period: {start_year} to {end_year}\n")
        f.write(f"Total records: {len(observations)}\n")
        f.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n")
        f.write(f"\nField completeness:\n")
        for column in df.columns:
            if column != 'date':
                completeness = 1 - df[column].isna().mean()
                f.write(f"  {column}: {completeness:.1%}\n")
        
        f.write(f"\nQuality distribution:\n")
        if 'quality_level' in df.columns:
            quality_counts = df['quality_level'].value_counts()
            for level, count in quality_counts.items():
                f.write(f"  {level}: {count} ({count/len(df):.1%})\n")
    
    logging.info(f"Summary saved to {summary_path}")
    
    return csv_path


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Batch download Wunderground historical data (2000-2019)'
    )
    parser.add_argument('--station', type=str, default='all',
                       choices=['ZSPD', 'KDEN', 'all'],
                       help='Station ID (ZSPD, KDEN, or all)')
    parser.add_argument('--start-year', type=int, default=2000,
                       help='Start year (default: 2000)')
    parser.add_argument('--end-year', type=int, default=2019,
                       help='End year (default: 2019)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if already completed')
    parser.add_argument('--export', action='store_true',
                       help='Export to CSV after download')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for CSV files (default: data)')
    parser.add_argument('--db-path', type=str, default='data/wunderground.db',
                       help='Path to SQLite database (default: data/wunderground.db)')
    parser.add_argument('--cache-dir', type=str, default='data/raw/wunderground',
                       help='Cache directory for HTML pages (default: data/raw/wunderground)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize scraper
    scraper = WundergroundScraper(
        config_path="configs/stations.yaml",
        cache_dir=args.cache_dir,
        db_path=args.db_path
    )
    
    try:
        # Determine which stations to process
        if args.station == 'all':
            stations = ['ZSPD', 'KDEN']
        else:
            stations = [args.station]
        
        results = []
        
        for station_id in stations:
            # Download data
            result = download_station_data(
                scraper, station_id, args.start_year, args.end_year, args.force
            )
            results.append(result)
            
            # Export to CSV if requested
            if args.export and result.get('success', False):
                csv_path = export_to_csv(
                    scraper, station_id, args.start_year, args.end_year, args.output_dir
                )
                if csv_path:
                    result['csv_path'] = csv_path
        
        # Print summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        
        for result in results:
            station = result['station_id']
            success = result['success']
            total = result['total_months']
            completed = result['completed_months']
            failed = result['failed_months']
            records = result['total_records']
            
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"\n{station}: {status}")
            print(f"  Months: {completed}/{total} completed, {failed} failed")
            print(f"  Records: {records}")
            
            if 'csv_path' in result:
                print(f"  CSV: {result['csv_path']}")
            
            if 'message' in result:
                print(f"  Note: {result['message']}")
        
        print("\n" + "="*60)
        
        # Check if any failed
        if any(not r['success'] for r in results):
            print("\n⚠ Some downloads failed. You can:")
            print("  1. Run again to retry failed months")
            print("  2. Check wunderground_download.log for details")
            print("  3. Use --force to re-download everything")
            return 1
        else:
            print("\n✅ All downloads completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print("Progress has been saved. Run again to resume.")
        return 130
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        return 1
    finally:
        scraper.close()


if __name__ == "__main__":
    sys.exit(main())
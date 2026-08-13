#!/usr/bin/env python3
"""
Enhanced batch download script for Wunderground historical data (2000-2019)
Supports resume capability and handles 403 errors with intelligent retry logic.
"""

import argparse
import sys
import os
from datetime import datetime, date, timedelta
import time
import signal
import json
from typing import List, Tuple, Optional, Dict, Any
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_acquisition.wunderground_scraper import WundergroundScraper


def setup_logging(verbose: bool = False, log_file: str = 'wunderground_download_enhanced.log') -> logging.Logger:
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger('wunderground_batch')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


class BatchController:
    """Controller for batch download with 403 handling"""
    
    def __init__(self, scraper: WundergroundScraper, logger: logging.Logger):
        self.scraper = scraper
        self.logger = logger
        self.should_stop = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # State file for resuming
        self.state_file = 'download_state.json'
        self.state = self._load_state()
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        self.logger.info(f"Received signal {signum}, stopping gracefully...")
        self.should_stop = True
    
    def _load_state(self) -> Dict[str, Any]:
        """Load download state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load state file: {e}")
        
        return {
            'current_station': None,
            'current_year': None,
            'current_month': None,
            'completed_months': [],
            'failed_months': [],
            'consecutive_403s': 0,
            'last_403_time': 0,
            'start_time': None,
            'total_records': 0
        }
    
    def _save_state(self):
        """Save download state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save state file: {e}")
    
    def _update_state(self, station_id: str, year: int, month: int, 
                     status: str, records: int = 0, error: str = None):
        """Update download state"""
        month_key = f"{station_id}_{year}_{month:02d}"
        
        if status == 'completed':
            if month_key not in self.state['completed_months']:
                self.state['completed_months'].append(month_key)
            self.state['total_records'] += records
        elif status == 'failed':
            if month_key not in self.state['failed_months']:
                self.state['failed_months'].append(month_key)
                if error:
                    self.state.setdefault('failed_reasons', {})[month_key] = error
        
        self.state['current_station'] = station_id
        self.state['current_year'] = year
        self.state['current_month'] = month
        
        self._save_state()
    
    def _check_403_threshold(self) -> bool:
        """Check if we should stop due to too many 403s"""
        # Get current 403 count from scraper
        consecutive_403 = getattr(self.scraper, 'consecutive_403_count', 0)
        max_consecutive_403 = getattr(self.scraper, 'max_consecutive_403', 10)
        
        if consecutive_403 >= max_consecutive_403:
            self.logger.error(f"⚠ Too many consecutive 403s ({consecutive_403}/{max_consecutive_403}).")
            self.logger.error("  Stopping batch download to avoid IP blocking.")
            self.logger.error(f"  Please wait at least {self.scraper.forbidden_retry_delay} seconds before resuming.")
            return True
        
        # Also check time-based threshold
        last_403_time = getattr(self.scraper, 'last_403_time', 0)
        if last_403_time > 0:
            time_since_last_403 = time.time() - last_403_time
            if time_since_last_403 < 300:  # 5 minutes
                self.logger.warning(f"  Recent 403 detected ({time_since_last_403:.0f}s ago)")
        
        return False
    
    def download_month(self, station_id: str, year: int, month: int, 
                      max_retries: int = 5) -> Tuple[bool, int, Optional[str]]:
        """
        Download data for a specific month with enhanced error handling
        
        Args:
            station_id: Station identifier
            year: Year to download
            month: Month to download (1-12)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (success, records_downloaded, error_message)
        """
        self.logger.info(f"Downloading {station_id} - {year}-{month:02d}")
        
        # Check if we should stop due to signals
        if self.should_stop:
            return False, 0, "Stopped by user"
        
        # Check 403 threshold
        if self._check_403_threshold():
            return False, 0, "Too many consecutive 403s"
        
        # Mark as in progress
        self.scraper.save_download_progress(station_id, year, month, 'in_progress', 0)
        
        for attempt in range(max_retries):
            try:
                # Check 403 threshold before each attempt
                if self._check_403_threshold():
                    return False, 0, "Too many consecutive 403s"
                
                # Fetch complete observation data for the month
                observations = self.scraper.fetch_station_month_full(station_id, year, month)
                
                if observations:
                    # Save to database
                    saved_count = self.scraper.save_observations(observations)
                    
                    # Mark as completed
                    self.scraper.save_download_progress(
                        station_id, year, month, 'completed', 
                        records_downloaded=len(observations)
                    )
                    
                    # Update state
                    self._update_state(station_id, year, month, 'completed', len(observations))
                    
                    self.logger.info(f"  ✓ Downloaded {len(observations)} records with all 12 fields, saved {saved_count} observations")
                    return True, len(observations), None
                    
                else:
                    error_msg = f"No data found for {year}-{month:02d}"
                    self.logger.warning(f"  ⚠ {error_msg}")
                    self.scraper.save_download_progress(
                        station_id, year, month, 'failed', 
                        records_downloaded=0, 
                        error_message=error_msg
                    )
                    self._update_state(station_id, year, month, 'failed', 0, error_msg)
                    return False, 0, error_msg
                    
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                self.logger.error(f"  ✗ {error_msg}")
                
                # Check if error is due to too many 403s
                if "Too many consecutive 403s" in str(e):
                    self.logger.error("  ⚠ Stopping batch due to excessive 403s")
                    self._update_state(station_id, year, month, 'failed', 0, str(e))
                    return False, 0, str(e)
                
                if attempt < max_retries - 1:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt  # 1, 2, 4, 8... seconds
                    self.logger.info(f"  Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
                    # Check 403 threshold again after waiting
                    if self._check_403_threshold():
                        return False, 0, "Too many consecutive 403s"
                else:
                    # Final attempt failed
                    self.scraper.save_download_progress(
                        station_id, year, month, 'failed', 
                        records_downloaded=0, 
                        error_message=str(e)
                    )
                    self._update_state(station_id, year, month, 'failed', 0, str(e))
                    return False, 0, str(e)
        
        return False, 0, "Max retries exceeded"
    
    def get_date_range_for_station(self, station_id: str, 
                                   start_year: int, end_year: int,
                                   force_redownload: bool = False) -> List[Tuple[int, int]]:
        """
        Get list of (year, month) pairs to download, considering existing data
        
        Args:
            station_id: Station identifier
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            force_redownload: If True, re-download even if already completed
            
        Returns:
            List of (year, month) pairs that need to be downloaded
        """
        months_to_download = []
        
        # Load state to resume from where we left off
        completed_months = set(self.state.get('completed_months', []))
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):  # 1 to 12
                month_key = f"{station_id}_{year}_{month:02d}"
                
                # Skip if already completed in this session
                if month_key in completed_months and not force_redownload:
                    self.logger.debug(f"  {year}-{month:02d}: Already completed in this session")
                    continue
                
                # Check if already downloaded in database
                progress = self.scraper.get_download_progress(station_id, year, month)
                
                if not progress:
                    # No progress record, need to download
                    months_to_download.append((year, month))
                    self.logger.info(f"  {year}-{month:02d}: Not started")
                elif progress['status'] == 'completed' and not force_redownload:
                    # Already completed and not forcing re-download
                    self.logger.debug(f"  {year}-{month:02d}: Already completed ({progress['records_downloaded']} records)")
                elif progress['status'] == 'failed':
                    # Failed previously, retry
                    months_to_download.append((year, month))
                    self.logger.info(f"  {year}-{month:02d}: Previously failed - {progress['error_message']}")
                elif progress['status'] == 'in_progress':
                    # Was in progress, resume
                    months_to_download.append((year, month))
                    self.logger.info(f"  {year}-{month:02d}: Resume from {progress['records_downloaded']} records")
                else:  # pending or force_redownload
                    # Pending or force re-download
                    months_to_download.append((year, month))
                    self.logger.info(f"  {year}-{month:02d}: Pending")
        
        return months_to_download
    
    def download_station_data(self, station_id: str,
                             start_year: int, end_year: int, 
                             force_redownload: bool = False) -> dict:
        """
        Download data for a station for a range of years with enhanced error handling
        
        Args:
            station_id: Station identifier
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            force_redownload: If True, re-download even if already completed
            
        Returns:
            Dictionary with download statistics
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Downloading data for station: {station_id}")
        self.logger.info(f"Period: {start_year} to {end_year}")
        self.logger.info(f"{'='*60}")
        
        # Check if station exists
        if station_id not in self.scraper.stations:
            self.logger.error(f"Station {station_id} not found in configuration")
            return {
                'station_id': station_id,
                'success': False,
                'error': f"Station {station_id} not found"
            }
        
        # Initialize state
        if not self.state.get('start_time'):
            self.state['start_time'] = time.time()
            self._save_state()
        
        # Reset 403 counter at start of batch
        if hasattr(self.scraper, 'reset_403_counter'):
            self.scraper.reset_403_counter()
        
        # Get months to download
        months_to_download = self.get_date_range_for_station(station_id, start_year, end_year, force_redownload)
        
        if not months_to_download:
            self.logger.info(f"All data already downloaded for {station_id}")
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
        
        self.logger.info(f"Starting download of {total_months} months...")
        
        for i, (year, month) in enumerate(months_to_download, 1):
            # Check if we should stop
            if self.should_stop:
                self.logger.warning(f"Batch stopped by user at {i}/{total_months}")
                break
            
            # Check 403 threshold before each month
            if self._check_403_threshold():
                self.logger.error(f"Stopping batch due to excessive 403s at {i}/{total_months}")
                break
            
            self.logger.info(f"\n[{i}/{total_months}] Processing {year}-{month:02d}")
            
            success, records, error_msg = self.download_month(station_id, year, month)
            
            if success:
                completed_months += 1
                total_records += records
                self.logger.info(f"  ✓ Completed ({records} records)")
            else:
                failed_months += 1
                self.logger.error(f"  ✗ Failed: {error_msg}")
                
                # If failed due to 403 threshold, stop the batch
                if "Too many consecutive 403s" in error_msg:
                    self.logger.error(f"  ⚠ Stopping batch due to 403 threshold")
                    break
            
            # Progress update
            progress_pct = (i / total_months) * 100
            self.logger.info(f"  Progress: {progress_pct:.1f}% ({completed_months} completed, {failed_months} failed)")
            
            # Small delay between months (rate limiting is handled in scraper)
            if i < total_months and not self.should_stop:
                time.sleep(1)
        
        # Summary
        elapsed_time = time.time() - self.state.get('start_time', time.time())
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Download complete for {station_id}")
        self.logger.info(f"  Total months: {total_months}")
        self.logger.info(f"  Completed: {completed_months}")
        self.logger.info(f"  Failed: {failed_months}")
        self.logger.info(f"  Total records: {total_records}")
        self.logger.info(f"  Elapsed time: {elapsed_time:.1f} seconds")
        
        # Report 403 status
        consecutive_403 = getattr(self.scraper, 'consecutive_403_count', 0)
        if consecutive_403 > 0:
            self.logger.info(f"  Consecutive 403s: {consecutive_403}")
        
        self.logger.info(f"{'='*60}")
        
        # Clean up state
        if completed_months + failed_months == total_months:
            # All months processed, clear state
            os.remove(self.state_file) if os.path.exists(self.state_file) else None
        
        return {
            'station_id': station_id,
            'success': failed_months == 0 and not self.should_stop,
            'total_months': total_months,
            'completed_months': completed_months,
            'failed_months': failed_months,
            'total_records': total_records,
            'elapsed_time': elapsed_time,
            'consecutive_403s': consecutive_403,
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
        description='Enhanced batch download Wunderground historical data (2000-2019) with 403 handling'
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
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous state file')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Initialize scraper
    scraper = WundergroundScraper(
        config_path="configs/stations.yaml",
        cache_dir=args.cache_dir,
        db_path=args.db_path
    )
    
    # Initialize batch controller
    controller = BatchController(scraper, logger)
    
    try:
        # Determine which stations to process
        if args.station == 'all':
            stations = ['ZSPD', 'KDEN']
        else:
            stations = [args.station]
        
        results = []
        
        for station_id in stations:
            # Download data
            result = controller.download_station_data(
                station_id, args.start_year, args.end_year, args.force
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
            elapsed = result.get('elapsed_time', 0)
            consecutive_403s = result.get('consecutive_403s', 0)
            
            status = "✅ SUCCESS" if success else "⚠ PARTIAL" if completed > 0 else "❌ FAILED"
            print(f"\n{station}: {status}")
            print(f"  Months: {completed}/{total} completed, {failed} failed")
            print(f"  Records: {records}")
            print(f"  Time: {elapsed:.1f} seconds")
            
            if consecutive_403s > 0:
                print(f"  Consecutive 403s: {consecutive_403s}")
            
            if 'csv_path' in result:
                print(f"  CSV: {result['csv_path']}")
            
            if 'message' in result:
                print(f"  Note: {result['message']}")
        
        print("\n" + "="*60)
        
        # Check if any failed due to 403s
        if any(r.get('consecutive_403s', 0) >= 10 for r in results):
            print("\n⚠ Batch stopped due to too many 403s.")
            print("  Recommendations:")
            print("  1. Wait at least 30 minutes before resuming")
            print("  2. Consider using a VPN or proxy")
            print("  3. Use --resume flag to continue where you left off")
            print("\n  To resume:")
            print(f"    python {__file__} --station {' '.join(stations)} --start-year {args.start_year} --end-year {args.end_year} --resume")
        
        # Check if any failed
        if any(not r['success'] for r in results):
            print("\n⚠ Some downloads failed or were stopped.")
            print("  You can:")
            print("  1. Run again to retry failed months")
            print("  2. Use --resume flag to continue from state file")
            print("  3. Check wunderground_download_enhanced.log for details")
            print("  4. Use --force to re-download everything")
            return 1
        else:
            print("\n✅ All downloads completed successfully!")
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print("Progress has been saved. Use --resume flag to continue.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        return 1
    finally:
        scraper.close()


if __name__ == "__main__":
    sys.exit(main())
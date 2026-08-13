#!/usr/bin/env python3
"""
Enhanced Wunderground scraper with better anti-bot evasion techniques.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
import os
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import sqlite3
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import cloudscraper  # Alternative to requests for Cloudflare protection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wunderground_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StationConfig:
    """Station configuration"""
    id: str
    name: str
    country: str
    city: str
    start_year: int
    end_year: int
    timezone: str
    elevation: float = 0.0
    polymarket_id: str = ""
    temperature_unit: str = "C"


class WundergroundScraperEnhanced:
    """Enhanced Wunderground scraper with anti-bot evasion techniques"""
    
    def __init__(self, config_path: Optional[str] = None, use_cloudscraper: bool = False):
        """
        Initialize enhanced scraper
        
        Args:
            config_path: Path to YAML configuration file
            use_cloudscraper: Use cloudscraper for Cloudflare protection
        """
        self.config = self._load_config(config_path)
        self.use_cloudscraper = use_cloudscraper
        
        # Request rotation lists
        self.user_agents = self._load_user_agents()
        self.referers = [
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
            "https://www.wunderground.com/",
            ""
        ]
        
        # Initialize session with enhanced headers
        if use_cloudscraper:
            self.session = cloudscraper.create_scraper()
            logger.info("Using cloudscraper for Cloudflare protection")
        else:
            self.session = requests.Session()
        
        self._setup_session_headers()
        
        # Rate limiting and request tracking
        self.request_count = 0
        self.last_request_time = time.time()
        self.min_request_interval = 3.0  # Minimum 3 seconds between requests
        self.random_delay_range = (1.0, 5.0)  # Random delay between requests
        
        # Cache setup
        self.cache_dir = Path("./cache/wunderground")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized enhanced scraper with {'cloudscraper' if use_cloudscraper else 'requests'}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        default_config = {
            'stations': [
                {
                    'id': 'ZSPD',
                    'name': 'Shanghai Pudong International Airport',
                    'country': 'cn',
                    'city': 'shanghai',
                    'start_year': 2000,
                    'end_year': 2019,
                    'timezone': 'Asia/Shanghai',
                    'elevation': 4.0,
                    'polymarket_id': 'shanghai',
                    'temperature_unit': 'C'
                },
                {
                    'id': 'KDEN',
                    'name': 'Denver International Airport',
                    'country': 'us',
                    'city': 'denver',
                    'start_year': 2000,
                    'end_year': 2019,
                    'timezone': 'America/Denver',
                    'elevation': 1655.0,
                    'polymarket_id': 'denver',
                    'temperature_unit': 'F'
                }
            ],
            'request_settings': {
                'max_retries': 5,
                'timeout': 60,
                'verify_ssl': True,
                'proxies': None,  # Can add proxy support here
            },
            'cache_settings': {
                'enabled': True,
                'max_age_days': 7,
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with defaults
                import copy
                merged_config = copy.deepcopy(default_config)
                merged_config.update(user_config)
                return merged_config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        
        return default_config
    
    def _load_user_agents(self) -> List[str]:
        """Load a list of user agents to rotate"""
        return [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Chrome on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            # Firefox on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/120.0',
            # Firefox on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/120.0',
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        ]
    
    def _setup_session_headers(self):
        """Setup session headers to mimic real browser"""
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Microsoft Edge";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
        }
        
        self.session.headers.update(headers)
    
    def _rotate_headers(self):
        """Rotate headers to avoid detection"""
        # Rotate User-Agent
        self.session.headers['User-Agent'] = random.choice(self.user_agents)
        
        # Rotate Referer
        self.session.headers['Referer'] = random.choice(self.referers)
        
        # Add some random headers occasionally
        if random.random() < 0.3:
            self.session.headers['Accept-Encoding'] = random.choice(['gzip, deflate, br', 'gzip, deflate'])
            self.session.headers['Accept-Language'] = random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.8'])
    
    def _rate_limit(self):
        """Enhanced rate limiting with random delays"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Ensure minimum interval
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        # Add random delay
        random_delay = random.uniform(*self.random_delay_range)
        time.sleep(random_delay)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Rotate headers every 5 requests
        if self.request_count % 5 == 0:
            self._rotate_headers()
        
        # Longer pause every 20 requests
        if self.request_count % 20 == 0:
            logger.info(f"Made {self.request_count} requests, taking a longer break...")
            time.sleep(random.uniform(10.0, 30.0))
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        # Remove query parameters that might change
        parsed = urlparse(url)
        clean_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # params
            '',  # query
            ''   # fragment
        ))
        
        # Create hash of clean URL
        return hashlib.md5(clean_url.encode()).hexdigest()
    
    def _get_cached_response(self, url: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.html"
        
        if cache_file.exists():
            # Check if cache is expired
            cache_age = time.time() - cache_file.stat().st_mtime
            max_age = self.config['cache_settings']['max_age_days'] * 24 * 3600
            
            if cache_age < max_age:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    logger.debug(f"Using cached response for {url}")
                    return content
                except Exception as e:
                    logger.warning(f"Failed to read cache for {url}: {e}")
        
        return None
    
    def _cache_response(self, url: str, content: str):
        """Cache response content"""
        if not self.config['cache_settings']['enabled']:
            return
        
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.html"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"Cached response for {url}")
        except Exception as e:
            logger.warning(f"Failed to cache response for {url}: {e}")
    
    def fetch_page(self, url: str, max_retries: int = 5) -> Optional[str]:
        """
        Fetch page with enhanced anti-bot evasion
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content or None if all attempts fail
        """
        # Check cache first
        cached = self._get_cached_response(url)
        if cached:
            return cached
        
        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()
                
                # Rotate headers for each attempt
                self._rotate_headers()
                
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
                
                # Make request
                response = self.session.get(
                    url,
                    timeout=self.config['request_settings']['timeout'],
                    verify=self.config['request_settings']['verify_ssl'],
                    proxies=self.config['request_settings']['proxies']
                )
                
                # Check response
                if response.status_code == 403:
                    logger.warning(f"403 Forbidden for {url}. Site may be blocking requests.")
                    
                    if attempt < max_retries - 1:
                        # Try different evasion techniques
                        if attempt == 1:
                            logger.info("Switching to cloudscraper for next attempt...")
                            self.use_cloudscraper = True
                            self.session = cloudscraper.create_scraper()
                            self._setup_session_headers()
                        elif attempt == 2:
                            logger.info("Adding longer delay and different headers...")
                            self.min_request_interval = 10.0
                            self.random_delay_range = (5.0, 15.0)
                        
                        wait_time = 2 ** attempt + random.uniform(1, 5)
                        logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"All attempts failed with 403 Forbidden for {url}")
                        return None
                
                response.raise_for_status()
                
                # Check if we got a valid HTML response
                if not response.text or len(response.text) < 100:
                    logger.warning(f"Received empty or very short response for {url}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                
                # Cache successful response
                self._cache_response(url, response.text)
                return response.text
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {url}")
                    return None
        
        return None
    
    def build_wunderground_url(self, station_id: str, year: int, month: int) -> str:
        """Build Wunderground URL for station, year, month"""
        # Get station config
        station = None
        for s in self.config['stations']:
            if s['id'] == station_id:
                station = s
                break
        
        if not station:
            raise ValueError(f"Station {station_id} not found in configuration")
        
        # Build URL
        base_url = "https://www.wunderground.com/history/monthly"
        url = f"{base_url}/{station['country']}/{station['city']}/{station_id}/date/{year}-{month}"
        
        return url
    
    def parse_monthly_page(self, html: str, station_id: str, year: int, month: int) -> List[Dict]:
        """
        Parse monthly Wunderground page to extract daily observations
        
        Args:
            html: HTML content of the monthly page
            station_id: Station ID
            year: Year
            month: Month
            
        Returns:
            List of daily observation dictionaries
        """
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        observations = []
        
        try:
            # First try table parsing (current Wunderground structure)
            observations = self._extract_from_table(soup, station_id, year, month)
            
            # If table parsing fails, try JSON extraction as fallback
            if not observations:
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string and 'window.__NUXT__' in script.string:
                        # Extract JSON data
                        json_match = re.search(r'window\.__NUXT__\s*=\s*({.*?});', script.string, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            data = json.loads(json_str)
                            observations = self._extract_from_json(data, station_id, year, month)
                            break
                
        except Exception as e:
            logger.error(f"Error parsing page for {station_id} {year}-{month}: {e}")
        
        return observations
    
    def _extract_from_json(self, data: Dict, station_id: str, year: int, month: int) -> List[Dict]:
        """Extract observations from JSON data"""
        observations = []
        
        try:
            # Navigate through the JSON structure to find observations
            # This structure may change, so we need to be flexible
            observations_data = data.get('state', {}).get('data', {}).get('history', {}).get('days', [])
            
            for day_data in observations_data:
                try:
                    obs = {
                        'station_id': station_id,
                        'date': f"{year}-{month:02d}-{int(day_data.get('date', {}).get('day', 0)):02d}",
                        'temp_max': float(day_data.get('temperature', {}).get('max', {}).get('value', 0)),
                        'temp_min': float(day_data.get('temperature', {}).get('min', {}).get('value', 0)),
                        'dew_point_max': float(day_data.get('dewPoint', {}).get('max', {}).get('value', 0)),
                        'dew_point_min': float(day_data.get('dewPoint', {}).get('min', {}).get('value', 0)),
                        'humidity_avg': float(day_data.get('humidity', {}).get('avg', 0)),
                        'wind_speed_max': float(day_data.get('windSpeed', {}).get('max', {}).get('value', 0)),
                        'pressure_avg': float(day_data.get('pressure', {}).get('avg', {}).get('value', 0)),
                        'precipitation': float(day_data.get('precipitation', {}).get('total', {}).get('value', 0)),
                    }
                    observations.append(obs)
                except (ValueError, AttributeError, KeyError) as e:
                    logger.debug(f"Error parsing day data: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"JSON extraction failed: {e}")
        
        return observations
    
    def _extract_from_table(self, soup: BeautifulSoup, station_id: str, year: int, month: int) -> List[Dict]:
        """Extract observations from HTML table (fallback method)"""
        observations = []
        
        # Look for observation tables
        tables = soup.find_all('table', class_=re.compile(r'observation|history'))
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header row
                cols = row.find_all('td')
                if len(cols) >= 3:  # Need at least date, max temp, min temp
                    try:
                        day = int(cols[0].text.strip())
                        temp_max_str = cols[1].text.strip().replace('°', '')
                        temp_min_str = cols[2].text.strip().replace('°', '')
                        
                        obs = {
                            'station_id': station_id,
                            'date': f"{year}-{month:02d}-{day:02d}",
                            'temp_max': float(temp_max_str),
                            'temp_min': float(temp_min_str),
                        }
                        observations.append(obs)
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Error parsing table row: {e}")
                        continue
        
        return observations
    
    def fetch_station_data(self, station_id: str, start_year: int, end_year: int) -> List[Dict]:
        """
        Fetch data for a station over a range of years
        
        Args:
            station_id: Station ID (e.g., 'ZSPD', 'KDEN')
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            List of daily observations
        """
        all_observations = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                logger.info(f"Fetching {station_id} {year}-{month:02d}")
                
                url = self.build_wunderground_url(station_id, year, month)
                html = self.fetch_page(url)
                
                if html:
                    observations = self.parse_monthly_page(html, station_id, year, month)
                    all_observations.extend(observations)
                    
                    logger.info(f"Found {len(observations)} days for {station_id} {year}-{month:02d}")
                    
                    # Save progress periodically
                    if len(all_observations) % 100 == 0:
                        self._save_progress(station_id, all_observations)
                else:
                    logger.warning(f"Failed to fetch {station_id} {year}-{month:02d}")
                
                # Be extra careful with rate limiting
                time.sleep(random.uniform(2.0, 4.0))
        
        # Final save
        self._save_progress(station_id, all_observations)
        return all_observations
    
    def _save_progress(self, station_id: str, observations: List[Dict]):
        """Save progress to file"""
        if not observations:
            return
        
        output_dir = Path("./data/wunderground")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{station_id}_progress.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(observations, f, indent=2, default=str)
            logger.info(f"Progress saved: {len(observations)} observations for {station_id}")
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def convert_to_dataframe(self, observations: List[Dict]) -> pd.DataFrame:
        """Convert observations list to pandas DataFrame"""
        if not observations:
            return pd.DataFrame()
        
        df = pd.DataFrame(observations)
        
        # Convert date strings to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
        
        return df


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Wunderground Scraper')
    parser.add_argument('--station', type=str, default='ZSPD', help='Station ID (ZSPD or KDEN)')
    parser.add_argument('--start-year', type=int, default=2023, help='Start year')
    parser.add_argument('--end-year', type=int, default=2023, help='End year')
    parser.add_argument('--use-cloudscraper', action='store_true', help='Use cloudscraper for Cloudflare protection')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = WundergroundScraperEnhanced(
        config_path=args.config,
        use_cloudscraper=args.use_cloudscraper
    )
    
    # Fetch data
    observations = scraper.fetch_station_data(
        station_id=args.station,
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    # Convert to DataFrame
    df = scraper.convert_to_dataframe(observations)
    
    # Save results
    if not df.empty:
        output_file = f"./data/wunderground/{args.station}_{args.start_year}_{args.end_year}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} observations to {output_file}")
        print(f"\nSample data:")
        print(df.head())
    else:
        print("No data fetched")


if __name__ == "__main__":
    main()
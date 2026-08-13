#!/usr/bin/env python3
"""
Enhanced Wunderground historical data scraper for Shanghai (ZSPD) and Denver (KDEN) stations.
Supports robust data collection for 2000-2019 period with error handling, rate limiting,
and data validation.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import csv
import os
import logging
import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import yaml
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json


class DataQuality(Enum):
    """Data quality levels for meteorological observations"""
    EXCELLENT = "excellent"  # All fields present and valid
    GOOD = "good"            # Minor issues, data usable
    WARNING = "warning"      # Some issues, use with caution
    ERROR = "error"          # Major issues, consider discarding


@dataclass
class DailyObservation:
    """Complete daily observation data structure with all 12 fields"""
    # Required fields
    date: date
    station_id: str
    
    # Temperature fields (2)
    temp_high: Optional[float] = None
    temp_low: Optional[float] = None
    
    # Dew point fields (2)
    dew_pt_high: Optional[float] = None
    dew_pt_low: Optional[float] = None
    
    # Humidity field (1)
    humidity: Optional[float] = None
    
    # Wind fields (3)
    wind: Optional[str] = None
    max_wind: Optional[float] = None
    wind_gust: Optional[float] = None
    
    # Pressure field (1)
    pressure: Optional[float] = None
    
    # Precipitation field (1)
    precipitation: Optional[float] = None
    
    # Condition field (1)
    condition: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 1.0
    quality_level: DataQuality = DataQuality.EXCELLENT
    validation_errors: List[str] = field(default_factory=list)
    
    # Metadata
    data_source: str = "wunderground"
    scraped_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'date': self.date.isoformat(),
            'station_id': self.station_id,
            'temp_high': self.temp_high,
            'temp_low': self.temp_low,
            'dew_pt_high': self.dew_pt_high,
            'dew_pt_low': self.dew_pt_low,
            'humidity': self.humidity,
            'wind': self.wind,
            'max_wind': self.max_wind,
            'wind_gust': self.wind_gust,
            'pressure': self.pressure,
            'precipitation': self.precipitation,
            'condition': self.condition,
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'validation_errors': self.validation_errors.copy(),
            'data_source': self.data_source,
            'scraped_at': self.scraped_at.isoformat() if self.scraped_at else None
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DailyObservation':
        """Create from dictionary"""
        import copy
        data = copy.deepcopy(data)
        
        # Convert string date to date object
        if 'date' in data and isinstance(data['date'], str):
            data['date'] = date.fromisoformat(data['date'])
        
        # Convert string quality_level to enum
        if 'quality_level' in data and isinstance(data['quality_level'], str):
            data['quality_level'] = DataQuality(data['quality_level'])
        
        # Convert string scraped_at to datetime
        if 'scraped_at' in data and data['scraped_at'] and isinstance(data['scraped_at'], str):
            data['scraped_at'] = datetime.fromisoformat(data['scraped_at'])
        
        return cls(**data)
    
    @classmethod
    def from_daily_temperature(cls, dt: 'DailyTemperature') -> 'DailyObservation':
        """Convert from existing DailyTemperature object"""
        return cls(
            date=dt.date,
            station_id=dt.station_id,
            temp_high=dt.temp_max,
            temp_low=dt.temp_min,
            quality_score=dt.quality_score,
            data_source=dt.data_source
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StationConfig:
    """Configuration for a weather station"""
    station_id: str
    wunderground_id: str
    country: str
    city: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    temperature_unit: str  # "C" or "F"
    polymarket_id: str


@dataclass
class DailyTemperature:
    """Daily temperature data structure (legacy, for backward compatibility)"""
    date: date
    temp_max: Optional[float]
    temp_min: Optional[float]
    quality_score: float  # 0.0 to 1.0
    station_id: str
    data_source: str = "wunderground"
    
    def to_daily_observation(self) -> DailyObservation:
        """Convert to DailyObservation"""
        return DailyObservation(
            date=self.date,
            station_id=self.station_id,
            temp_high=self.temp_max,
            temp_low=self.temp_min,
            quality_score=self.quality_score,
            data_source=self.data_source
        )


class WundergroundScraper:
    """Enhanced scraper for Wunderground historical data"""
    
    def __init__(self, config_path: str = "configs/stations.yaml", cache_dir: str = "data/raw/wunderground", db_path: str = "data/wunderground.db"):
        """
        Initialize the scraper with configuration and caching
        
        Args:
            config_path: Path to station configuration YAML file
            cache_dir: Directory to cache downloaded data
            db_path: Path to SQLite database for persistent storage
        """
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Database path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load station configurations
        self.stations = self._load_station_configs()
        
        # Initialize session with rate limiting
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 5.0  # Increased to 5 seconds for Task 1.1
        
        # Initialize databases
        self._init_cache_db()
        self._init_sqlite_db()
    
    def _load_station_configs(self) -> Dict[str, StationConfig]:
        """Load station configurations from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            stations = {}
            for station_id, station_data in config.get('stations', {}).items():
                stations[station_id] = StationConfig(
                    station_id=station_id,
                    wunderground_id=station_data['wunderground_id'],
                    country=station_data['country'],
                    city=station_data['city'],
                    latitude=station_data['latitude'],
                    longitude=station_data['longitude'],
                    elevation=station_data['elevation'],
                    timezone=station_data['timezone'],
                    temperature_unit=station_data['temperature_unit'],
                    polymarket_id=station_data['polymarket_id']
                )
            return stations
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            logger.warning(f"Config file {self.config_path} not found, using default stations")
            return {
                'ZSPD': StationConfig(
                    station_id='ZSPD',
                    wunderground_id='ZSPD',
                    country='cn',
                    city='shanghai',
                    latitude=31.15,
                    longitude=121.80,
                    elevation=4.0,
                    timezone='Asia/Shanghai',
                    temperature_unit='C',
                    polymarket_id='shanghai'
                ),
                'KDEN': StationConfig(
                    station_id='KDEN',
                    wunderground_id='KDEN',
                    country='us',
                    city='denver',
                    latitude=39.86,
                    longitude=-104.67,
                    elevation=1655.0,
                    timezone='America/Denver',
                    temperature_unit='F',
                    polymarket_id='denver'
                )
            }
    
    def _init_cache_db(self):
        """Initialize SQLite database for caching"""
        cache_db_path = self.cache_dir / "cache.db"
        self.conn = sqlite3.connect(cache_db_path)
        cursor = self.conn.cursor()
        
        # Create cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS page_cache (
                url_hash TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                html_content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status_code INTEGER,
                error_message TEXT
            )
        ''')
        
        # Create temperature data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS temperature_data (
                station_id TEXT,
                date DATE,
                temp_max REAL,
                temp_min REAL,
                quality_score REAL,
                source_url TEXT,
                scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (station_id, date)
            )
        ''')
        
        self.conn.commit()
    
    def _init_sqlite_db(self):
        """Initialize SQLite database for persistent storage of observations"""
        # Create main database connection
        self.db_conn = sqlite3.connect(self.db_path)
        cursor = self.db_conn.cursor()
        
        # Create observations table with all fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                station_id TEXT NOT NULL,
                temp_high REAL,
                temp_low REAL,
                dew_pt_high REAL,
                dew_pt_low REAL,
                humidity REAL,
                wind TEXT,
                max_wind REAL,
                wind_gust REAL,
                pressure REAL,
                precipitation REAL,
                condition TEXT,
                quality_score REAL,
                quality_level TEXT,
                validation_errors TEXT,  -- JSON array of strings
                data_source TEXT,
                scraped_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, station_id)
            )
        ''')
        
        # Create download progress tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                status TEXT NOT NULL,  -- 'pending', 'in_progress', 'completed', 'failed'
                records_downloaded INTEGER DEFAULT 0,
                error_message TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                UNIQUE(station_id, year, month)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_station_date ON observations(station_id, date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_observations_date ON observations(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_station_year_month ON download_progress(station_id, year, month)')
        
        self.db_conn.commit()
        
        logger.info(f"Initialized SQLite database at {self.db_path}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cached_page(self, url: str) -> Optional[str]:
        """Get page from cache if available and not expired"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT html_content, timestamp 
            FROM page_cache 
            WHERE url_hash = ? AND timestamp > datetime('now', '-7 days')
        ''', (url_hash,))
        
        result = cursor.fetchone()
        if result:
            html_content, timestamp = result
            logger.debug(f"Cache hit for URL: {url} (cached at {timestamp})")
            return html_content
        
        return None
    
    def _cache_page(self, url: str, html_content: Optional[str], status_code: int, error_message: Optional[str] = None):
        """Cache page content"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO page_cache 
            (url_hash, url, html_content, status_code, error_message) 
            VALUES (?, ?, ?, ?, ?)
        ''', (url_hash, url, html_content, status_code, error_message))
        
        self.conn.commit()
    
    def fetch_page_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch page with retry logic and rate limiting
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            HTML content or None if all attempts fail
        """
        # Check cache first
        cached_content = self._get_cached_page(url)
        if cached_content:
            return cached_content
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                logger.info(f"Fetching {url} (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Cache successful response
                self._cache_page(url, response.text, response.status_code)
                return response.text
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {url}")
                    self._cache_page(url, None, 0, str(e))
                    return None
        
        return None
    
    def build_wunderground_url(self, station: StationConfig, year: int, month: int) -> str:
        """
        Build Wunderground URL for a specific station, year, and month
        
        Args:
            station: Station configuration
            year: Year (e.g., 2020)
            month: Month (1-12)
            
        Returns:
            Wunderground URL
        """
        return f"https://www.wunderground.com/history/monthly/{station.country}/{station.city}/{station.wunderground_id}/date/{year}-{month}"
    
    def parse_number(self, num_str: str) -> Optional[float]:
        """
        Parse number string, handling various formats
        
        Args:
            num_str: Number string (e.g., "42", "42.5", "42°C", "75%", "25 km/h")
            
        Returns:
            Parsed float or None if cannot parse
        """
        if not num_str or num_str.strip() == '-' or num_str.strip() == '':
            return None
        
        # Remove common units and symbols
        cleaned = num_str.strip()
        cleaned = cleaned.replace('%', '').replace('hPa', '').replace('mm', '')
        cleaned = cleaned.replace('kph', '').replace('km/h', '').replace('mph', '')
        cleaned = cleaned.replace('°C', '').replace('°F', '').replace('°', '')
        cleaned = cleaned.replace(',', '')  # Remove thousand separators
        
        # Extract numeric part
        match = re.search(r'[-+]?\d*\.?\d+', cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        
        return None
    
    def parse_temperature(self, temp_str: str, unit: str) -> Optional[float]:
        """
        Parse temperature string and convert to Celsius if needed
        
        Args:
            temp_str: Temperature string (e.g., "32°C", "89°F", "32")
            unit: Temperature unit ("C" or "F")
            
        Returns:
            Temperature in Celsius, or None if cannot parse
        """
        if not temp_str or temp_str.strip() == '-':
            return None
        
        value = self.parse_number(temp_str)
        if value is None:
            return None
        
        # Convert Fahrenheit to Celsius if needed
        if unit == 'F':
            value = (value - 32) * 5/9
        
        return value
    
    def parse_temperature_string(self, temp_str: str, unit: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse temperature string to extract max and min temperatures
        
        Args:
            temp_str: Temperature string (e.g., "32°C / 25°C" or "89°F / 72°F")
            unit: Temperature unit ("C" or "F")
            
        Returns:
            Tuple of (max_temp, min_temp) in Celsius
        """
        if not temp_str or temp_str.strip() == '-':
            return None, None
        
        # Try to extract numbers with unit symbols
        patterns = [
            rf'(-?\d+)\s*°{unit}\s*/\s*(-?\d+)\s*°{unit}',  # With unit symbols
            r'(-?\d+)\s*[^\d]*/\s*(-?\d+)\s*[^\d]*',  # Generic pattern
            r'(-?\d+)\s*/\s*(-?\d+)',  # Simple numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, temp_str)
            if match:
                try:
                    max_temp = float(match.group(1))
                    min_temp = float(match.group(2))
                    
                    # Convert to Celsius if needed
                    if unit == 'F':
                        max_temp = (max_temp - 32) * 5/9
                        min_temp = (min_temp - 32) * 5/9
                    
                    return max_temp, min_temp
                except (ValueError, IndexError):
                    continue
        
        logger.debug(f"Could not parse temperature string: {temp_str}")
        return None, None
    
    def parse_wind_speed(self, wind_str: str) -> Optional[float]:
        """
        Parse wind speed string
        
        Args:
            wind_str: Wind speed string (e.g., "25 kph", "15 mph", "Gusts to 35")
            
        Returns:
            Wind speed in km/h, or None if cannot parse
        """
        if not wind_str:
            return None
        
        # Try to extract numeric value
        value = self.parse_number(wind_str)
        if value is not None:
            return value
        
        # Try to parse common wind speed patterns
        patterns = [
            r'(\d+)\s*kph',           # e.g., "25 kph"
            r'(\d+)\s*km/h',          # e.g., "25 km/h"
            r'(\d+)\s*mph',           # e.g., "15 mph"
            r'Gusts?\s*to\s*(\d+)',  # e.g., "Gusts to 35"
            r'(\d+)',                  # Just a number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, wind_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def extract_daily_temperatures(self, html: str, station: StationConfig) -> List[DailyTemperature]:
        """
        Extract daily temperatures from Wunderground HTML
        
        Args:
            html: HTML content
            station: Station configuration
            
        Returns:
            List of DailyTemperature objects
        """
        soup = BeautifulSoup(html, 'html.parser')
        daily_temps = []
        
        # Find the daily observations table
        daily_table = soup.find('table', class_='observations-table')
        
        if not daily_table:
            logger.warning("Daily observations table not found in HTML")
            return []
        
        # Extract rows (skip header)
        rows = daily_table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2:  # Need at least date and temperature columns
                continue
            
            # Extract date
            date_text = cols[0].text.strip()
            
            # Parse date (format varies, try common patterns)
            date_obj = None
            for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d %b %Y', '%b %d, %Y']:
                try:
                    date_obj = datetime.strptime(date_text, date_format).date()
                    break
                except ValueError:
                    continue
            
            if not date_obj:
                logger.debug(f"Could not parse date: {date_text}")
                continue
            
            # Extract temperature string (usually in second column)
            temp_str = cols[1].text.strip() if len(cols) > 1 else ""
            
            # Parse temperatures
            temp_max, temp_min = self.parse_temperature_string(temp_str, station.temperature_unit)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(temp_max, temp_min)
            
            daily_temp = DailyTemperature(
                date=date_obj,
                temp_max=temp_max,
                temp_min=temp_min,
                quality_score=quality_score,
                station_id=station.station_id
            )
            
            daily_temps.append(daily_temp)
        
        logger.info(f"Extracted {len(daily_temps)} daily temperature records")
        return daily_temps
    
    def extract_daily_observations(self, html: str, station: StationConfig) -> List[DailyObservation]:
        """
        Extract complete daily observations from Wunderground HTML
        
        Args:
            html: HTML content of the Wunderground monthly page
            station: Station configuration
            
        Returns:
            List of DailyObservation objects with all 12 fields
        """
        soup = BeautifulSoup(html, 'html.parser')
        observations = []
        
        # Find the observations table
        # First try to find by class names
        table = soup.find('table', class_='observations-table')
        if not table:
            table = soup.find('table', class_='history-observations-table')
        if not table:
            table = soup.find('table', class_='daily-history-table')
        
        # If not found by class, try by ID
        if not table:
            table = soup.find('table', id='observations-table')
        if not table:
            table = soup.find('table', id='history-observations-table')
        
        # If still not found, find the first table with td elements
        if not table:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('td'):
                    table = t
                    break
        
        if not table:
            logger.warning("No observations table found in HTML")
            return observations
        
        # Find all table rows (skip header row if present)
        rows = table.find_all('tr')
        # Skip header rows (those with th tags or containing header text)
        data_rows = []
        for row in rows:
            # Skip rows that are likely headers
            if row.find('th'):
                continue
            # Skip rows with very few cells (likely not data)
            cells = row.find_all('td')
            if len(cells) >= 2:  # Need at least date and temperature
                data_rows.append(row)
        
        rows = data_rows
        
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 10:  # Need at least all columns
                continue
            
            try:
                # Extract date
                date_str = cells[0].get_text(strip=True)
                if not date_str:
                    continue
                
                # Parse date (format: MM/DD/YYYY)
                try:
                    obs_date = datetime.strptime(date_str, '%m/%d/%Y').date()
                except ValueError:
                    # Try alternative format
                    try:
                        obs_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        logger.warning(f"Could not parse date: {date_str}")
                        continue
                
                # Extract all fields
                # Column indices based on Wunderground table structure:
                # 0: Date, 1: Temp High/Low, 2: Dew Pt High/Low, 3: Humidity, 
                # 4: Wind, 5: Max Wind, 6: Wind Gust, 7: Pressure, 
                # 8: Precipitation, 9: Condition
                
                # Temperature (col 1)
                temp_str = cells[1].get_text(strip=True)
                temp_high, temp_low = self.parse_temperature_string(temp_str, station.temperature_unit)
                
                # Dew point (col 2)
                dew_pt_str = cells[2].get_text(strip=True)
                dew_pt_high, dew_pt_low = self.parse_temperature_string(dew_pt_str, station.temperature_unit)
                
                # Humidity (col 3)
                humidity_str = cells[3].get_text(strip=True)
                humidity = self.parse_number(humidity_str)
                
                # Wind description (col 4)
                wind = cells[4].get_text(strip=True) or None
                
                # Max wind speed (col 5)
                max_wind_str = cells[5].get_text(strip=True)
                max_wind = self.parse_wind_speed(max_wind_str)
                
                # Wind gust (col 6)
                wind_gust_str = cells[6].get_text(strip=True)
                wind_gust = self.parse_wind_speed(wind_gust_str)
                
                # Pressure (col 7)
                pressure_str = cells[7].get_text(strip=True)
                pressure = self.parse_number(pressure_str)
                
                # Precipitation (col 8)
                precipitation_str = cells[8].get_text(strip=True)
                precipitation = self.parse_number(precipitation_str)
                
                # Condition (col 9)
                condition = cells[9].get_text(strip=True) or None
                
                # Calculate quality score for all fields
                quality_score = self._calculate_quality_score_all_fields(
                    temp_high, temp_low, dew_pt_high, dew_pt_low,
                    humidity, max_wind, wind_gust, pressure, precipitation
                )
                
                # Determine quality level
                quality_level = self._determine_quality_level(quality_score)
                
                # Create DailyObservation object
                observation = DailyObservation(
                    date=obs_date,
                    station_id=station.station_id,
                    temp_high=temp_high,
                    temp_low=temp_low,
                    dew_pt_high=dew_pt_high,
                    dew_pt_low=dew_pt_low,
                    humidity=humidity,
                    wind=wind,
                    max_wind=max_wind,
                    wind_gust=wind_gust,
                    pressure=pressure,
                    precipitation=precipitation,
                    condition=condition,
                    quality_score=quality_score,
                    quality_level=quality_level,
                    data_source="wunderground"
                )
                
                observations.append(observation)
                
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                continue
        
        logger.info(f"Extracted {len(observations)} daily observations with all fields")
        return observations
    
    def _calculate_quality_score(self, temp_max: Optional[float], temp_min: Optional[float]) -> float:
        """
        Calculate data quality score (0.0 to 1.0)
        
        Args:
            temp_max: Maximum temperature
            temp_min: Minimum temperature
            
        Returns:
            Quality score (1.0 = perfect, 0.0 = invalid)
        """
        score = 1.0
        
        # Check if values are present
        if temp_max is None or temp_min is None:
            score *= 0.5
        
        # Check if max > min (when both present)
        if temp_max is not None and temp_min is not None:
            if temp_max < temp_min:
                score *= 0.3
        
        # Check for reasonable temperature ranges (in Celsius)
        if temp_max is not None:
            if temp_max < -50 or temp_max > 60:  # Unlikely but possible extremes
                score *= 0.7
            elif temp_max < -30 or temp_max > 50:  # Very extreme
                score *= 0.5
        
        if temp_min is not None:
            if temp_min < -60 or temp_min > 50:
                score *= 0.7
            elif temp_min < -40 or temp_min > 40:
                score *= 0.5
        
        return score
    
    def _calculate_quality_score_all_fields(
        self,
        temp_high: Optional[float],
        temp_low: Optional[float],
        dew_pt_high: Optional[float],
        dew_pt_low: Optional[float],
        humidity: Optional[float],
        max_wind: Optional[float],
        wind_gust: Optional[float],
        pressure: Optional[float],
        precipitation: Optional[float]
    ) -> float:
        """
        Calculate quality score considering all meteorological fields
        
        Args:
            All field values (can be None)
            
        Returns:
            Quality score (1.0 = perfect, 0.0 = invalid)
        """
        score = 1.0
        total_fields = 9
        missing_fields = 0
        
        # Check for missing values
        fields = [temp_high, temp_low, dew_pt_high, dew_pt_low, humidity,
                  max_wind, wind_gust, pressure, precipitation]
        
        for field in fields:
            if field is None:
                missing_fields += 1
        
        # Deduct for missing fields (more missing = lower score)
        if missing_fields > 0:
            score -= (missing_fields / total_fields) * 0.5
        
        # Check temperature consistency
        if temp_high is not None and temp_low is not None:
            if temp_high < temp_low:
                score -= 0.1  # High < Low is illogical
            
            # Check for extreme temperatures (in Celsius)
            if temp_high > 60 or temp_high < -80:
                score -= 0.1
            if temp_low > 60 or temp_low < -80:
                score -= 0.1
        
        # Check dew point consistency
        if dew_pt_high is not None and dew_pt_low is not None:
            if dew_pt_high < dew_pt_low:
                score -= 0.1  # High < Low is illogical
            
            # Dew point should generally be lower than temperature
            if temp_high is not None and dew_pt_high is not None:
                if dew_pt_high > temp_high + 5:  # Allow small margin
                    score -= 0.05
        
        # Check humidity range (0-100%)
        if humidity is not None:
            if humidity < 0 or humidity > 100:
                score -= 0.1
        
        # Check wind speed consistency
        if max_wind is not None and wind_gust is not None:
            if max_wind > wind_gust:
                score -= 0.05  # Max wind should not exceed gust
            if max_wind > 200:  # Unlikely high wind speed (km/h)
                score -= 0.1
            if wind_gust > 250:  # Unlikely high gust (km/h)
                score -= 0.1
        
        # Check pressure range (reasonable atmospheric pressure in hPa)
        if pressure is not None:
            if pressure < 870 or pressure > 1085:  # World record extremes
                score -= 0.1
        
        # Check precipitation (non-negative)
        if precipitation is not None:
            if precipitation < 0:
                score -= 0.1
            if precipitation > 1000:  # Unlikely daily precipitation (mm)
                score -= 0.1
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))
    
    def _determine_quality_level(self, quality_score: float) -> DataQuality:
        """
        Determine quality level based on score
        
        Args:
            quality_score: Score from 0.0 to 1.0
            
        Returns:
            DataQuality enum value
        """
        if quality_score >= 0.9:
            return DataQuality.EXCELLENT
        elif quality_score >= 0.7:
            return DataQuality.GOOD
        elif quality_score >= 0.5:
            return DataQuality.WARNING
        else:
            return DataQuality.ERROR
    
    def validate_daily_data(self, daily_temps: List[DailyTemperature]) -> Dict[str, Any]:
        """
        Validate daily temperature data
        
        Args:
            daily_temps: List of daily temperature records
            
        Returns:
            Dictionary with validation results
        """
        if not daily_temps:
            return {
                'valid': False,
                'num_records': 0,
                'num_valid': 0,
                'completeness': 0.0,
                'avg_quality': 0.0,
                'issues': ['No data']
            }
        
        num_records = len(daily_temps)
        num_valid = sum(1 for dt in daily_temps if dt.temp_max is not None and dt.temp_min is not None)
        completeness = num_valid / num_records if num_records > 0 else 0.0
        avg_quality = sum(dt.quality_score for dt in daily_temps) / num_records
        
        issues = []
        
        # Check completeness
        if completeness < 0.8:
            issues.append(f"Low completeness: {completeness:.1%}")
        
        # Check quality
        if avg_quality < 0.7:
            issues.append(f"Low average quality: {avg_quality:.2f}")
        
        # Check for date gaps
        dates = sorted([dt.date for dt in daily_temps])
        if len(dates) > 1:
            date_gaps = []
            for i in range(1, len(dates)):
                gap = (dates[i] - dates[i-1]).days
                if gap > 1:
                    date_gaps.append(f"{dates[i-1]} to {dates[i]}: {gap} days")
            
            if date_gaps:
                issues.append(f"Date gaps found: {', '.join(date_gaps[:3])}")
        
        # Check temperature consistency
        temp_diffs = []
        for dt in daily_temps:
            if dt.temp_max is not None and dt.temp_min is not None:
                diff = dt.temp_max - dt.temp_min
                if diff < 0:
                    issues.append(f"Negative temp diff on {dt.date}: max={dt.temp_max}, min={dt.temp_min}")
                elif diff > 30:  # Unusually large daily range
                    issues.append(f"Large temp diff on {dt.date}: {diff:.1f}°C")
        
        return {
            'valid': len(issues) == 0,
            'num_records': num_records,
            'num_valid': num_valid,
            'completeness': completeness,
            'avg_quality': avg_quality,
            'issues': issues
        }
    
    def save_temperature_data(self, daily_temps: List[DailyTemperature], station_id: str, year: int, month: int):
        """
        Save temperature data to cache database and CSV
        
        Args:
            daily_temps: List of daily temperature records
            station_id: Station ID
            year: Year
            month: Month
        """
        if not daily_temps:
            logger.warning(f"No data to save for {station_id} {year}-{month}")
            return
        
        # Save to database
        cursor = self.conn.cursor()
        for dt in daily_temps:
            cursor.execute('''
                INSERT OR REPLACE INTO temperature_data 
                (station_id, date, temp_max, temp_min, quality_score, source_url)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                dt.station_id,
                dt.date.isoformat(),
                dt.temp_max,
                dt.temp_min,
                dt.quality_score,
                f"wunderground/{station_id}/{year}/{month}"
            ))
        
        self.conn.commit()
        
        # Save to CSV
        station_dir = self.cache_dir / station_id / str(year)
        station_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = station_dir / f"{month:02d}.csv"
        data_dicts = []
        for dt in daily_temps:
            data_dicts.append({
                'date': dt.date.isoformat(),
                'temp_max_c': dt.temp_max,
                'temp_min_c': dt.temp_min,
                'quality_score': dt.quality_score,
                'station_id': dt.station_id,
                'data_source': dt.data_source
            })
        
        df = pd.DataFrame(data_dicts)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(daily_temps)} records to {csv_path}")
    
    def fetch_station_month(self, station_id: str, year: int, month: int) -> List[DailyTemperature]:
        """
        Fetch temperature data for a specific station, year, and month
        
        Args:
            station_id: Station ID (e.g., 'ZSPD', 'KDEN')
            year: Year
            month: Month (1-12)
            
        Returns:
            List of DailyTemperature objects
        """
        if station_id not in self.stations:
            logger.error(f"Unknown station: {station_id}")
            return []
        
        station = self.stations[station_id]
        
        # Check if data already exists in cache
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM temperature_data 
            WHERE station_id = ? AND strftime('%Y', date) = ? AND strftime('%m', date) = ?
        ''', (station_id, str(year), f"{month:02d}"))
        
        count = cursor.fetchone()[0]
        if count > 20:  # If we have most days cached
            logger.info(f"Found {count} cached records for {station_id} {year}-{month}")
            cursor.execute('''
                SELECT date, temp_max, temp_min, quality_score 
                FROM temperature_data 
                WHERE station_id = ? AND strftime('%Y', date) = ? AND strftime('%m', date) = ?
            ''', (station_id, str(year), f"{month:02d}"))
            
            daily_temps = []
            for row in cursor.fetchall():
                daily_temps.append(DailyTemperature(
                    date=datetime.strptime(row[0], '%Y-%m-%d').date(),
                    temp_max=row[1],
                    temp_min=row[2],
                    quality_score=row[3],
                    station_id=station_id
                ))
            
            return daily_temps
        
        # Fetch from Wunderground
        url = self.build_wunderground_url(station, year, month)
        html = self.fetch_page_with_retry(url)
        
        if not html:
            logger.error(f"Failed to fetch data for {station_id} {year}-{month}")
            return []
        
        # Parse data
        daily_temps = self.extract_daily_temperatures(html, station)
        
        # Validate data
        validation = self.validate_daily_data(daily_temps)
        if not validation['valid']:
            logger.warning(f"Data validation issues for {station_id} {year}-{month}: {validation['issues']}")
        
        # Save data
        self.save_temperature_data(daily_temps, station_id, year, month)
        
        return daily_temps
    
    def fetch_station_range(self, station_id: str, start_date: date, end_date: date) -> List[DailyTemperature]:
        """
        Fetch temperature data for a date range
        
        Args:
            station_id: Station ID
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of DailyTemperature objects for the entire range
        """
        all_daily_temps = []
        
        current_date = start_date.replace(day=1)  # Start from first of month
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            logger.info(f"Fetching {station_id} {year}-{month:02d}")
            
            monthly_temps = self.fetch_station_month(station_id, year, month)
            all_daily_temps.extend(monthly_temps)
            
            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)
        
        # Filter to date range
        filtered_temps = [dt for dt in all_daily_temps if start_date <= dt.date <= end_date]
        
        # Validate overall dataset
        validation = self.validate_daily_data(filtered_temps)
        logger.info(f"Fetched {len(filtered_temps)} records for {station_id} from {start_date} to {end_date}")
        logger.info(f"Validation: {validation['num_valid']}/{validation['num_records']} valid records "
                   f"({validation['completeness']:.1%} completeness, "
                   f"quality: {validation['avg_quality']:.2f})")
        
        if validation['issues']:
            logger.warning(f"Validation issues: {validation['issues']}")
        
        return filtered_temps
    
    def fetch_station_month_full(self, station_id: str, year: int, month: int) -> List[DailyObservation]:
        """
        Fetch complete observation data for a specific station, year, and month
        
        Args:
            station_id: Station ID (e.g., 'ZSPD', 'KDEN')
            year: Year
            month: Month (1-12)
            
        Returns:
            List of DailyObservation objects with all 12 fields
        """
        if station_id not in self.stations:
            logger.error(f"Unknown station: {station_id}")
            return []
        
        station = self.stations[station_id]
        
        # Build URL
        url = self.build_wunderground_url(station, year, month)
        logger.info(f"Fetching full data from: {url}")
        
        # Fetch HTML page
        html = self.fetch_page_with_retry(url)
        if not html:
            logger.error(f"Failed to fetch data for {station_id} {year}-{month:02d}")
            return []
        
        # Extract complete observations
        observations = self.extract_daily_observations(html, station)
        
        if observations:
            logger.info(f"Extracted {len(observations)} daily observations for {station_id} {year}-{month:02d}")
            
            # Cache the HTML for future use
            self._cache_page(url, html, 200)
            
            # Save temperature data to cache (for backward compatibility)
            daily_temps = [DailyTemperature(
                date=obs.date,
                temp_max=obs.temp_high,
                temp_min=obs.temp_low,
                quality_score=obs.quality_score,
                station_id=obs.station_id,
                data_source=obs.data_source
            ) for obs in observations]
            
            self.save_temperature_data(daily_temps, station_id, year, month)
        else:
            logger.warning(f"No observations extracted for {station_id} {year}-{month:02d}")
        
        return observations
    
    def fetch_station_range_full(self, station_id: str, start_date: date, end_date: date) -> List[DailyObservation]:
        """
        Fetch complete observation data for a date range
        
        Args:
            station_id: Station ID
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of DailyObservation objects with all 12 fields
        """
        if station_id not in self.stations:
            logger.error(f"Unknown station: {station_id}")
            return []
        
        all_observations = []
        
        current_date = start_date.replace(day=1)  # Start from first of month
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            logger.info(f"Fetching full data for {station_id} {year}-{month:02d}")
            
            monthly_observations = self.fetch_station_month_full(station_id, year, month)
            all_observations.extend(monthly_observations)
            
            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year + 1, month=1)
            else:
                current_date = current_date.replace(month=month + 1)
        
        # Filter to date range
        filtered_observations = [obs for obs in all_observations if start_date <= obs.date <= end_date]
        
        logger.info(f"Fetched {len(filtered_observations)} daily observations for {station_id} from {start_date} to {end_date}")
        
        return filtered_observations
    
    def export_to_dataframe(self, daily_temps: List[DailyTemperature]) -> pd.DataFrame:
        """
        Convert list of DailyTemperature objects to pandas DataFrame
        
        Args:
            daily_temps: List of daily temperature records
            
        Returns:
            pandas DataFrame with temperature data
        """
        data = []
        for dt in daily_temps:
            data.append({
                'date': dt.date,
                'temp_max_c': dt.temp_max,
                'temp_min_c': dt.temp_min,
                'quality_score': dt.quality_score,
                'station_id': dt.station_id,
                'data_source': dt.data_source
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def save_observations(self, observations: List[DailyObservation]) -> int:
        """
        Save observations to SQLite database
        
        Args:
            observations: List of DailyObservation objects
            
        Returns:
            Number of observations saved
        """
        if not observations:
            return 0
        
        saved_count = 0
        try:
            cursor = self.db_conn.cursor()
            
            for obs in observations:
                # Convert to dictionary for insertion
                obs_dict = obs.to_dict()
                
                # Prepare values for insertion
                values = (
                    obs_dict['date'],
                    obs_dict['station_id'],
                    obs_dict['temp_high'],
                    obs_dict['temp_low'],
                    obs_dict['dew_pt_high'],
                    obs_dict['dew_pt_low'],
                    obs_dict['humidity'],
                    obs_dict['wind'],
                    obs_dict['max_wind'],
                    obs_dict['wind_gust'],
                    obs_dict['pressure'],
                    obs_dict['precipitation'],
                    obs_dict['condition'],
                    obs_dict['quality_score'],
                    obs_dict['quality_level'],
                    json.dumps(obs_dict['validation_errors']),  # Convert list to JSON
                    obs_dict['data_source'],
                    obs_dict['scraped_at']
                )
                
                # UPSERT (INSERT OR REPLACE) to handle duplicates
                cursor.execute('''
                    INSERT OR REPLACE INTO observations 
                    (date, station_id, temp_high, temp_low, dew_pt_high, dew_pt_low,
                     humidity, wind, max_wind, wind_gust, pressure, precipitation,
                     condition, quality_score, quality_level, validation_errors,
                     data_source, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', values)
                
                saved_count += 1
            
            self.db_conn.commit()
            logger.info(f"Saved {saved_count} observations to database")
            
        except Exception as e:
            logger.error(f"Error saving observations to database: {e}")
            self.db_conn.rollback()
            raise
        
        return saved_count
    
    def load_observations(self, station_id: str, start_date: date, end_date: date) -> List[DailyObservation]:
        """
        Load observations from SQLite database for a specific station and date range
        
        Args:
            station_id: Station identifier
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            List of DailyObservation objects
        """
        observations = []
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT date, station_id, temp_high, temp_low, dew_pt_high, dew_pt_low,
                       humidity, wind, max_wind, wind_gust, pressure, precipitation,
                       condition, quality_score, quality_level, validation_errors,
                       data_source, scraped_at
                FROM observations
                WHERE station_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            ''', (station_id, start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            
            for row in rows:
                # Convert row to dictionary
                obs_dict = {
                    'date': row[0],
                    'station_id': row[1],
                    'temp_high': row[2],
                    'temp_low': row[3],
                    'dew_pt_high': row[4],
                    'dew_pt_low': row[5],
                    'humidity': row[6],
                    'wind': row[7],
                    'max_wind': row[8],
                    'wind_gust': row[9],
                    'pressure': row[10],
                    'precipitation': row[11],
                    'condition': row[12],
                    'quality_score': row[13],
                    'quality_level': row[14],
                    'validation_errors': json.loads(row[15]) if row[15] else [],
                    'data_source': row[16],
                    'scraped_at': row[17]
                }
                
                # Create DailyObservation object
                observation = DailyObservation.from_dict(obs_dict)
                observations.append(observation)
            
            logger.info(f"Loaded {len(observations)} observations for station {station_id} from {start_date} to {end_date}")
            
        except Exception as e:
            logger.error(f"Error loading observations from database: {e}")
            raise
        
        return observations
    
    def get_observation_count(self, station_id: Optional[str] = None) -> int:
        """
        Get count of observations in database
        
        Args:
            station_id: Optional station identifier to filter by station
            
        Returns:
            Number of observations
        """
        try:
            cursor = self.db_conn.cursor()
            
            if station_id:
                cursor.execute('SELECT COUNT(*) FROM observations WHERE station_id = ?', (station_id,))
            else:
                cursor.execute('SELECT COUNT(*) FROM observations')
            
            count = cursor.fetchone()[0]
            return count
            
        except Exception as e:
            logger.error(f"Error getting observation count: {e}")
            return 0
    
    def get_observation_date_range(self, station_id: str) -> Tuple[Optional[date], Optional[date]]:
        """
        Get date range of observations for a station
        
        Args:
            station_id: Station identifier
            
        Returns:
            Tuple of (start_date, end_date) or (None, None) if no data
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Get earliest date
            cursor.execute('SELECT MIN(date) FROM observations WHERE station_id = ?', (station_id,))
            min_date_str = cursor.fetchone()[0]
            
            # Get latest date
            cursor.execute('SELECT MAX(date) FROM observations WHERE station_id = ?', (station_id,))
            max_date_str = cursor.fetchone()[0]
            
            if min_date_str and max_date_str:
                min_date = date.fromisoformat(min_date_str)
                max_date = date.fromisoformat(max_date_str)
                return min_date, max_date
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting observation date range: {e}")
            return None, None
    
    def delete_observations(self, station_id: str) -> int:
        """
        Delete all observations for a station
        
        Args:
            station_id: Station identifier
            
        Returns:
            Number of observations deleted
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Get count before deletion
            cursor.execute('SELECT COUNT(*) FROM observations WHERE station_id = ?', (station_id,))
            count_before = cursor.fetchone()[0]
            
            # Delete observations
            cursor.execute('DELETE FROM observations WHERE station_id = ?', (station_id,))
            deleted_count = cursor.rowcount
            
            self.db_conn.commit()
            logger.info(f"Deleted {deleted_count} observations for station {station_id}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting observations: {e}")
            self.db_conn.rollback()
            return 0
    
    def save_download_progress(self, station_id: str, year: int, month: int, 
                               status: str, records_downloaded: int = 0, 
                               error_message: Optional[str] = None) -> bool:
        """
        Save download progress for a station and month
        
        Args:
            station_id: Station identifier
            year: Year
            month: Month (1-12)
            status: Download status ('pending', 'in_progress', 'completed', 'failed')
            records_downloaded: Number of records downloaded
            error_message: Error message if failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db_conn.cursor()
            
            completed_at = datetime.now().isoformat() if status in ['completed', 'failed'] else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO download_progress 
                (station_id, year, month, status, records_downloaded, error_message, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (station_id, year, month, status, records_downloaded, error_message, completed_at))
            
            self.db_conn.commit()
            logger.debug(f"Saved download progress: {station_id} {year}-{month:02d} - {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving download progress: {e}")
            self.db_conn.rollback()
            return False
    
    def get_download_progress(self, station_id: str, year: int, month: int) -> Optional[Dict[str, Any]]:
        """
        Get download progress for a station and month
        
        Args:
            station_id: Station identifier
            year: Year
            month: Month (1-12)
            
        Returns:
            Dictionary with progress info or None if not found
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT station_id, year, month, status, records_downloaded, 
                       error_message, started_at, completed_at
                FROM download_progress
                WHERE station_id = ? AND year = ? AND month = ?
            ''', (station_id, year, month))
            
            row = cursor.fetchone()
            if row:
                return {
                    'station_id': row[0],
                    'year': row[1],
                    'month': row[2],
                    'status': row[3],
                    'records_downloaded': row[4],
                    'error_message': row[5],
                    'started_at': row[6],
                    'completed_at': row[7]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting download progress: {e}")
            return None
    
    def get_all_download_progress(self, station_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all download progress records
        
        Args:
            station_id: Optional station identifier to filter by station
            
        Returns:
            List of progress dictionaries
        """
        try:
            cursor = self.db_conn.cursor()
            
            if station_id:
                cursor.execute('''
                    SELECT station_id, year, month, status, records_downloaded, 
                           error_message, started_at, completed_at
                    FROM download_progress
                    WHERE station_id = ?
                    ORDER BY year, month
                ''', (station_id,))
            else:
                cursor.execute('''
                    SELECT station_id, year, month, status, records_downloaded, 
                           error_message, started_at, completed_at
                    FROM download_progress
                    ORDER BY station_id, year, month
                ''')
            
            rows = cursor.fetchall()
            progress_list = []
            
            for row in rows:
                progress_list.append({
                    'station_id': row[0],
                    'year': row[1],
                    'month': row[2],
                    'status': row[3],
                    'records_downloaded': row[4],
                    'error_message': row[5],
                    'started_at': row[6],
                    'completed_at': row[7]
                })
            
            return progress_list
            
        except Exception as e:
            logger.error(f"Error getting all download progress: {e}")
            return []
    
    def export_observations_to_dataframe(self, observations: List[DailyObservation]) -> pd.DataFrame:
        """
        Export observations to pandas DataFrame
        
        Args:
            observations: List of DailyObservation objects
            
        Returns:
            pandas DataFrame with all observation fields
        """
        if not observations:
            return pd.DataFrame()
        
        data = []
        for obs in observations:
            obs_dict = obs.to_dict()
            # Convert quality_level from enum to string
            obs_dict['quality_level'] = obs.quality_level.value
            data.append(obs_dict)
        
        df = pd.DataFrame(data)
        if len(df) > 0 and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Cache database connection closed")
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
            logger.info("Observations database connection closed")


def main():
    """Main function for testing and demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wunderground historical data scraper')
    parser.add_argument('--station', type=str, default='ZSPD', 
                       choices=['ZSPD', 'KDEN'], help='Station ID')
    parser.add_argument('--start-year', type=int, default=2000, help='Start year')
    parser.add_argument('--end-year', type=int, default=2019, help='End year')
    parser.add_argument('--output', type=str, default='temperature_data.csv', 
                       help='Output CSV file')
    parser.add_argument('--config', type=str, default='configs/stations.yaml',
                       help='Path to station configuration file')
    
    args = parser.parse_args()
    
    # Create scraper
    scraper = WundergroundScraper(config_path=args.config)
    
    try:
        # Fetch data for the specified range
        start_date = date(args.start_year, 1, 1)
        end_date = date(args.end_year, 12, 31)
        
        logger.info(f"Fetching data for {args.station} from {start_date} to {end_date}")
        
        daily_temps = scraper.fetch_station_range(args.station, start_date, end_date)
        
        if daily_temps:
            # Export to DataFrame and save
            df = scraper.export_to_dataframe(daily_temps)
            df.to_csv(args.output, index=False)
            logger.info(f"Data saved to {args.output}")
            
            # Print summary
            print(f"\n=== Data Summary for {args.station} ===")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"Completeness: {(1 - df['temp_max_c'].isna().mean()):.1%} for max temp")
            print(f"Completeness: {(1 - df['temp_min_c'].isna().mean()):.1%} for min temp")
            print(f"Average quality score: {df['quality_score'].mean():.3f}")
            print(f"\nFirst 5 records:")
            print(df.head().to_string())
            
            # Save validation report
            validation = scraper.validate_daily_data(daily_temps)
            report_path = f"validation_report_{args.station}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Validation Report for {args.station}\n")
                f.write(f"Date range: {start_date} to {end_date}\n")
                f.write(f"Total records: {validation['num_records']}\n")
                f.write(f"Valid records: {validation['num_valid']}\n")
                f.write(f"Completeness: {validation['completeness']:.1%}\n")
                f.write(f"Average quality: {validation['avg_quality']:.3f}\n")
                f.write(f"Valid: {validation['valid']}\n")
                f.write("\nIssues:\n")
                for issue in validation['issues']:
                    f.write(f"  - {issue}\n")
            
            logger.info(f"Validation report saved to {report_path}")
        else:
            logger.error("No data fetched")
            
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
MarketDataLoader - Fetch OHLCV data for equities, ETFs, FX, crypto, bonds/futures
and options chains using Yahoo Finance API.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
from typing import Dict, Tuple, Optional, Union


class MarketDataLoader:
    """
    Fetches multi-asset market data via Yahoo Finance with caching capabilities.
    Supports period-based and explicit date range queries.
    """
    
    def __init__(self, interval: str = "1d", period: str = "1y"):
        """
        Initialize MarketDataLoader with default interval and period.
        
        Args:
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            period: Default period for period-based queries (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        self.interval = interval
        self.period = period
        self._period_cache = {}  # Cache for period-based queries
        self._range_cache = {}   # Cache for date-range queries
    
    def _rename_and_tz(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to standard format and ensure UTC timezone-aware DatetimeIndex.
        
        Args:
            df: Raw DataFrame from yfinance
            
        Returns:
            DataFrame with standardized columns and UTC timezone
        """
        # Rename columns to standard format
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'last_price',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'last_price', 'volume']
        for col in required_cols:
            if col not in df.columns:
                # If Close exists but last_price doesn't, map it
                if col == 'last_price' and 'Close' in df.columns:
                    df['last_price'] = df['Close']
                elif col not in df.columns:
                    df[col] = 0  # Default to 0 for missing columns
        
        # Ensure UTC timezone-aware DatetimeIndex
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif df.index.tz != pytz.UTC:
            df.index = df.index.tz_convert('UTC')
            
        return df
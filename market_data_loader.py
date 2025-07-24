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
    def _load_period(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a symbol using the default period setting.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{self.interval}_{self.period}"
        
        if cache_key in self._period_cache:
            return self._period_cache[cache_key]
        
        try:
            df = yf.download(
                symbol, 
                period=self.period, 
                interval=self.interval, 
                auto_adjust=True,
                progress=False
            )
            
            if df.empty:
                raise ValueError(f"No data returned for symbol {symbol}")
            
            df = self._rename_and_tz(df)
            self._period_cache[cache_key] = df
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_history(self, symbol: str, start: Optional[str] = None, 
                   end: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol
            start: Start date (YYYY-MM-DD format), optional
            end: End date (YYYY-MM-DD format), optional
            
        Returns:
            DataFrame with columns: open, high, low, last_price, volume
        """
        if start is not None and end is not None:
            # Use explicit date range
            cache_key = f"{symbol}_{self.interval}_{start}_{end}"
            
            if cache_key in self._range_cache:
                return self._range_cache[cache_key]
            
            try:
                df = yf.download(
                    symbol,
                    start=start,
                    end=end,
                    interval=self.interval,
                    auto_adjust=True,
                    progress=False
                )
                
                if df.empty:
                    raise ValueError(f"No data returned for symbol {symbol} in range {start} to {end}")
                
                df = self._rename_and_tz(df)
                self._range_cache[cache_key] = df
                return df
                
            except Exception as e:
                raise ValueError(f"Failed to fetch data for {symbol} ({start} to {end}): {str(e)}")
        else:
            # Use period-based query
            return self._load_period(symbol)
    
    def _locate_timestamp(self, df: pd.DataFrame, ts: datetime) -> int:
        """
        Locate the index position for a given timestamp in the DataFrame.
        
        Args:
            df: DataFrame with DatetimeIndex
            ts: Target timestamp
            
        Returns:
            Integer index position
        """
        # Convert timestamp to DataFrame's timezone if needed
        if ts.tzinfo is None:
            ts = pytz.UTC.localize(ts)
        elif ts.tzinfo != df.index.tz:
            ts = ts.astimezone(df.index.tz)
        
        # If exact timestamp exists, return it
        if ts in df.index:
            return df.index.get_loc(ts)
        
        # Otherwise, find nearest prior bar using forward fill logic
        indexer = df.index.get_indexer([ts], method="ffill")
        return indexer[0] if indexer[0] != -1 else 0
    
    def _scalar_to_float(self, x) -> float:
        """Convert pandas scalar to Python float."""
        if pd.isna(x):
            return 0.0
        if hasattr(x, 'item'):
            return float(x.item())
        return float(x)
    
    def _scalar_to_int(self, x) -> int:
        """Convert pandas scalar to Python int."""
        if pd.isna(x):
            return 0
        if hasattr(x, 'item'):
            return int(x.item())
        return int(x)
    
    def get_price(self, symbol: str, timestamp: datetime) -> float:
        """
        Get the last price for a symbol at a specific timestamp.
        
        Args:
            symbol: Ticker symbol
            timestamp: Target timestamp
            
        Returns:
            Last price as float
        """
        try:
            df = self._load_period(symbol)
            idx = self._locate_timestamp(df, timestamp)
            price = df.iloc[idx]['last_price']
            return self._scalar_to_float(price)
        except Exception as e:
            raise ValueError(f"Failed to get price for {symbol} at {timestamp}: {str(e)}")
    
    def get_bid_ask(self, symbol: str, timestamp: datetime) -> Tuple[float, float]:
        """
        Get bid and ask prices for a symbol at a specific timestamp.
        Estimates spreads based on asset type since Yahoo Finance doesn't provide bid/ask.
        
        Args:
            symbol: Ticker symbol
            timestamp: Target timestamp
            
        Returns:
            Tuple of (bid, ask) prices
        """
        try:
            price = self.get_price(symbol, timestamp)
            
            # Estimate spread based on asset type
            if '=X' in symbol:  # FX pairs
                spread_pct = 0.0001  # 1 pip spread
            elif 'BTC' in symbol or 'ETH' in symbol:  # Crypto
                spread_pct = 0.001   # 0.1% spread
            elif any(bond in symbol for bond in ['^TNX', '^FVX', 'ZN=F']):  # Bonds/Futures
                spread_pct = 0.0002  # 2 basis points
            else:  # Equities/ETFs
                spread_pct = 0.0001  # 1 basis point
            
            half_spread = price * spread_pct / 2
            bid = price - half_spread
            ask = price + half_spread
            
            return (bid, ask)
            
        except Exception as e:
            raise ValueError(f"Failed to get bid/ask for {symbol} at {timestamp}: {str(e)}")
    
    def get_volume(self, symbol: str, start: str, end: str) -> int:
        """
        Get total volume for a symbol between start and end dates.
        
        Args:
            symbol: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            Total volume as integer
        """
        try:
            df = self.get_history(symbol, start=start, end=end)
            total_volume = df['volume'].sum()
            return self._scalar_to_int(total_volume)
        except Exception as e:
            raise ValueError(f"Failed to get volume for {symbol} ({start} to {end}): {str(e)}")
    
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying ticker symbol
            expiry: Expiration date (YYYY-MM-DD), if None uses nearest expiry
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
            
            # Choose expiry date
            if expiry is None:
                target_expiry = expirations[0]  # Nearest expiry
            else:
                # Convert expiry to the format used by yfinance
                target_date = datetime.strptime(expiry, '%Y-%m-%d').date()
                target_expiry = None
                
                for exp_str in expirations:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                    if exp_date >= target_date:
                        target_expiry = exp_str
                        break
                
                if target_expiry is None:
                    target_expiry = expirations[-1]  # Use latest if no match found
            
            # Get option chain for the target expiry
            opt_chain = ticker.option_chain(target_expiry)
            
            # Standardize column names
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()
            
            # Add expiry column
            calls['expiry'] = target_expiry
            puts['expiry'] = target_expiry
            
            return {
                "calls": calls,
                "puts": puts
            }
            
        except Exception as e:
            # Return empty DataFrames if options data unavailable
            return {
                "calls": pd.DataFrame(), 
                "puts": pd.DataFrame()
            }
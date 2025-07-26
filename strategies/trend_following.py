#!/usr/bin/env python3
"""
Trend Following Strategy - Enter long on MA crossover, exit on reverse crossover
"""

import pandas as pd
import numpy as np
from order import Order
from oms import OrderManagementSystem
from order_book import LimitOrderBook
from position_tracker import PositionTracker
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Any


def run_backtest(history: pd.DataFrame, short_win: int = 10, long_win: int = 50, 
                risk_params: Dict = None, starting_cash: float = 100000.0) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Run trend following backtest using moving average crossover strategy.
    
    Args:
        history: DataFrame with OHLCV data (columns: open, high, low, last_price, volume)
        short_win: Short moving average window
        long_win: Long moving average window  
        risk_params: Risk management parameters
        starting_cash: Starting cash amount
        
    Returns:
        Tuple of (signals_df, trades_list, metrics_dict)
    """
        # Default risk parameters
    if risk_params is None:
        risk_params = {
            "max_position": 1000,  # Maximum position size
            "position_pct": 0.1,   # Percentage of cash to risk per trade
            "stop_loss": None,     # Stop loss percentage (optional)
            "take_profit": None    # Take profit percentage (optional)
        }
    
    symbol = "TREND_ASSET"  # Generic symbol for backtesting
    
    # Validate inputs
    if len(history) < long_win:
        raise ValueError(f"History length {len(history)} is less than long window {long_win}")
    
    # Make a copy to avoid modifying original
    df = history.copy()
    
    # Calculate moving averages
    df["ma_short"] = df["last_price"].rolling(window=short_win, min_periods=short_win).mean()
    df["ma_long"] = df["last_price"].rolling(window=long_win, min_periods=long_win).mean()
    
    # Generate raw signals
    df["ma_short_prev"] = df["ma_short"].shift(1)
    df["ma_long_prev"] = df["ma_long"].shift(1)
    
    # Detect crossovers
    # Golden cross: short MA crosses above long MA -> buy signal
    df["golden_cross"] = (
        (df["ma_short"] > df["ma_long"]) & 
        (df["ma_short_prev"] <= df["ma_long_prev"])
    )
    
    # Death cross: short MA crosses below long MA -> sell signal  
    df["death_cross"] = (
        (df["ma_short"] < df["ma_long"]) & 
        (df["ma_short_prev"] >= df["ma_long_prev"])
    )
    
    # Generate position signals
    # +1 = long, 0 = flat, -1 = short (not used in this strategy)
    df["raw_signal"] = 0
    df.loc[df["golden_cross"], "raw_signal"] = 1   # Enter long
    df.loc[df["death_cross"], "raw_signal"] = 0    # Exit to flat
    
    # Forward fill signals to maintain position until next signal
    df["signal"] = df["raw_signal"].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    # Clean signals - only trade on actual signal changes
    df["signal_change"] = df["signal"] != df["signal"].shift(1)
    df["prev_signal"] = df["signal"].shift(1).fillna(0)
    
    # Create signals DataFrame
    signals_df = df[["ma_short", "ma_long", "golden_cross", "death_cross", 
                    "raw_signal", "signal", "signal_change", "prev_signal"]].copy()
    
    # Initialize trading infrastructure
    oms = OrderManagementSystem()
    book = LimitOrderBook(symbol)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades_list = []

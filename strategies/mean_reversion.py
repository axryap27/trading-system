"""
Mean Reversion Strategy - Use Bollinger Bands to buy low and sell high
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


def run_backtest(history: pd.DataFrame, bollinger_win: int = 20, num_std: float = 2.0,
                risk_params: Dict = None, starting_cash: float = 100000.0) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Run mean reversion backtest using Bollinger Bands strategy.
    
    Args:
        history: DataFrame with OHLCV data (columns: open, high, low, last_price, volume)
        bollinger_win: Bollinger Bands lookback window
        num_std: Number of standard deviations for bands
        risk_params: Risk management parameters
        starting_cash: Starting cash amount
        
    Returns:
        Tuple of (signals_df, trades_list, metrics_dict)
    """
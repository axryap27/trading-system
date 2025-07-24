"""
Cross-Asset Arbitrage Strategy - Trade spread between correlated assets
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
# Remove the duplicate import statement
from scipy import stats
from sklearn.linear_model import LinearRegression


def run_backtest(hist1: pd.DataFrame, hist2: pd.DataFrame, symbol1: str = "ASSET1", symbol2: str = "ASSET2",
                lookback_window: int = 60, threshold: float = 2.0, risk_params: Dict = None,
                starting_cash: float = 100000.0, transaction_cost: float = 0.001) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Run pairs trading arbitrage backtest between two correlated assets.
    
    Args:
        hist1: DataFrame with OHLCV data for first asset
        hist2: DataFrame with OHLCV data for second asset  
        symbol1: Symbol name for first asset
        symbol2: Symbol name for second asset
        lookback_window: Window for calculating spread statistics
        threshold: Z-score threshold for entry signals
        risk_params: Risk management parameters
        starting_cash: Starting cash amount
        transaction_cost: Transaction cost per trade (as fraction of notional)
        
    Returns:
        Tuple of (signals_df, trades_list, metrics_dict)
    """
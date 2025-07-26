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

     # risk parameters
    if risk_params is None:
        risk_params = {
            "max_position": 1000,    # max position per asset
            "position_pct": 0.1,     # % of cash to risk per trade
            "stop_loss": 3.0,        # Stop loss in Z-score units
            "max_hold_days": 20,     # max days to hold position
            "min_correlation": 0.7   # min correlation to trade
        }
    
    # Align histories and ensure we have matching timestamps
    df = pd.DataFrame({
        "p1": hist1["last_price"],
        "p2": hist2["last_price"]
    }).dropna()
    
    if len(df) < lookback_window * 2:
        raise ValueError(f"Insufficient data: need at least {lookback_window * 2} aligned points")
    
    # Calculate rolling statistics for hedge ratio and spread
    results = []
    
    for i in range(lookback_window, len(df)):
        # Get lookback window data
        window_data = df.iloc[i-lookback_window:i]
        current_data = df.iloc[i]
        
        # Calculate hedge ratio using linear regression
        X = window_data["p2"].values.reshape(-1, 1)
        y = window_data["p1"].values
        
        reg = LinearRegression().fit(X, y)
        hedge_ratio = reg.coef_[0]
        intercept = reg.intercept_
        
        # Calculate current spread
        current_spread = current_data["p1"] - hedge_ratio * current_data["p2"]
        
        # Calculate spread statistics over lookback window
        window_spreads = window_data["p1"] - hedge_ratio * window_data["p2"]
        spread_mean = window_spreads.mean()
        spread_std = window_spreads.std()
        
        # Calculate Z-score
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Calculate correlation
        correlation = window_data["p1"].corr(window_data["p2"])
        
        # Calculate R-squared of regression
        from sklearn.metrics import r2_score
        predicted = reg.predict(X)
        r_squared = r2_score(y, predicted)
        
        results.append({
            "timestamp": df.index[i],
            "p1": current_data["p1"],
            "p2": current_data["p2"],
            "hedge_ratio": hedge_ratio,
            "intercept": intercept,
            "spread": current_spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "z_score": z_score,
            "correlation": correlation,
            "r_squared": r_squared
        })

def execute_trade(symbol: str, side: str, qty: int, price: float, timestamp: datetime,
                 tracker: PositionTracker, trades_list: List[Dict], reason: str,
                 transaction_cost: float, z_score: float, hedge_ratio: float, spread_position: int):
    """
    Execute a single trade and update tracking.
    """
    if qty <= 0:
        return
    
    # Create synthetic execution report
    execution_report = {
        "order_id": str(uuid.uuid4()),
        "symbol": symbol,
        "side": side,
        "filled_qty": qty,
        "price": price,
        "timestamp": timestamp,
        "status": "filled"
    }
    
    # Apply transaction costs
    cost_adjustment = qty * price * transaction_cost
    if side == "buy":
        tracker.cash -= cost_adjustment  # Additional cost for buys
    else:
        tracker.cash -= cost_adjustment  # Cost reduces proceeds from sells
    
    # Update position tracker
    tracker.update(execution_report)
    
    # Add to trades list with additional context
    trades_list.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "price": price,
        "reason": reason,
        "z_score": z_score,
        "hedge_ratio": hedge_ratio,
        "spread_position": spread_position,
        "transaction_cost": cost_adjustment
    })

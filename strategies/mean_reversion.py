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

        
    # Default risk parameters
    if risk_params is None:
        risk_params = {
            "max_position": 1000,      # Maximum position size
            "position_pct": 0.1,       # Percentage of cash to risk per trade
            "hold_periods": 5,         # Maximum periods to hold position
            "stop_loss": 0.05,         # 5% stop loss
            "profit_target": 0.03      # 3% profit target
        }
    
    symbol = "MEAN_REVERT_ASSET"
    
    # Validate inputs
    if len(history) < bollinger_win:
        raise ValueError(f"History length {len(history)} is less than Bollinger window {bollinger_win}")
    
    # Make a copy to avoid modifying original
    df = history.copy()
    
    # Calculate Bollinger Bands
    rolling_mean = df["last_price"].rolling(window=bollinger_win, min_periods=bollinger_win).mean()
    rolling_std = df["last_price"].rolling(window=bollinger_win, min_periods=bollinger_win).std()
    
    df["bb_middle"] = rolling_mean
    df["bb_upper"] = rolling_mean + num_std * rolling_std
    df["bb_lower"] = rolling_mean - num_std * rolling_std
    
    # Calculate additional indicators
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]  # Normalized band width
    df["bb_position"] = (df["last_price"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # % position in bands
    
    # Generate signals
    df["price_prev"] = df["last_price"].shift(1)
    df["bb_upper_prev"] = df["bb_upper"].shift(1) 
    df["bb_lower_prev"] = df["bb_lower"].shift(1)
    df["bb_middle_prev"] = df["bb_middle"].shift(1)
    
    # Entry signals
    # Buy when price crosses below lower band (oversold)
    df["oversold_entry"] = (
        (df["last_price"] < df["bb_lower"]) & 
        (df["price_prev"] >= df["bb_lower_prev"])
    )
    
    # Sell when price crosses above upper band (overbought) 
    df["overbought_entry"] = (
        (df["last_price"] > df["bb_upper"]) &
        (df["price_prev"] <= df["bb_upper_prev"])
    )
    
    # Exit signals  
    # Exit long when price crosses back to middle band
    df["long_exit"] = (
        (df["last_price"] > df["bb_middle"]) &
        (df["price_prev"] <= df["bb_middle_prev"])
    )
    
    # Exit short when price crosses back to middle band
    df["short_exit"] = (
        (df["last_price"] < df["bb_middle"]) &
        (df["price_prev"] >= df["bb_middle_prev"])
    )
    
    # Generate position signals
    # +1 = long, -1 = short, 0 = flat
    df["raw_signal"] = 0
    df.loc[df["oversold_entry"], "raw_signal"] = 1     # Enter long
    df.loc[df["overbought_entry"], "raw_signal"] = -1  # Enter short  
    df.loc[df["long_exit"] | df["short_exit"], "raw_signal"] = 0  # Exit to flat
    
    # Track signal changes
    df["signal_change"] = df["raw_signal"] != 0
    df["prev_signal"] = 0  # Will be updated in backtest loop
    
    # Create signals DataFrame
    signals_df = df[["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
                    "oversold_entry", "overbought_entry", "long_exit", "short_exit",
                    "raw_signal", "signal_change"]].copy()
    
    # Initialize trading infrastructure
    oms = OrderManagementSystem()
    book = LimitOrderBook(symbol)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades_list = []
    
    # Backtest loop
    current_position = 0
    entry_price = 0
    entry_timestamp = None
    hold_periods_count = 0
    
    for timestamp, row in df.iterrows():
        # Skip if no valid Bollinger Bands data
        if pd.isna(row["bb_middle"]) or pd.isna(row["bb_upper"]) or pd.isna(row["bb_lower"]):
            continue
            
        current_price = row["last_price"]
        current_signal = row["raw_signal"]
        
        # Check for forced exit conditions
        forced_exit = False
        exit_reason = ""
        
        if current_position != 0 and entry_price > 0:
            # Check holding period limit
            hold_periods_count += 1
            if hold_periods_count >= risk_params["hold_periods"]:
                forced_exit = True
                exit_reason = "max_hold_period"
            
            # Check stop loss
            elif current_position > 0:  # Long position
                if current_price <= entry_price * (1 - risk_params["stop_loss"]):
                    forced_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= entry_price * (1 + risk_params["profit_target"]):
                    forced_exit = True
                    exit_reason = "profit_target"
                    
            elif current_position < 0:  # Short position
                if current_price >= entry_price * (1 + risk_params["stop_loss"]):
                    forced_exit = True
                    exit_reason = "stop_loss"
                elif current_price <= entry_price * (1 - risk_params["profit_target"]):
                    forced_exit = True
                    exit_reason = "profit_target"
        
        # Determine trade action
        trade_side = None
        trade_qty = 0
        trade_reason = ""
        
        # Handle forced exits
        if forced_exit and current_position != 0:
            if current_position > 0:
                trade_side = "sell"
                trade_qty = abs(current_position)
            else:
                trade_side = "buy"
                trade_qty = abs(current_position)
            trade_reason = exit_reason
            
        # Handle signal-based trades
        elif current_signal != 0:
            if current_signal == 1 and current_position <= 0:
                # Enter/add to long position
                if current_position < 0:
                    # Close short first
                    trade_side = "buy"
                    trade_qty = abs(current_position)
                else:
                    # Enter long
                    max_cash_risk = tracker.cash * risk_params["position_pct"]
                    max_qty_by_cash = int(max_cash_risk / current_price)
                    max_qty_by_position = risk_params["max_position"]
                    trade_qty = min(max_qty_by_cash, max_qty_by_position)
                    trade_side = "buy"
                trade_reason = "oversold_entry"
                
            elif current_signal == -1 and current_position >= 0:
                # Enter/add to short position
                if current_position > 0:
                    # Close long first
                    trade_side = "sell"
                    trade_qty = abs(current_position)
                else:
                    # Enter short
                    max_cash_risk = tracker.cash * risk_params["position_pct"]
                    max_qty_by_cash = int(max_cash_risk / current_price)
                    max_qty_by_position = risk_params["max_position"]
                    trade_qty = min(max_qty_by_cash, max_qty_by_position)
                    trade_side = "sell"
                trade_reason = "overbought_entry"
                
            elif current_signal == 0:
                # Exit signal
                if current_position != 0:
                    if current_position > 0:
                        trade_side = "sell"
                        trade_qty = abs(current_position)
                        trade_reason = "mean_reversion_exit"
                    else:
                        trade_side = "buy"
                        trade_qty = abs(current_position)
                        trade_reason = "mean_reversion_exit"
        
        # Execute trade if needed
        if trade_side and trade_qty > 0:
            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=trade_side,
                quantity=trade_qty,
                type="market",
                timestamp=timestamp
            )
            
            try:
                # Submit to OMS
                ack = oms.new_order(order)
                
                # Create synthetic execution
                execution_report = {
                    "order_id": order.id,
                    "symbol": symbol,
                    "side": trade_side,
                    "filled_qty": trade_qty,
                    "price": current_price,
                    "timestamp": timestamp,
                    "status": "filled"
                }
                
                # Update position tracker
                tracker.update(execution_report)
                
                # Add to trades list
                trades_list.append({
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": trade_side,
                    "quantity": trade_qty,
                    "price": current_price,
                    "signal": current_signal,
                    "reason": trade_reason,
                    "bb_position": row["bb_position"],
                    "bb_width": row["bb_width"],
                    "position_before": current_position
                })
                
                # Update position tracking
                if trade_side == "buy":
                    new_position = current_position + trade_qty
                else:
                    new_position = current_position - trade_qty
                
                # Reset entry tracking for new positions
                if (current_position == 0 and new_position != 0) or (current_position * new_position <= 0):
                    entry_price = current_price
                    entry_timestamp = timestamp
                    hold_periods_count = 0
                elif new_position == 0:
                    entry_price = 0
                    entry_timestamp = None
                    hold_periods_count = 0
                
                current_position = new_position
                
            except Exception as e:
                print(f"Trade execution failed at {timestamp}: {e}")
    
    # Calculate final metrics
    final_price = df["last_price"].iloc[-1]
    current_prices = {symbol: final_price}
    
    # Get performance summary
    performance = tracker.get_performance_metrics(current_prices)
    
    # Calculate strategy-specific metrics
    if len(trades_list) > 0:
        trade_df = pd.DataFrame(trades_list)
        
        # Analyze trade reasons
        trade_reasons = trade_df["reason"].value_counts().to_dict()
        
        # Calculate mean reversion specific metrics
        entry_trades = trade_df[trade_df["reason"].isin(["oversold_entry", "overbought_entry"])]
        exit_trades = trade_df[trade_df["reason"].isin(["mean_reversion_exit", "stop_loss", "profit_target"])]
        
        # Average band position at entry
        avg_entry_bb_position = entry_trades["bb_position"].mean() if len(entry_trades) > 0 else 0.5
        avg_bb_width = entry_trades["bb_width"].mean() if len(entry_trades) > 0 else 0
        
    else:
        trade_reasons = {}
        avg_entry_bb_position = 0.5
        avg_bb_width = 0
    
    # Compile metrics
    metrics_dict = {
        "strategy": "mean_reversion",
        "total_return": performance["total_return"],
        "total_pnl": performance["total_pnl"],
        "max_drawdown": performance["max_drawdown"],
        "sharpe_ratio": performance["sharpe_ratio"],
        "win_rate": performance["win_rate"],
        "total_trades": len(trades_list),
        "final_position": current_position,
        "final_cash": tracker.cash,
        "bollinger_window": bollinger_win,
        "num_std": num_std,
        "starting_cash": starting_cash,
        "trade_reasons": trade_reasons,
        "avg_entry_bb_position": avg_entry_bb_position,
        "avg_bb_width": avg_bb_width
    }
    
    return signals_df, trades_list, metrics_dict
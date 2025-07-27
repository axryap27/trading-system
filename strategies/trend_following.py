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
    # Backtest loop - process each signal change
    current_position = 0  # Track our current position
    
    for timestamp, row in df.iterrows():
        if not row["signal_change"] or pd.isna(row["ma_short"]) or pd.isna(row["ma_long"]):
            continue
            
        current_signal = row["signal"]
        prev_signal = row["prev_signal"]
        current_price = row["last_price"]
        
        # Determine trade action
        trade_side = None
        trade_qty = 0
        
        if current_signal == 1 and prev_signal == 0:
            # Enter long position
            trade_side = "buy"
            # Calculate position size based on risk parameters
            max_cash_risk = starting_cash * risk_params["position_pct"]
            max_qty_by_cash = int(max_cash_risk / current_price)
            max_qty_by_position = risk_params["max_position"]
            trade_qty = min(max_qty_by_cash, max_qty_by_position)
            
        elif current_signal == 0 and prev_signal == 1:
            # Exit long position
            if current_position > 0:
                trade_side = "sell"
                trade_qty = current_position
        
        # Execute trade if needed
        if trade_side and trade_qty > 0:
            # Create order
            order = Order(
                id=str(uuid.uuid4()),
                symbol=symbol,
                side=trade_side,
                quantity=trade_qty,
                type="market",  # Use market orders for simplicity
                timestamp=timestamp
            )
            
            # Submit to OMS
            try:
                ack = oms.new_order(order)
                
                # Create synthetic execution (since we don't have real matching)
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
                    "ma_short": row["ma_short"],
                    "ma_long": row["ma_long"]
                })
                
                # Update current position
                if trade_side == "buy":
                    current_position += trade_qty
                else:
                    current_position -= trade_qty
                    
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
        
        # Calculate trade-level returns
        buy_trades = trade_df[trade_df["side"] == "buy"]
        sell_trades = trade_df[trade_df["side"] == "sell"]
        
        # Match buy/sell pairs for trade analysis
        trade_returns = []
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            for _, sell_trade in sell_trades.iterrows():
                # Find the most recent buy trade before this sell
                prior_buys = buy_trades[buy_trades["timestamp"] < sell_trade["timestamp"]]
                if len(prior_buys) > 0:
                    buy_trade = prior_buys.iloc[-1]
                    trade_return = (sell_trade["price"] - buy_trade["price"]) / buy_trade["price"]
                    trade_returns.append(trade_return)
        
        # Win rate calculation
        if trade_returns:
            winning_trades = [r for r in trade_returns if r > 0]
            win_rate = len(winning_trades) / len(trade_returns)
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    
    # Compile metrics
    metrics_dict = {
        "strategy": "trend_following",
        "total_return": performance["total_return"],
        "total_pnl": performance["total_pnl"],
        "max_drawdown": performance["max_drawdown"],
        "sharpe_ratio": performance["sharpe_ratio"],
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_trades": len(trades_list),
        "final_position": current_position,
        "final_cash": tracker.cash,
        "short_window": short_win,
        "long_window": long_win,
        "starting_cash": starting_cash
    }
    
    return signals_df, trades_list, metrics_dict
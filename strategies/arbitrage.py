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
        
        # Calculate hedge ratio using np.polyfit as specified in homework
        X = window_data["p2"].values
        y = window_data["p1"].values
        
        # Use np.polyfit for linear regression (degree 1)
        coeffs = np.polyfit(X, y, 1)
        hedge_ratio = coeffs[0]  # slope (beta)
        intercept = coeffs[1]    # intercept
        
        # Calculate R-squared manually
        y_pred = hedge_ratio * X + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Calculate current spread
        current_spread = current_data["p1"] - hedge_ratio * current_data["p2"]
        
        # Calculate spread statistics over lookback window
        window_spreads = window_data["p1"] - hedge_ratio * window_data["p2"]
        spread_mean = window_spreads.mean()
        spread_std = window_spreads.std()
        
        # Calculate Z-score
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
        
        # Calculate correlation manually
        correlation = np.corrcoef(window_data["p1"], window_data["p2"])[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # R-squared already calculated in regression function
        
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
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(results).set_index("timestamp")
    
    # Generate trading signals
    signals_df["entry_long_spread"] = (
        (signals_df["z_score"] < -threshold) &  # Spread below lower threshold
        (signals_df["correlation"] > risk_params["min_correlation"]) &  # Strong correlation
        (signals_df["r_squared"] > 0.5)  # Good fit
    )
    
    signals_df["entry_short_spread"] = (
        (signals_df["z_score"] > threshold) &   # Spread above upper threshold  
        (signals_df["correlation"] > risk_params["min_correlation"]) &
        (signals_df["r_squared"] > 0.5)
    )
    
    signals_df["exit_spread"] = (
        (abs(signals_df["z_score"]) < 0.5) |  # Spread returned to normal
        (signals_df["correlation"] < risk_params["min_correlation"])  # Correlation breakdown
    )
    
    # Generate position signals
    # +1 = long spread (buy asset1, sell asset2)
    # -1 = short spread (sell asset1, buy asset2)
    # 0 = flat
    signals_df["raw_signal"] = 0
    signals_df.loc[signals_df["entry_long_spread"], "raw_signal"] = 1
    signals_df.loc[signals_df["entry_short_spread"], "raw_signal"] = -1
    signals_df.loc[signals_df["exit_spread"], "raw_signal"] = 0
    
    # Track signal changes
    signals_df["signal_change"] = signals_df["raw_signal"] != signals_df["raw_signal"].shift(1)
    signals_df["prev_signal"] = signals_df["raw_signal"].shift(1).fillna(0)
    
    # Initialize trading infrastructure
    oms1 = OrderManagementSystem()
    oms2 = OrderManagementSystem()
    book1 = LimitOrderBook(symbol1)
    book2 = LimitOrderBook(symbol2)
    tracker = PositionTracker(starting_cash=starting_cash)
    trades_list = []
    
    # Backtest loop  
    current_spread_position = 0  # +1 = long spread, -1 = short spread, 0 = flat
    entry_z_score = 0
    entry_timestamp = None
    days_in_position = 0
    
    for timestamp, row in signals_df.iterrows():
        current_signal = row["raw_signal"]
        z_score = row["z_score"]
        hedge_ratio = row["hedge_ratio"]
        p1_price = row["p1"]
        p2_price = row["p2"]
        
        # Check for forced exit conditions
        forced_exit = False
        exit_reason = ""
        
        if current_spread_position != 0:
            days_in_position += 1
            
            # Check maximum hold period
            if days_in_position >= risk_params["max_hold_days"]:
                forced_exit = True
                exit_reason = "max_hold_days"
            
            # Check stop loss (Z-score moved further against us)
            elif current_spread_position == 1 and z_score < -risk_params["stop_loss"]:
                forced_exit = True
                exit_reason = "stop_loss"
            elif current_spread_position == -1 and z_score > risk_params["stop_loss"]:
                forced_exit = True
                exit_reason = "stop_loss"
        
        # Determine trade action
        trade_signal = current_signal
        trade_reason = ""
        
        if forced_exit:
            trade_signal = 0
            trade_reason = exit_reason
        elif row["signal_change"]:
            if current_signal == 1:
                trade_reason = "long_spread_entry"
            elif current_signal == -1:
                trade_reason = "short_spread_entry"
            elif current_signal == 0:
                trade_reason = "spread_exit"
        else:
            continue  # No trade
        
        # Execute spread trade if signal changed
        if current_spread_position != trade_signal:
            # Calculate position sizes
            available_cash = tracker.cash * risk_params["position_pct"]
            
            # For pairs trading, we want equal dollar amounts in each leg
            # Adjust for hedge ratio
            total_notional = available_cash / (1 + abs(hedge_ratio))
            
            qty1 = min(int(total_notional / p1_price), risk_params["max_position"])
            qty2 = min(int(total_notional * abs(hedge_ratio) / p2_price), risk_params["max_position"])
            
            if qty1 > 0 and qty2 > 0:
                # Close existing position first if needed
                if current_spread_position != 0:
                    # Close current position
                    close_trades = []
                    
                    if current_spread_position == 1:  # Close long spread
                        close_trades.extend([
                            (symbol1, "sell", qty1, p1_price),
                            (symbol2, "buy", qty2, p2_price)
                        ])
                    else:  # Close short spread
                        close_trades.extend([
                            (symbol1, "buy", qty1, p1_price),
                            (symbol2, "sell", qty2, p2_price)
                        ])
                    
                    # Execute closing trades
                    for symbol, side, qty, price in close_trades:
                        execute_trade(symbol, side, qty, price, timestamp, tracker, 
                                    trades_list, trade_reason + "_close", transaction_cost,
                                    z_score, hedge_ratio, current_spread_position)
                
                # Open new position if not going flat
                if trade_signal != 0:
                    open_trades = []
                    
                    if trade_signal == 1:  # Long spread: buy asset1, sell asset2
                        open_trades.extend([
                            (symbol1, "buy", qty1, p1_price),
                            (symbol2, "sell", qty2, p2_price)
                        ])
                    else:  # Short spread: sell asset1, buy asset2
                        open_trades.extend([
                            (symbol1, "sell", qty1, p1_price),
                            (symbol2, "buy", qty2, p2_price)
                        ])
                    
                    # Execute opening trades
                    for symbol, side, qty, price in open_trades:
                        execute_trade(symbol, side, qty, price, timestamp, tracker,
                                    trades_list, trade_reason, transaction_cost,
                                    z_score, hedge_ratio, trade_signal)
                    
                    # Update position tracking
                    entry_z_score = z_score
                    entry_timestamp = timestamp
                    days_in_position = 0
                
                # Update current position
                current_spread_position = trade_signal
    
    # Calculate final metrics
    final_p1 = signals_df["p1"].iloc[-1]
    final_p2 = signals_df["p2"].iloc[-1]
    current_prices = {symbol1: final_p1, symbol2: final_p2}
    
    # Get performance summary
    performance = tracker.get_performance_metrics(current_prices)
    
    # Calculate strategy-specific metrics
    if len(trades_list) > 0:
        trade_df = pd.DataFrame(trades_list)
        
        # Analyze trade reasons
        trade_reasons = trade_df["reason"].value_counts().to_dict()
        
        # Calculate pairs-specific metrics
        avg_entry_z_score = abs(trade_df[trade_df["reason"].str.contains("entry")]["z_score"]).mean()
        avg_hedge_ratio = trade_df["hedge_ratio"].mean()
        
        # Count complete round trips
        entry_trades = trade_df[trade_df["reason"].str.contains("entry")]
        exit_trades = trade_df[trade_df["reason"].str.contains("exit|close")]
        round_trips = min(len(entry_trades), len(exit_trades)) // 2  # Each spread trade has 2 legs
        
    else:
        trade_reasons = {}
        avg_entry_z_score = 0
        avg_hedge_ratio = 1
        round_trips = 0
    
    # Compile metrics
    metrics_dict = {
        "strategy": "arbitrage",
        "total_return": performance["total_return"],
        "total_pnl": performance["total_pnl"],
        "max_drawdown": performance["max_drawdown"],
        "sharpe_ratio": performance["sharpe_ratio"],
        "win_rate": performance["win_rate"],
        "total_trades": len(trades_list),
        "round_trips": round_trips,
        "final_spread_position": current_spread_position,
        "final_cash": tracker.cash,
        "lookback_window": lookback_window,
        "threshold": threshold,
        "starting_cash": starting_cash,
        "transaction_cost": transaction_cost,
        "trade_reasons": trade_reasons,
        "avg_entry_z_score": avg_entry_z_score,
        "avg_hedge_ratio": avg_hedge_ratio,
        "final_correlation": signals_df["correlation"].iloc[-1],
        "avg_correlation": signals_df["correlation"].mean()
    }
    
    return signals_df, trades_list, metrics_dict

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


def analyze_arbitrage_signals(signals_df: pd.DataFrame) -> Dict:
    """
    Analyze arbitrage signals for quality metrics.
    
    Args:
        signals_df: DataFrame with signal data
        
    Returns:
        Dictionary with signal analysis metrics
    """
    total_bars = len(signals_df)
    
    # Signal counts
    long_spread_entries = signals_df["entry_long_spread"].sum()
    short_spread_entries = signals_df["entry_short_spread"].sum()
    spread_exits = signals_df["exit_spread"].sum()
    
    # Spread statistics
    avg_z_score = abs(signals_df["z_score"]).mean()
    max_z_score = abs(signals_df["z_score"]).max()
    z_score_std = signals_df["z_score"].std()
    
    # Correlation statistics
    avg_correlation = signals_df["correlation"].mean()
    min_correlation = signals_df["correlation"].min()
    correlation_std = signals_df["correlation"].std()
    
    # Hedge ratio statistics
    avg_hedge_ratio = signals_df["hedge_ratio"].mean()
    hedge_ratio_std = signals_df["hedge_ratio"].std()
    
    # R-squared statistics
    avg_r_squared = signals_df["r_squared"].mean()
    min_r_squared = signals_df["r_squared"].min()
    
    return {
        "total_bars": total_bars,
        "long_spread_entries": long_spread_entries,
        "short_spread_entries": short_spread_entries,
        "spread_exits": spread_exits,
        "signal_frequency": (long_spread_entries + short_spread_entries) / total_bars if total_bars > 0 else 0,
        "avg_z_score": avg_z_score,
        "max_z_score": max_z_score,
        "z_score_volatility": z_score_std,
        "avg_correlation": avg_correlation,
        "min_correlation": min_correlation,
        "correlation_stability": 1 - correlation_std,  # Higher is more stable
        "avg_hedge_ratio": avg_hedge_ratio,
        "hedge_ratio_stability": 1 - hedge_ratio_std,
        "avg_r_squared": avg_r_squared,
        "min_r_squared": min_r_squared
    }


def optimize_parameters(hist1: pd.DataFrame, hist2: pd.DataFrame,
                       symbol1: str = "ASSET1", symbol2: str = "ASSET2",
                       lookback_range: Tuple[int, int] = (30, 120),
                       threshold_range: Tuple[float, float] = (1.5, 3.0),
                       lookback_step: int = 10,
                       threshold_step: float = 0.25) -> pd.DataFrame:
    """
    Optimize arbitrage parameters using grid search.
    
    Args:
        hist1, hist2: Price history DataFrames
        symbol1, symbol2: Asset symbols
        lookback_range: Range for lookback window (min, max)
        threshold_range: Range for Z-score threshold (min, max)
        lookback_step: Step size for lookback search
        threshold_step: Step size for threshold search
        
    Returns:
        DataFrame with optimization results
    """
    results = []
    
    threshold_values = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    for lookback in range(lookback_range[0], lookback_range[1] + 1, lookback_step):
        for threshold in threshold_values:
            try:
                signals_df, trades_list, metrics = run_backtest(
                    hist1, hist2, symbol1, symbol2,
                    lookback_window=lookback, threshold=threshold
                )
                
                results.append({
                    "lookback_window": lookback,
                    "threshold": round(threshold, 2),
                    "total_return": metrics["total_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "total_trades": metrics["total_trades"],
                    "round_trips": metrics["round_trips"],
                    "avg_correlation": metrics["avg_correlation"]
                })
                
            except Exception as e:
                print(f"Failed for lookback={lookback}, threshold={threshold:.2f}: {e}")
                continue
    
    return pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)


if __name__ == "__main__":
    # Example usage and testing
    print("Arbitrage Strategy Test")
    print("=" * 25)
    
    # Create sample correlated assets
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # Generate correlated price series
    # Asset 1: base price with trend
    returns1 = np.random.normal(0.0005, 0.02, 252)  # Daily returns
    prices1 = 100 * np.exp(np.cumsum(returns1))
    
    # Asset 2: correlated with Asset 1 but with some noise
    correlation = 0.8
    returns2_corr = correlation * returns1 + np.sqrt(1 - correlation**2) * np.random.normal(0.0005, 0.02, 252)
    prices2 = 150 * np.exp(np.cumsum(returns2_corr))  # Different base price
    
    # Create DataFrames
    hist1 = pd.DataFrame({
        'open': prices1 * 0.999,
        'high': prices1 * 1.01,
        'low': prices1 * 0.99,
        'last_price': prices1,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    hist2 = pd.DataFrame({
        'open': prices2 * 0.999,
        'high': prices2 * 1.01,
        'low': prices2 * 0.99,
        'last_price': prices2,
        'volume': np.random.randint(1000, 10000, 252)
    }, index=dates)
    
    # Run backtest
    signals, trades, metrics = run_backtest(
        hist1, hist2, "STOCK_A", "STOCK_B",
        lookback_window=60,
        threshold=2.0,
        risk_params={
            "max_position": 100,
            "position_pct": 0.2,
            "stop_loss": 3.0,
            "max_hold_days": 15,
            "min_correlation": 0.7
        },
        transaction_cost=0.001
    )
    
    print(f"Backtest Results:")
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'rate' in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTotal trades executed: {len(trades)}")
    if trades:
        print("Sample trades:")
        for trade in trades[:6]:
            print(f"  {trade['timestamp'].date()}: {trade['side']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f} ({trade['reason']}, Z={trade['z_score']:.2f})")
    
    # Analyze signals
    signal_analysis = analyze_arbitrage_signals(signals)
    print(f"\nSignal Analysis:")
    for key, value in signal_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Show correlation between assets
    correlation = hist1['last_price'].corr(hist2['last_price'])
    print(f"\nAsset correlation: {correlation:.3f}")


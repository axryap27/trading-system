#!/usr/bin/env python3
"""
Position Tracker - tracks positions and cash, records trades, and computes P&L
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime


class PositionTracker:
    """
    Tracks positions and cash, records trades, and computes P&L.
    Acts as the P&L engine for the trading system.
    """
    
    def __init__(self, starting_cash: float = 0.0):
        """
        Initialize position tracker with starting cash.
        
        Args:
            starting_cash: Initial cash balance
        """
        # Net position per symbol (symbol -> shares/contracts)
        self.positions: Dict[str, int] = {}
        
        # Cash balance
        self.cash: float = starting_cash
        self.starting_cash: float = starting_cash
        
        # List of trade records (one dict per fill)
        self.blotter: List[Dict] = []
        
        # Track statistics
        self.total_trades = 0
        self.total_volume = 0
        self.total_commissions = 0.0
    
    def update(self, report: Dict) -> None:
        """
        Process one execution report with keys:
        - order_id
        - symbol  
        - side ('buy' or 'sell')
        - filled_qty (int)
        - price (float)
        - timestamp (datetime)
        
        Updates positions, cash, and appends to the blotter list.
        
        Args:
            report: Execution report dictionary from matching engine
        """
        # Extract key information
        symbol = report["symbol"]
        qty = report["filled_qty"]
        price = report["price"]
        side = report["side"]
        timestamp = report["timestamp"]
        order_id = report.get("order_id", "UNKNOWN")
        
        # Validate inputs
        if qty <= 0:
            raise ValueError(f"Fill quantity must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"Fill price must be positive, got {price}")
        
        # 1) Update net position
        # A 'buy' increases your holding, 'sell' decreases it
        delta = qty if side == "buy" else -qty
        self.positions[symbol] = self.positions.get(symbol, 0) + delta
        
        # 2) Update cash
        # Cash flows negative for buys (cash out), positive for sells (cash in)
        cash_flow = -qty * price if side == "buy" else qty * price
        self.cash += cash_flow
        
        # 3) Update statistics
        self.total_trades += 1
        self.total_volume += qty
        
        # 4) Record in blotter
        # We'll compute realized P&L later in summary
        self.blotter.append({
            "timestamp": timestamp,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "price": price,
            "cash_flow": cash_flow,
            "position_after": self.positions[symbol]
        })
    
    def get_blotter(self) -> pd.DataFrame:
        """
        Return a DataFrame of all fills with columns:
        timestamp, order_id, symbol, side, quantity, price, cash_flow, position_after
        
        Returns:
            DataFrame of all trade records
        """
        if not self.blotter:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'timestamp', 'order_id', 'symbol', 'side', 
                'quantity', 'price', 'cash_flow', 'position_after'
            ])
        
        df = pd.DataFrame(self.blotter)
        
        # Set timestamp as index for time series analysis
        df = df.set_index('timestamp').sort_index()
        
        return df
    
    def get_pnl_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """
        Returns a dict with:
        - 'realized_pnl': float
        - 'unrealized_pnl': float (0 if no current_prices given)
        - 'total_pnl': sum of realized + unrealized
        - 'current_cash': self.cash
        - 'positions': copy of self.positions dict
        - 'starting_cash': original cash amount
        - 'total_return': (total_pnl / starting_cash) if starting_cash > 0
        
        Args:
            current_prices: Optional dict mapping symbols to current market prices
            
        Returns:
            Dictionary with P&L summary
        """
        # Realized P&L = change in cash from starting cash
        realized_pnl = self.cash - self.starting_cash
        
        # Unrealized P&L = current market value of positions
        unrealized_pnl = 0.0
        position_values = {}
        
        if current_prices:
            for symbol, position in self.positions.items():
                if symbol in current_prices and position != 0:
                    market_value = position * current_prices[symbol]
                    position_values[symbol] = market_value
                    unrealized_pnl += market_value
        
        # Total P&L
        total_pnl = realized_pnl + unrealized_pnl
        
        # Total return calculation
        total_return = 0.0
        if self.starting_cash != 0:
            total_return = total_pnl / abs(self.starting_cash)
        
        return {
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "total_pnl": float(total_pnl),
            "current_cash": float(self.cash),
            "starting_cash": float(self.starting_cash),
            "positions": dict(self.positions),
            "position_values": position_values,
            "total_return": float(total_return),
            "total_trades": self.total_trades,
            "total_volume": self.total_volume
        }
    
    def get_equity_curve(self, current_prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate equity curve showing portfolio value over time.
        
        Args:
            current_prices: Current market prices for unrealized P&L
            
        Returns:
            DataFrame with timestamp index and equity values
        """
        if not self.blotter:
            return pd.DataFrame(columns=['equity', 'cash', 'unrealized_pnl'])
        
        blotter_df = self.get_blotter()
        
        # Calculate running cash balance
        equity_data = []
        running_positions = {}
        
        for timestamp, row in blotter_df.iterrows():
            # Update running position
            symbol = row['symbol']
            side = row['side']
            qty = row['quantity']
            
            delta = qty if side == "buy" else -qty
            running_positions[symbol] = running_positions.get(symbol, 0) + delta
            
            # Calculate unrealized P&L at this point
            unrealized = 0.0
            if current_prices:
                for sym, pos in running_positions.items():
                    if sym in current_prices and pos != 0:
                        unrealized += pos * current_prices[sym]
            
            # Calculate equity = starting_cash + cumulative cash flows + unrealized
            cash_at_time = self.starting_cash + blotter_df.loc[:timestamp, 'cash_flow'].sum()
            equity = cash_at_time + unrealized
            
            equity_data.append({
                'timestamp': timestamp,
                'equity': equity,
                'cash': cash_at_time,
                'unrealized_pnl': unrealized
            })
        
        if not equity_data:
            return pd.DataFrame(columns=['equity', 'cash', 'unrealized_pnl'])
        
        equity_df = pd.DataFrame(equity_data)
        equity_df = equity_df.set_index('timestamp').sort_index()
        
        return equity_df
    
    def get_returns(self, freq: str = 'D') -> pd.Series:
        """
        Calculate returns from the equity curve.
        
        Args:
            freq: Frequency for return calculation ('D' for daily, 'H' for hourly, etc.)
            
        Returns:
            Series of returns
        """
        equity_curve = self.get_equity_curve()
        
        if equity_curve.empty:
            return pd.Series(dtype=float)
        
        # Resample to desired frequency and calculate returns
        equity_resampled = equity_curve['equity'].resample(freq).last().fillna(method='ffill')
        returns = equity_resampled.pct_change().fillna(0)
        
        return returns
    
    def get_performance_metrics(self, current_prices: Optional[Dict[str, float]] = None, 
                              risk_free_rate: float = 0.02) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            current_prices: Current market prices
            risk_free_rate: Annual risk-free rate for Sharpe ratio
            
        Returns:
            Dictionary of performance metrics
        """
        pnl_summary = self.get_pnl_summary(current_prices)
        equity_curve = self.get_equity_curve(current_prices)
        
        metrics = {
            "total_pnl": pnl_summary["total_pnl"],
            "total_return": pnl_summary["total_return"],
            "realized_pnl": pnl_summary["realized_pnl"],
            "unrealized_pnl": pnl_summary["unrealized_pnl"],
            "total_trades": self.total_trades,
            "total_volume": self.total_volume
        }
        
        if not equity_curve.empty and len(equity_curve) > 1:
            # Calculate drawdown
            running_max = equity_curve['equity'].expanding().max()
            drawdown = (equity_curve['equity'] - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calculate returns for Sharpe ratio
            returns = self.get_returns('D')  # Daily returns
            
            if len(returns) > 1 and returns.std() > 0:
                # Annualized Sharpe ratio
                excess_return = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
                sharpe_ratio = (excess_return / returns.std()) * (252 ** 0.5)  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Win rate and average win/loss
            winning_days = returns[returns > 0]
            losing_days = returns[returns < 0]
            
            win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
            avg_win = winning_days.mean() if len(winning_days) > 0 else 0
            avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
            
            metrics.update({
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": float(sharpe_ratio),
                "win_rate": float(win_rate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "profit_factor": float(-avg_win / avg_loss) if avg_loss < 0 else float('inf')
            })
        else:
            # Default values for insufficient data
            metrics.update({
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0
            })
        
        return metrics
    
    def get_position_summary(self) -> pd.DataFrame:
        """
        Get summary of current positions.
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame(columns=['symbol', 'quantity', 'market_value'])
        
        position_data = []
        for symbol, quantity in self.positions.items():
            if quantity != 0:  # Only show non-zero positions
                position_data.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'market_value': 0.0  # Will be updated if current prices provided
                })
        
        return pd.DataFrame(position_data)
    
    def reset(self, new_starting_cash: Optional[float] = None):
        """
        Reset the position tracker to initial state.
        
        Args:
            new_starting_cash: New starting cash amount, uses current if None
        """
        if new_starting_cash is not None:
            self.starting_cash = new_starting_cash
            self.cash = new_starting_cash
        else:
            self.cash = self.starting_cash
        
        self.positions.clear()
        self.blotter.clear()
        self.total_trades = 0
        self.total_volume = 0
        self.total_commissions = 0.0
    
    def __str__(self):
        """String representation showing key stats."""
        pnl = self.cash - self.starting_cash
        return (f"PositionTracker(cash=${self.cash:,.2f}, "
                f"pnl=${pnl:,.2f}, positions={len([p for p in self.positions.values() if p != 0])}, "
                f"trades={self.total_trades})")
    
    def __repr__(self):
        """Detailed representation."""
        return (f"PositionTracker(starting_cash={self.starting_cash}, "
                f"current_cash={self.cash}, positions={dict(self.positions)}, "
                f"trades={len(self.blotter)})")
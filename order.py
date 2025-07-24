#!/usr/bin/env python3
"""
Order data class - simple container for order details
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Order:
    """
    Represents a single trade instruction.
    """
    id: str                    # unique identifier (e.g. UUID or string)
    symbol: str               # ticker or asset code (e.g. "AAPL", "EURUSD=X")
    side: str                 # "buy" or "sell"
    quantity: int             # must be > 0
    type: str                 # "market", "limit", or "stop"
    price: Optional[float] = None        # limit/stop price, None for market orders
    timestamp: Optional[datetime] = None # when the order was created
    
    def __post_init__(self):
        """Validate order fields after initialization"""
        if self.side not in ("buy", "sell"):
            raise ValueError("Side must be 'buy' or 'sell'")
        
        if self.quantity <= 0:
            raise ValueError("Quantity must be > 0")
            
        if self.type not in ("market", "limit", "stop"):
            raise ValueError("Type must be 'market', 'limit', or 'stop'")
            
        if self.type in ("limit", "stop") and self.price is None:
            raise ValueError("Limit/stop orders require a price")
    
    def __str__(self):
        """String representation for debugging"""
        price_str = f"@${self.price:.2f}" if self.price else "@market"
        return f"Order({self.id}: {self.side.upper()} {self.quantity} {self.symbol} {self.type} {price_str})"
    
    def __repr__(self):
        """Detailed representation"""
        return (f"Order(id='{self.id}', symbol='{self.symbol}', side='{self.side}', "
                f"quantity={self.quantity}, type='{self.type}', price={self.price}, "
                f"timestamp={self.timestamp})")
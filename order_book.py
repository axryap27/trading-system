#!/usr/bin/env python3
"""
LimitOrderBook - A simple price-time priority limit order book
"""

from datetime import datetime
from typing import List, Dict, Optional
from order import Order
import copy


class LimitOrderBook:
    """
    A simple price-time priority limit order book.
    Maintains two sides of a book (bids and asks) and matches incoming orders.
    """
    
    def __init__(self, symbol: str):
        """
        Initialize order book for a specific symbol.
        
        Args:
            symbol: The trading symbol this book handles
        """
        self.symbol = symbol
        
        # bids: list of resting buy orders, highest price first
        self.bids: List[Order] = []
        
        # asks: list of resting sell orders, lowest price first  
        self.asks: List[Order] = []
        
        # Track total volume and number of trades
        self.total_volume = 0
        self.trade_count = 0
    
    def add_order(self, order: Order) -> List[Dict]:
        """
        Handle a new incoming order (market, limit, or stop).
        Returns a list of execution report dicts.
        
        Args:
            order: Order to process
            
        Returns:
            List of execution reports (dicts) for each fill
        """
        if order.symbol != self.symbol:
            raise ValueError(f"Order symbol {order.symbol} doesn't match book symbol {self.symbol}")
        
        # Make a copy to avoid modifying the original
        working_order = copy.deepcopy(order)
        reports = []
        
        if working_order.type == "market":
            reports += self._execute_market(working_order)
        elif working_order.type == "limit":
            # Try to match immediately
            reports += self._match_limit(working_order)
            # If there's leftover quantity, add to book  
            if working_order.quantity > 0:
                self._insert_resting(working_order)
        else:  # stop orders
            # For now, treat stop orders as plain limit orders
            # In practice, stop logic would be handled externally
            reports += self._match_limit(working_order)
            if working_order.quantity > 0:
                self._insert_resting(working_order)
        
        return reports
    
    def _match_limit(self, order: Order) -> List[Dict]:
        """
        Match a limit order against the book.
        Fill as much as possible at prices satisfying the limit.
        
        Args:
            order: Limit order to match
            
        Returns:
            List of execution reports
        """
        reports = []
        
        # Choose opposite side
        opposite = self.asks if order.side == "buy" else self.bids
        
        # Continue matching while we still have quantity
        # and there is a resting order that satisfies the price
        while order.quantity > 0 and opposite:
            best = opposite[0]
            
            # Check price condition
            # Buy order matches if best ask <= order.price
            if order.side == "buy" and best.price > order.price:
                break
            # Sell order matches if best bid >= order.price    
            if order.side == "sell" and best.price < order.price:
                break
            
            # A fill occurs: trade quantity = min(incoming, resting)
            fill_qty = min(order.quantity, best.quantity)
            trade_price = best.price  # Price improvement goes to aggressor
            timestamp = datetime.utcnow()
            
            # Build execution report for the incoming (aggressor) order
            reports.append({
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "filled_qty": fill_qty,
                "price": trade_price,
                "timestamp": timestamp,
                "status": "filled" if fill_qty == order.quantity else "partial_fill"
            })
            
            # Also build report for the resting order
            reports.append({
                "order_id": best.id,
                "symbol": best.symbol,
                "side": best.side,
                "filled_qty": fill_qty,
                "price": trade_price,
                "timestamp": timestamp,
                "status": "filled" if fill_qty == best.quantity else "partial_fill"
            })
            
            # Update statistics
            self.total_volume += fill_qty
            self.trade_count += 1
            
            # Decrement quantities
            order.quantity -= fill_qty
            best.quantity -= fill_qty
            
            # Remove resting order if fully filled
            if best.quantity == 0:
                opposite.pop(0)
        
        return reports
    
    def _execute_market(self, order: Order) -> List[Dict]:
        """
        Fill a market order against the full depth of the book.
        
        Args:
            order: Market order to execute
            
        Returns:
            List of execution reports
        """
        reports = []
        
        # Opposite side = asks if buy; bids if sell
        opposite = self.asks if order.side == "buy" else self.bids
        
        while order.quantity > 0 and opposite:
            best = opposite[0]
            fill_qty = min(order.quantity, best.quantity)
            trade_price = best.price
            timestamp = datetime.utcnow()
            
            # Aggressor (market order) report
            reports.append({
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "filled_qty": fill_qty,
                "price": trade_price,
                "timestamp": timestamp,
                "status": "filled" if fill_qty == order.quantity else "partial_fill"
            })
            
            # Resting order report
            reports.append({
                "order_id": best.id,
                "symbol": best.symbol,
                "side": best.side,
                "filled_qty": fill_qty,
                "price": trade_price,
                "timestamp": timestamp,
                "status": "filled" if fill_qty == best.quantity else "partial_fill"
            })
            
            # Update statistics
            self.total_volume += fill_qty
            self.trade_count += 1
            
            # Decrement quantities
            order.quantity -= fill_qty
            best.quantity -= fill_qty
            
            # Remove resting order if fully filled
            if best.quantity == 0:
                opposite.pop(0)
        
        return reports
    
    def _insert_resting(self, order: Order):
        """
        Place a remainder limit order into bids or asks,
        maintaining sorted order (price-time priority).
        
        Args:
            order: Order to insert into the book
        """
        book = self.bids if order.side == "buy" else self.asks
        
        # Find insertion index to maintain price-time priority
        idx = 0
        while idx < len(book):
            # For bids: descending price (higher prices first)
            # For asks: ascending price (lower prices first)
            if order.side == "buy":
                # Insert after orders with better (higher) prices
                if book[idx].price > order.price:
                    idx += 1
                    continue
                # For same price, maintain time priority (newer orders go after)
                elif book[idx].price == order.price:
                    idx += 1
                    continue
            else:  # sell order
                # Insert after orders with better (lower) prices
                if book[idx].price < order.price:
                    idx += 1
                    continue
                # For same price, maintain time priority
                elif book[idx].price == order.price:
                    idx += 1
                    continue
            break
        
        # Insert at the found position
        book.insert(idx, order)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order in the book.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if order was found and canceled, False otherwise
        """
        # Search in bids
        for i, order in enumerate(self.bids):
            if order.id == order_id:
                self.bids.pop(i)
                return True
        
        # Search in asks
        for i, order in enumerate(self.asks):
            if order.id == order_id:
                self.asks.pop(i)
                return True
        
        return False
    
    def get_best_bid(self) -> Optional[float]:
        """Get the best (highest) bid price."""
        return self.bids[0].price if self.bids else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get the best (lowest) ask price."""
        return self.asks[0].price if self.asks else None
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price (average of best bid and ask)."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        return None
    
    def get_book_depth(self, levels: int = 5) -> Dict:
        """
        Get book depth showing top N levels.
        
        Args:
            levels: Number of price levels to show
            
        Returns:
            Dict with 'bids' and 'asks' lists
        """
        def format_level(orders_at_level):
            if not orders_at_level:
                return None
            price = orders_at_level[0].price
            total_qty = sum(order.quantity for order in orders_at_level)
            return {"price": price, "quantity": total_qty, "orders": len(orders_at_level)}
        
        # Group bids by price level
        bid_levels = []
        current_price = None
        current_group = []
        
        for order in self.bids[:levels*10]:  # Get more orders to ensure we have enough levels
            if current_price is None or order.price == current_price:
                current_group.append(order)
                current_price = order.price
            else:
                if len(bid_levels) < levels:
                    bid_levels.append(format_level(current_group))
                current_group = [order]
                current_price = order.price
                
        if current_group and len(bid_levels) < levels:
            bid_levels.append(format_level(current_group))
        
        # Group asks by price level
        ask_levels = []
        current_price = None
        current_group = []
        
        for order in self.asks[:levels*10]:
            if current_price is None or order.price == current_price:
                current_group.append(order)
                current_price = order.price
            else:
                if len(ask_levels) < levels:
                    ask_levels.append(format_level(current_group))
                current_group = [order]
                current_price = order.price
                
        if current_group and len(ask_levels) < levels:
            ask_levels.append(format_level(current_group))
        
        return {
            "bids": bid_levels,
            "asks": ask_levels,
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price()
        }
    
    def get_statistics(self) -> Dict:
        """Get book statistics."""
        return {
            "symbol": self.symbol,
            "total_volume": self.total_volume,
            "trade_count": self.trade_count,
            "bid_orders": len(self.bids),
            "ask_orders": len(self.asks),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price()
        }
    
    def __str__(self):
        """String representation of the order book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        
        return (f"OrderBook({self.symbol}: "
                f"bid={best_bid:.2f if best_bid else 'None'}, "
                f"ask={best_ask:.2f if best_ask else 'None'}, "
                f"spread={spread:.4f if spread else 'None'})")
    
    def __repr__(self):
        """Detailed representation."""
        return (f"LimitOrderBook(symbol='{self.symbol}', "
                f"bids={len(self.bids)}, asks={len(self.asks)}, "
                f"volume={self.total_volume}, trades={self.trade_count})")
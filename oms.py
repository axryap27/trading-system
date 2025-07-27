#!/usr/bin/env python3
"""
Order Management System (OMS) - validates, tracks, and routes orders
"""

from order import Order
from datetime import datetime
from typing import Dict, Optional, List


class OrderManagementSystem:
    """
    Validates, tracks, and optionally routes orders.
    Acts as the gatekeeper between strategy and exchange.
    """
    
    def __init__(self, matching_engine=None):
        """
        Initialize OMS with optional matching engine.
        
        Args:
            matching_engine: Optional matching engine to forward orders to
        """
        # Store orders (Order objects) and statuses by order ID
        self._orders: Dict[str, Order] = {}
        self._statuses: Dict[str, str] = {}
        
        # Optional matching engine to forward orders
        self.matching_engine = matching_engine
    
    def new_order(self, order: Order) -> dict:
        """
        Accept and validate a new order.
        
        Args:
            order: Order object to process
            
        Returns:
            Acknowledgment dictionary with order_id, status, timestamp
            
        Raises:
            ValueError: If order validation fails
        """
        # 1) Basic field checks (Order.__post_init__ handles basic validation)
        # Additional OMS-level validations can go here
        
        # Check if order ID already exists
        if order.id in self._orders:
            raise ValueError(f"Order ID {order.id} already exists")
        
        # 2) Timestamp if missing
        if order.timestamp is None:
            order.timestamp = datetime.utcnow()
        
        # 3) Save order & status
        self._orders[order.id] = order
        self._statuses[order.id] = "accepted"
        
        # 4) Forward to matching engine if available
        if self.matching_engine:
            try:
                self.matching_engine.add_order(order)
            except Exception as e:
                # If matching engine fails, mark order as rejected
                self._statuses[order.id] = "rejected"
                raise ValueError(f"Matching engine rejected order: {str(e)}")
        
        # 5) Return acknowledgment
        return {
            "order_id": order.id,
            "status": "accepted",
            "timestamp": order.timestamp
        }
    
    def cancel_order(self, order_id: str) -> dict:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            Cancellation acknowledgment dictionary
            
        Raises:
            KeyError: If order not found
            ValueError: If order cannot be canceled
        """
        # Look up the order
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        
        # Check current status
        current_status = self._statuses[order_id]
        if current_status in ("canceled", "filled"):
            raise ValueError(f"Cannot cancel order in status '{current_status}'")
        
        # Update status to canceled
        self._statuses[order_id] = "canceled"
        
        # Notify matching engine if available
        if self.matching_engine and hasattr(self.matching_engine, 'cancel_order'):
            try:
                self.matching_engine.cancel_order(order_id)
            except Exception as e:
                # Log warning but don't fail the cancellation
                print(f"Warning: Matching engine cancel failed for {order_id}: {e}")
        
        return {
            "order_id": order_id,
            "status": "canceled",
            "timestamp": datetime.utcnow()
        }
    
    def amend_order(self, order_id: str, new_qty: Optional[int] = None, 
                   new_price: Optional[float] = None) -> dict:
        """
        Amend an existing order's quantity and/or price.
        
        Args:
            order_id: ID of order to amend
            new_qty: New quantity (optional)
            new_price: New price (optional)
            
        Returns:
            Amendment acknowledgment dictionary
            
        Raises:
            KeyError: If order not found
            ValueError: If order cannot be amended or parameters invalid
        """
        # Look up the order
        if order_id not in self._orders:
            raise KeyError(f"Order {order_id} not found")
        
        # Only allow amend when status is "accepted"
        if self._statuses[order_id] != "accepted":
            raise ValueError(f"Only accepted orders can be amended (current status: {self._statuses[order_id]})")
        
        order = self._orders[order_id]
        
        # Validate and update quantity if provided
        if new_qty is not None:
            if new_qty <= 0:
                raise ValueError("Quantity must be > 0")
            order.quantity = new_qty
        
        # Validate and update price if provided
        if new_price is not None:
            if order.type not in ("limit", "stop"):
                raise ValueError("Only limit/stop orders can change price")
            order.price = new_price
        
        # Update timestamp
        order.timestamp = datetime.utcnow()
        
        # Notify matching engine if available
        if self.matching_engine and hasattr(self.matching_engine, 'amend_order'):
            try:
                self.matching_engine.amend_order(order_id, new_qty, new_price)
            except Exception as e:
                print(f"Warning: Matching engine amend failed for {order_id}: {e}")
        
        return {
            "order_id": order_id,
            "status": "amended",
            "timestamp": order.timestamp
        }
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status by ID."""
        return self._statuses.get(order_id)
    
    def get_all_orders(self) -> Dict[str, Order]:
        """Get all orders."""
        return self._orders.copy()
    
    def get_orders_by_status(self, status: str) -> Dict[str, Order]:
        """Get all orders with a specific status."""
        return {
            order_id: order 
            for order_id, order in self._orders.items() 
            if self._statuses[order_id] == status
        }
    
    def get_open_orders(self) -> Dict[str, Order]:
        """Get all orders that are still open (accepted status)."""
        return self.get_orders_by_status("accepted")
    
    def update_order_status(self, order_id: str, new_status: str) -> None:
        """
        Update order status (typically called by matching engine).
        
        Args:
            order_id: ID of order to update
            new_status: New status ("filled", "partial_fill", etc.)
        """
        if order_id in self._statuses:
            self._statuses[order_id] = new_status
    
    def get_order_summary(self) -> Dict[str, int]:
        """Get summary of order counts by status."""
        summary = {}
        for status in self._statuses.values():
            summary[status] = summary.get(status, 0) + 1
        return summary
    
    def __str__(self):
        """String representation showing order counts."""
        summary = self.get_order_summary()
        total = len(self._orders)
        return f"OMS({total} orders: {summary})"
    
    def __repr__(self):
        """Detailed representation."""
        return f"OrderManagementSystem(orders={len(self._orders)}, matching_engine={self.matching_engine is not None})"
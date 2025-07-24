#!/usr/bin/env python3
"""
Test script for Order class and OrderManagementSystem
"""

import uuid
from datetime import datetime
from order import Order
from oms import OrderManagementSystem


def print_section(title: str):
    """Print section header"""
    print(f"\n{title}")
    print("-" * len(title))


def test_order_class():
    """Test Order class functionality"""
    print_section("Testing Order Class")
    
    # Test 1: Valid market order
    try:
        market_order = Order(
            id="MKT001",
            symbol="AAPL",
            side="buy",
            quantity=100,
            type="market"
        )
        print(f"✓ Market order created: {market_order}")
    except Exception as e:
        print(f"✗ Market order failed: {e}")
    
    # Test 2: Valid limit order
    try:
        limit_order = Order(
            id="LMT001",
            symbol="AAPL",
            side="sell",
            quantity=50,
            type="limit",
            price=150.50,
            timestamp=datetime.utcnow()
        )
        print(f"✓ Limit order created: {limit_order}")
    except Exception as e:
        print(f"✗ Limit order failed: {e}")
    
    # Test 3: Invalid side
    try:
        invalid_order = Order(
            id="BAD001",
            symbol="AAPL",
            side="invalid",
            quantity=100,
            type="market"
        )
        print(f"✗ Should have failed: {invalid_order}")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid side: {e}")
    
    # Test 4: Invalid quantity
    try:
        invalid_order = Order(
            id="BAD002",
            symbol="AAPL",
            side="buy",
            quantity=-10,
            type="market"
        )
        print(f"✗ Should have failed: {invalid_order}")
    except ValueError as e:
        print(f"✓ Correctly rejected negative quantity: {e}")
    
    # Test 5: Limit order without price
    try:
        invalid_order = Order(
            id="BAD003",
            symbol="AAPL",
            side="buy",
            quantity=100,
            type="limit"
        )
        print(f"✗ Should have failed: {invalid_order}")
    except ValueError as e:
        print(f"✓ Correctly rejected limit order without price: {e}")


def test_oms_basic():
    """Test basic OMS functionality"""
    print_section("Testing OMS Basic Functionality")
    
    oms = OrderManagementSystem()
    
    # Test 1: New order acceptance
    order1 = Order(
        id="TEST001",
        symbol="AAPL",
        side="buy",
        quantity=100,
        type="market"
    )
    
    try:
        ack = oms.new_order(order1)
        print(f"✓ Order accepted: {ack}")
        print(f"  Order status: {oms.get_order_status('TEST001')}")
    except Exception as e:
        print(f"✗ Order acceptance failed: {e}")
    
    # Test 2: Duplicate order ID
    try:
        duplicate_order = Order(
            id="TEST001",  # Same ID
            symbol="MSFT",
            side="sell",
            quantity=50,
            type="market"
        )
        ack = oms.new_order(duplicate_order)
        print(f"✗ Should have rejected duplicate ID")
    except ValueError as e:
        print(f"✓ Correctly rejected duplicate ID: {e}")
    
    # Test 3: Order retrieval
    retrieved_order = oms.get_order("TEST001")
    if retrieved_order:
        print(f"✓ Retrieved order: {retrieved_order}")
    else:
        print("✗ Failed to retrieve order")
    
    # Test 4: Order summary
    summary = oms.get_order_summary()
    print(f"✓ Order summary: {summary}")
    print(f"✓ OMS status: {oms}")


def test_oms_order_lifecycle():
    """Test complete order lifecycle"""
    print_section("Testing Order Lifecycle")
    
    oms = OrderManagementSystem()
    
    # Create a limit order
    order = Order(
        id=str(uuid.uuid4()),
        symbol="AAPL",
        side="buy",
        quantity=100,
        type="limit",
        price=149.50
    )
    
    order_id = order.id
    
    # Step 1: Submit order
    ack = oms.new_order(order)
    print(f"✓ Order submitted: {ack['status']}")
    
    # Step 2: Amend order quantity
    try:
        amend_ack = oms.amend_order(order_id, new_qty=150)
        print(f"✓ Order amended (qty): {amend_ack['status']}")
        updated_order = oms.get_order(order_id)
        print(f"  New quantity: {updated_order.quantity}")
    except Exception as e:
        print(f"✗ Amendment failed: {e}")
    
    # Step 3: Amend order price
    try:
        amend_ack = oms.amend_order(order_id, new_price=150.00)
        print(f"✓ Order amended (price): {amend_ack['status']}")
        updated_order = oms.get_order(order_id)
        print(f"  New price: ${updated_order.price:.2f}")
    except Exception as e:
        print(f"✗ Price amendment failed: {e}")
    
    # Step 4: Cancel order
    try:
        cancel_ack = oms.cancel_order(order_id)
        print(f"✓ Order canceled: {cancel_ack['status']}")
        print(f"  Final status: {oms.get_order_status(order_id)}")
    except Exception as e:
        print(f"✗ Cancellation failed: {e}")
    
    # Step 5: Try to amend canceled order (should fail)
    try:
        amend_ack = oms.amend_order(order_id, new_qty=200)
        print(f"✗ Should not be able to amend canceled order")
    except ValueError as e:
        print(f"✓ Correctly rejected amend of canceled order: {e}")


def test_oms_edge_cases():
    """Test edge cases and error handling"""
    print_section("Testing Edge Cases")
    
    oms = OrderManagementSystem()
    
    # Test 1: Cancel non-existent order
    try:
        cancel_ack = oms.cancel_order("NONEXISTENT")
        print(f"✗ Should have failed")
    except KeyError as e:
        print(f"✓ Correctly handled non-existent order: {e}")
    
    # Test 2: Amend non-existent order
    try:
        amend_ack = oms.amend_order("NONEXISTENT", new_qty=100)
        print(f"✗ Should have failed")
    except KeyError as e:
        print(f"✓ Correctly handled non-existent order amendment: {e}")
    
    # Test 3: Invalid amendment parameters
    order = Order(
        id="AMEND_TEST",
        symbol="AAPL", 
        side="buy",
        quantity=100,
        type="market"
    )
    oms.new_order(order)
    
    try:
        # Try to set price on market order
        amend_ack = oms.amend_order("AMEND_TEST", new_price=150.0)
        print(f"✗ Should not be able to set price on market order")
    except ValueError as e:
        print(f"✓ Correctly rejected price on market order: {e}")
    
    try:
        # Try to set invalid quantity
        amend_ack = oms.amend_order("AMEND_TEST", new_qty=-50)
        print(f"✗ Should not accept negative quantity")
    except ValueError as e:
        print(f"✓ Correctly rejected negative quantity: {e}")


def test_oms_queries():
    """Test OMS query methods"""
    print_section("Testing OMS Query Methods")
    
    oms = OrderManagementSystem()
    
    # Create multiple orders with different statuses
    orders_data = [
        ("ORDER1", "buy", "accepted"),
        ("ORDER2", "sell", "accepted"), 
        ("ORDER3", "buy", "filled"),
        ("ORDER4", "sell", "canceled")
    ]
    
    for order_id, side, target_status in orders_data:
        order = Order(
            id=order_id,
            symbol="AAPL",
            side=side,
            quantity=100,
            type="market"
        )
        oms.new_order(order)
        
        # Manually set status for testing
        if target_status != "accepted":
            oms.update_order_status(order_id, target_status)
    
    # Test queries
    all_orders = oms.get_all_orders()
    print(f"✓ Total orders: {len(all_orders)}")
    
    open_orders = oms.get_open_orders()
    print(f"✓ Open orders: {len(open_orders)}")
    
    filled_orders = oms.get_orders_by_status("filled")
    print(f"✓ Filled orders: {len(filled_orders)}")
    
    canceled_orders = oms.get_orders_by_status("canceled")
    print(f"✓ Canceled orders: {len(canceled_orders)}")
    
    summary = oms.get_order_summary()
    print(f"✓ Status summary: {summary}")


class MockMatchingEngine:
    """Mock matching engine for testing OMS integration"""
    
    def __init__(self):
        self.received_orders = []
        self.should_fail = False
    
    def add_order(self, order):
        if self.should_fail:
            raise Exception("Mock matching engine failure")
        self.received_orders.append(order)
        print(f"  Mock engine received: {order.id}")


def test_oms_with_matching_engine():
    """Test OMS integration with matching engine"""
    print_section("Testing OMS with Matching Engine")
    
    mock_engine = MockMatchingEngine()
    oms = OrderManagementSystem(matching_engine=mock_engine)
    
    # Test successful order routing
    order = Order(
        id="ENGINE_TEST",
        symbol="AAPL",
        side="buy",
        quantity=100,
        type="market"
    )
    
    try:
        ack = oms.new_order(order)
        print(f"✓ Order routed successfully: {ack['status']}")
        print(f"  Engine received {len(mock_engine.received_orders)} orders")
    except Exception as e:
        print(f"✗ Order routing failed: {e}")
    
    # Test matching engine failure
    mock_engine.should_fail = True
    
    failed_order = Order(
        id="ENGINE_FAIL",
        symbol="AAPL",
        side="sell",
        quantity=50,
        type="market"
    )
    
    try:
        ack = oms.new_order(failed_order)
        print(f"✗ Should have failed due to engine error")
    except ValueError as e:
        print(f"✓ Correctly handled engine failure: {e}")
        print(f"  Order status: {oms.get_order_status('ENGINE_FAIL')}")


if __name__ == "__main__":
    print("Testing Order and OMS Implementation")
    print("=" * 50)
    
    test_order_class()
    test_oms_basic()
    test_oms_order_lifecycle()
    test_oms_edge_cases()
    test_oms_queries()
    test_oms_with_matching_engine()
    
    print("\n" + "="*50)
    print("Order and OMS testing completed!")
    print("="*50)
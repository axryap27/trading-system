#!/usr/bin/env python3
"""
Demo script matching the homework test format for LimitOrderBook
"""

from order import Order
from order_book import LimitOrderBook


def test_homework_examples():
    """Test the exact examples from the homework"""
    print("Testing Homework Examples")
    print("=" * 30)
    
    # Basic sanity test
    print("\n1. Basic sanity test:")
    lob = LimitOrderBook("AAPL")
    
    buy = Order("1", "AAPL", "buy", 10, "limit", 150.0)
    sell = Order("2", "AAPL", "sell", 5, "limit", 149.0)
    
    print("Adding buy order (no match expected):")
    result1 = lob.add_order(buy)
    print(f"Reports: {result1}")  # Should be empty - no match
    
    print("\nAdding sell order (should match):")
    result2 = lob.add_order(sell) 
    print(f"Reports: {len(result2)} execution reports")
    for i, report in enumerate(result2):
        print(f"  Report {i+1}: {report}")
    
    print(f"\nBook state after match:")
    print(f"Bids: {[(o.quantity, o.price) for o in lob.bids]}")
    print(f"Asks: {[(o.quantity, o.price) for o in lob.asks]}")
    print(f"Expected: bids: [(5, 150.0)], asks: []")
    
    # Market order test
    print("\n2. Market order test:")
    mk = Order("3", "AAPL", "buy", 100, "market")
    result3 = lob.add_order(mk)
    print(f"Market order reports: {len(result3)}")
    for report in result3:
        print(f"  {report}")
    
    print(f"\nFinal book state:")
    print(f"Bids: {[(o.quantity, o.price) for o in lob.bids]}")
    print(f"Asks: {[(o.quantity, o.price) for o in lob.asks]}")
    print("Expected: Both sides empty (market order consumed remaining 5@150.0)")


def test_edge_cases():
    """Test edge cases like multiple orders at same price"""
    print("\n\n3. Edge cases - Multiple orders at same price:")
    
    lob = LimitOrderBook("AAPL")
    
    # Add multiple orders at same price to test time priority
    orders = [
        Order("TIME1", "AAPL", "buy", 100, "limit", 150.0),
        Order("TIME2", "AAPL", "buy", 200, "limit", 150.0),  # Same price
        Order("TIME3", "AAPL", "buy", 50, "limit", 150.0),   # Same price
    ]
    
    for order in orders:
        lob.add_order(order)
    
    print("Added 3 buy orders at 150.0:")
    print(f"Book: {[(o.id, o.quantity, o.price) for o in lob.bids]}")
    
    # Aggressive sell that partially fills
    partial_sell = Order("PARTIAL", "AAPL", "sell", 250, "limit", 149.0)
    reports = lob.add_order(partial_sell)
    
    print(f"\nPartial fill with 250 sell order:")
    print(f"Execution reports: {len(reports)}")
    
    # Show which orders got filled in what order
    resting_fills = [r for r in reports if r['order_id'] != 'PARTIAL']
    print("Resting order fills (should be in time priority):")
    for fill in resting_fills:
        print(f"  Order {fill['order_id']}: {fill['filled_qty']} shares")
    
    print(f"\nRemaining book:")
    print(f"Bids: {[(o.id, o.quantity, o.price) for o in lob.bids]}")


def test_book_statistics():
    """Test book statistics and depth"""
    print("\n\n4. Book statistics and depth:")
    
    lob = LimitOrderBook("AAPL")
    
    # Build a more complex book
    orders = [
        # Bids (descending price)
        Order("B1", "AAPL", "buy", 100, "limit", 150.0),
        Order("B2", "AAPL", "buy", 200, "limit", 149.5),
        Order("B3", "AAPL", "buy", 150, "limit", 149.0),
        # Asks (ascending price)
        Order("A1", "AAPL", "sell", 75, "limit", 151.0),
        Order("A2", "AAPL", "sell", 125, "limit", 151.5),
        Order("A3", "AAPL", "sell", 200, "limit", 152.0),
    ]
    
    for order in orders:
        lob.add_order(order)
    
    print("Built complex book:")
    stats = lob.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nBook depth (top 3 levels):")
    depth = lob.get_book_depth(3)
    print(f"  Bids: {depth['bids']}")
    print(f"  Asks: {depth['asks']}")
    print(f"  Spread: {depth['spread']}")
    print(f"  Mid: {depth['mid_price']}")


if __name__ == "__main__":
    test_homework_examples()
    test_edge_cases()
    test_book_statistics()
    
    print("\n" + "="*50)
    print("All homework examples completed successfully!")
    print("="*50)
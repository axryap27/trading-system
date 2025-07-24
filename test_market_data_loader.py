#!/usr/bin/env python3
"""
Test script for MarketDataLoader - demonstrates all functionality
"""

import pandas as pd
from datetime import datetime
from market_data_loader import MarketDataLoader


def print_section(title: str):
    """Prints a section header surrounded by underlines."""
    print(f"\n{title}")
    print("-" * len(title))


def pretty_table(df: pd.DataFrame, caption: str = None, float_format: str = "{:.6f}"):
    """
    Prints DataFrame as an aligned ASCII table.
    """
    if caption:
        print(f"\n{caption}")
    
    if df.empty:
        print("No data available")
        return
        
    # Format floats
    formatters = {
        col: (lambda x, fmt=float_format: fmt.format(x) if pd.notna(x) else "")
        for col in df.select_dtypes(include=["float", "float64"]).columns
    }
    
    print(
        df.to_string(
            index=False,
            formatters=formatters,
            na_rep="",
            justify="left"
        )
    )


def test_basic_functionality():
    """Test basic MarketDataLoader functionality"""
    print("Testing MarketDataLoader Implementation")
    print("=" * 50)
    
    # Initialize loader
    loader = MarketDataLoader(interval="5m", period="1mo")
    
    # Test 1: Fixed-period history
    print_section("1) Fixed-period History: AAPL (Last 1 Month)")
    try:
        hist = loader.get_history("AAPL")
        print(f"Shape: {hist.shape}")
        print(f"Columns: {list(hist.columns)}")
        print(f"Date range: {hist.index.min()} to {hist.index.max()}")
        print(f"Timezone: {hist.index.tz}")
        
        df1 = hist.head(3).reset_index().rename(columns={"index": "timestamp"})
        df2 = hist.tail(3).reset_index().rename(columns={"index": "timestamp"})
        pretty_table(df1, caption="First 3 rows of AAPL history")
        pretty_table(df2, caption="Last 3 rows of AAPL history")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Explicit date range
    print_section("2) Explicit Date Range: Recent dates for AAPL")
    try:
        start, end = "2024-07-01", "2024-07-15"
        hist_range = loader.get_history("AAPL", start=start, end=end)
        
        if not hist_range.empty:
            print(f"Range Start: {hist_range.index.min()}")
            print(f"Range End: {hist_range.index.max()}")
            print(f"Shape: {hist_range.shape}")
            
            df3 = hist_range.head(3).reset_index().rename(columns={"index": "timestamp"})
            pretty_table(df3, caption="Sample rows from date range")
        else:
            print("No data available for the specified range")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Price lookup
    print_section("3) Price Lookup Test")
    try:
        test_date = datetime(2024, 7, 15, 15, 30)  # Use a reasonable test date
        price = loader.get_price("AAPL", test_date)
        bid, ask = loader.get_bid_ask("AAPL", test_date)
        
        print(f"Test timestamp: {test_date}")
        print(f"Price: ${price:.2f}")
        print(f"Bid: ${bid:.2f}")
        print(f"Ask: ${ask:.2f}")
        print(f"Spread: ${ask - bid:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Volume calculation
    print_section("4) Volume Test")
    try:
        vol = loader.get_volume("AAPL", start="2024-07-01", end="2024-07-15")
        print(f"Total Volume: {vol:,}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Options chain (may not work for all symbols)
    print_section("5) Options Chain Test")
    try:
        opts = loader.get_option_chain("AAPL")
        
        if not opts["calls"].empty:
            calls_sample = opts["calls"].head(3)
            puts_sample = opts["puts"].head(3)
            
            print("Calls sample:")
            print(calls_sample[['strike', 'lastPrice', 'bid', 'ask', 'volume']].to_string())
            
            print("\nPuts sample:")
            print(puts_sample[['strike', 'lastPrice', 'bid', 'ask', 'volume']].to_string())
        else:
            print("No options data available")
            
    except Exception as e:
        print(f"Options error (this is often expected): {e}")
    
    # Test 6: FX and Crypto
    print_section("6) Multi-Asset Test")
    
    # Test FX
    try:
        print("Testing EURUSD:")
        eur_hist = loader.get_history("EURUSD=X")
        if not eur_hist.empty:
            print(f"EURUSD shape: {eur_hist.shape}")
            latest_eur = eur_hist.iloc[-1]['last_price']
            print(f"Latest EURUSD: {latest_eur:.4f}")
    except Exception as e:
        print(f"EURUSD error: {e}")
    
    # Test Crypto
    try:
        print("\nTesting BTC-USD:")
        btc_hist = loader.get_history("BTC-USD")
        if not btc_hist.empty:
            print(f"BTC-USD shape: {btc_hist.shape}")
            latest_btc = btc_hist.iloc[-1]['last_price']
            print(f"Latest BTC-USD: ${latest_btc:,.2f}")
    except Exception as e:
        print(f"BTC-USD error: {e}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print_section("7) Edge Cases and Error Handling")
    
    loader = MarketDataLoader(interval="1d", period="5d")
    
    # Test invalid symbol
    try:
        invalid_data = loader.get_history("INVALID_SYMBOL_12345")
        print("Unexpected: Invalid symbol returned data")
    except Exception as e:
        print(f"✓ Invalid symbol correctly handled: {type(e).__name__}")
    
    # Test caching
    print("\nTesting caching:")
    import time
    start_time = time.time()
    hist1 = loader.get_history("AAPL")
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    hist2 = loader.get_history("AAPL")  # Should be cached
    second_call_time = time.time() - start_time
    
    print(f"First call time: {first_call_time:.3f}s")
    print(f"Second call time: {second_call_time:.3f}s")
    print(f"Data shapes match: {hist1.shape == hist2.shape}")
    print(f"✓ Caching appears to be working" if second_call_time < first_call_time else "⚠ Caching may not be working")


if __name__ == "__main__":
    test_basic_functionality()
    test_edge_cases()
    
    print("\n" + "="*50)
    print("MarketDataLoader testing completed!")
    print("="*50)
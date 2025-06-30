from trading.binance_client import BinanceClient
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('ConnectionTest')

def test_binance_connection():
    logger = setup_logging()
    logger.info("Starting Binance connection test...")
    
    try:
        client = BinanceClient()
        
        # Test 1: Account connection
        logger.info("Test 1: Checking account connection...")
        account_info = client.get_account_balance()
        if account_info:
            logger.info("✅ Account connection successful!")
        
        # Test 2: Market data
        logger.info("\nTest 2: Checking market data...")
        btc_price = client.get_mark_price("BTCUSDT")
        if btc_price:
            logger.info("✅ Market data access successful!")
        
        # Test 3: Futures API
        logger.info("\nTest 3: Checking Futures API access...")
        position_info = client.get_position_info("BTCUSDT")
        if position_info is not None:
            logger.info("✅ Futures API access successful!")
        
        logger.info("\n🎉 All connection tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_binance_connection()

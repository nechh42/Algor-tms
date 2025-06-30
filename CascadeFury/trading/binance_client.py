from binance.client import Client
from binance.enums import *
import config
import logging

class BinanceClient:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.test_mode = config.TEST_MODE
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BinanceClient')

    def get_symbol_info(self, symbol):
        try:
            return self.client.futures_exchange_info()['symbols'][symbol]
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None

    def get_account_balance(self):
        try:
            return self.client.futures_account_balance()
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return None

    def place_order(self, symbol, side, order_type, quantity, price=None, stop_price=None):
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }

            if price:
                params['price'] = price
            if stop_price:
                params['stopPrice'] = stop_price

            if self.test_mode:
                self.logger.info(f"TEST ORDER: {symbol} {side} {quantity} @ {price if price else stop_price} USDT")
                return {"orderId": "test", "status": "TEST"}
            else:
                return self.client.futures_create_order(**params)
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def set_leverage(self, symbol, leverage):
        try:
            return self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return None

    def get_position_info(self, symbol):
        try:
            return self.client.futures_position_information(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Error getting position info: {e}")
            return None

    def get_mark_price(self, symbol):
        try:
            mark_price_info = self.client.futures_mark_price(symbol=symbol)
            current_price = float(mark_price_info['markPrice'])
            self.logger.info(f"Market Data for {symbol}:")
            self.logger.info(f"Mark Price: {current_price}")
            
            # Get 24h price change
            ticker = self.client.futures_ticker(symbol=symbol)
            price_change = float(ticker['priceChangePercent'])
            volume = float(ticker['volume'])
            
            self.logger.info(f"24h Change: {price_change}%")
            self.logger.info(f"24h Volume: {volume}")
            
            return current_price
        except Exception as e:
            self.logger.error(f"Error getting mark price: {e}")
            return None

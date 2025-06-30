import numpy as np
from trading.binance_client import BinanceClient
import config
import logging

class GridStrategy:
    def __init__(self, symbol, grid_levels=config.GRID_LEVELS):
        self.symbol = symbol
        self.grid_levels = grid_levels
        self.client = BinanceClient()
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger('GridStrategy')

    def calculate_grid_levels(self, current_price, grid_width_percent=1.0):
        """Calculate grid levels around the current price"""
        try:
            half_levels = self.grid_levels // 2
            grid_width = current_price * (grid_width_percent / 100)
            
            # Create grid levels above and below current price
            upper_levels = np.linspace(current_price, current_price + grid_width, half_levels)
            lower_levels = np.linspace(current_price - grid_width, current_price, half_levels)
            
            # Combine and sort grid levels
            grid_prices = np.unique(np.concatenate([lower_levels, upper_levels]))
            
            # Log grid levels
            self.logger.info(f"\nGrid Levels for {self.symbol} (Current Price: {current_price:.2f}):")
            for i, price in enumerate(grid_prices):
                if price < current_price:
                    self.logger.info(f"Buy Level {i+1}: {price:.2f} USDT")
                elif price > current_price:
                    self.logger.info(f"Sell Level {i-half_levels+1}: {price:.2f} USDT")
                else:
                    self.logger.info(f"Current Price Level: {price:.2f} USDT")
            
            return grid_prices.tolist()
        except Exception as e:
            self.logger.error(f"Error calculating grid levels: {e}")
            return []

    def place_grid_orders(self, grid_prices, position_size):
        """Place orders at each grid level"""
        try:
            current_price = self.client.get_mark_price(self.symbol)
            if not current_price:
                return False

            for price in grid_prices:
                if price < current_price:
                    # Place buy order
                    self.client.place_order(
                        symbol=self.symbol,
                        side="BUY",
                        order_type="LIMIT",
                        quantity=position_size,
                        price=price
                    )
                else:
                    # Place sell order
                    self.client.place_order(
                        symbol=self.symbol,
                        side="SELL",
                        order_type="LIMIT",
                        quantity=position_size,
                        price=price
                    )
            return True
        except Exception as e:
            self.logger.error(f"Error placing grid orders: {e}")
            return False

    def set_stop_loss_take_profit(self, position_size, entry_price):
        """Set stop loss and take profit orders"""
        try:
            # Calculate stop loss and take profit prices
            stop_loss_price = entry_price * (1 - config.STOP_LOSS_PERCENT / 100)
            take_profit_price = entry_price * (1 + config.TAKE_PROFIT_PERCENT / 100)

            # Place stop loss order
            self.client.place_order(
                symbol=self.symbol,
                side="SELL",
                order_type="STOP_MARKET",
                quantity=position_size,
                stop_price=stop_loss_price
            )

            # Place take profit order
            self.client.place_order(
                symbol=self.symbol,
                side="SELL",
                order_type="TAKE_PROFIT_MARKET",
                quantity=position_size,
                stop_price=take_profit_price
            )
            return True
        except Exception as e:
            self.logger.error(f"Error setting SL/TP: {e}")
            return False

    def adjust_leverage(self, target_leverage):
        """Adjust position leverage"""
        try:
            return self.client.set_leverage(self.symbol, target_leverage)
        except Exception as e:
            self.logger.error(f"Error adjusting leverage: {e}")
            return None

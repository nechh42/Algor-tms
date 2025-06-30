import time
import schedule
from trading.binance_client import BinanceClient
from trading.coin_selector import CoinSelector
from trading.capital_manager import CapitalManager
from trading.performance_analyzer import PerformanceAnalyzer
from strategies.grid_strategy import GridStrategy
import config
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('CascadeFury')

class CascadeFury:
    def __init__(self):
        self.logger = setup_logging()
        self.client = BinanceClient()
        self.coin_selector = CoinSelector(self.client.client)
        self.capital_manager = CapitalManager(self.client.client)
        self.performance_analyzer = PerformanceAnalyzer()
        self.strategies = {}
        self.initialize_strategies()

    def initialize_strategies(self):
        """Initialize trading strategies for each symbol"""
        # Get best trading pairs
        trading_pairs = self.coin_selector.get_top_coins()
        
        # Calculate optimal position size
        position_size = self.capital_manager.calculate_position_sizes(len(trading_pairs))
        
        # Calculate total exposure
        total_exposure = position_size * len(trading_pairs)
        
        # Adjust leverage based on total exposure
        optimal_leverage = self.capital_manager.adjust_leverage(total_exposure)
        
        for symbol in trading_pairs:
            self.strategies[symbol] = GridStrategy(symbol)
            # Set optimal leverage
            self.strategies[symbol].adjust_leverage(optimal_leverage)
            
        self.logger.info(f"Initialized strategies for {len(trading_pairs)} pairs")
        self.logger.info(f"Position Size: {position_size} USDT")
        self.logger.info(f"Total Exposure: {total_exposure} USDT")
        self.logger.info(f"Leverage: {optimal_leverage}x")

    def run_trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            for symbol in self.strategies.keys():
                self.logger.info(f"Checking {symbol}...")
                strategy = self.strategies[symbol]
                
                # Get current market price
                current_price = self.client.get_mark_price(symbol)
                if not current_price:
                    continue
                    
                self.logger.info(f"{symbol} current price: {current_price}")

                # Calculate grid levels
                grid_prices = strategy.calculate_grid_levels(current_price)
                if grid_prices:
                    self.logger.info(f"{symbol} grid levels: {grid_prices}")
                    
                    # Place grid orders
                    strategy.place_grid_orders(grid_prices, config.POSITION_SIZE_USDT)
                    self.logger.info(f"Placed test orders for {symbol}")
                    
                    # Set stop loss and take profit for the position
                    strategy.set_stop_loss_take_profit(
                        config.POSITION_SIZE_USDT,
                        current_price
                    )
                    self.logger.info(f"Set SL/TP for {symbol}")
                    
                # Execute trades
                trade_result = strategy.execute_trades()
                
                # Log trade results for analysis
                if trade_result:
                    self.performance_analyzer.add_trade(
                        symbol=trade_result['symbol'],
                        entry_price=trade_result['entry_price'],
                        exit_price=trade_result['exit_price'],
                        position_size=trade_result['position_size'],
                        side=trade_result['side']
                    )
                    
            # Her 4 saatte bir performans analizi
            if datetime.now().hour % 4 == 0:
                ready_for_live, message = self.performance_analyzer.analyze_performance()
                if ready_for_live:
                    self.logger.warning("🚀 Bot gerçek trading'e hazır! TEST_MODE=false yapabilirsiniz.")
                    
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")

    def start(self):
        """Start the trading bot"""
        self.logger.info("Starting CascadeFury Trading Bot...")
        self.logger.info(f"Test Mode: {'Enabled' if config.TEST_MODE else 'Disabled'}")
        
        # Run first cycle immediately
        self.run_trading_cycle()
        
        # Schedule regular trading cycles
        schedule.every(5).minutes.do(self.run_trading_cycle)
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Shutting down CascadeFury Trading Bot...")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    bot = CascadeFury()
    bot.start()

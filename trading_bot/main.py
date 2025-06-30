# main.py - Ana program
from config import API_KEY, API_SECRET
from config import *
from indicators import calculate_all_indicators
from risk_manager import RiskManager
from strategy import TradingStrategy
from database import Database
from ui import UserInterface
import ccxt
import asyncio

class BinanceBot:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True
        })
        self.risk_manager = RiskManager(self)
        self.strategy = TradingStrategy()
        self.db = Database()
        self.ui = UserInterface(self)
        
    async def start(self):
        await asyncio.gather(
            self.monitor_markets(),
            self.ui.render()
        )

if __name__ == "__main__":
    bot = BinanceBot()
    asyncio.run(bot.start())
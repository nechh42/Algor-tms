from binance.client import Client
from binance.enums import *
import config
import pandas as pd
from strategy import TradingStrategy
from risk_manager import RiskManager
import time
import logging
from datetime import datetime
from multi_market_analyzer import MultiMarketAnalyzer
from trade_executor import TradeExecutor
from trade_ai import TradeAI
from performance_tracker import PerformanceTracker

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

class NexusOracle:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.strategy = TradingStrategy(config)
        self.risk_manager = RiskManager(config)
        
    def get_historical_data(self):
        """Son verileri al"""
        klines = self.client.futures_klines(
            symbol=config.SYMBOL,
            interval=Client.KLINE_INTERVAL_15MINUTE,
            limit=100
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['close'] = pd.to_numeric(df['close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    def execute_trade(self, signal):
        """İşlem gerçekleştir"""
        try:
            if not self.risk_manager.can_open_position():
                logging.info("Maksimum pozisyon sayısına ulaşıldı")
                return
                
            balance = float(self.client.futures_account_balance()[0]['balance'])
            current_price = float(self.client.futures_symbol_ticker(symbol=config.SYMBOL)['price'])
            
            # Pozisyon büyüklüğünü hesapla
            quantity = self.risk_manager.calculate_position_size(current_price, balance)
            
            if signal == 'LONG':
                order = self.client.futures_create_order(
                    symbol=config.SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                stop_loss = self.risk_manager.calculate_stop_loss(current_price, 'LONG')
                take_profit = self.risk_manager.calculate_take_profit(current_price, 'LONG')
                
            elif signal == 'SHORT':
                order = self.client.futures_create_order(
                    symbol=config.SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                stop_loss = self.risk_manager.calculate_stop_loss(current_price, 'SHORT')
                take_profit = self.risk_manager.calculate_take_profit(current_price, 'SHORT')
                
            # Stop loss ve take profit emirlerini yerleştir
            self.place_sl_tp_orders(current_price, stop_loss, take_profit, signal)
            
            logging.info(f"İşlem gerçekleştirildi: {signal} - Fiyat: {current_price}")
            
        except Exception as e:
            logging.error(f"İşlem hatası: {str(e)}")
            
    def place_sl_tp_orders(self, entry_price, stop_loss, take_profit, position_type):
        """Stop loss ve take profit emirlerini yerleştir"""
        side = SIDE_SELL if position_type == 'LONG' else SIDE_BUY
        
        # Stop Loss emri
        self.client.futures_create_order(
            symbol=config.SYMBOL,
            side=side,
            type=ORDER_TYPE_STOP_MARKET,
            stopPrice=stop_loss,
            closePosition=True
        )
        
        # Take Profit emri
        self.client.futures_create_order(
            symbol=config.SYMBOL,
            side=side,
            type=ORDER_TYPE_TAKE_PROFIT_MARKET,
            stopPrice=take_profit,
            closePosition=True
        )
        
def main():
    analyzer = MultiMarketAnalyzer()
    executor = TradeExecutor()
    ai = TradeAI()
    tracker = PerformanceTracker()
    
    logging.info("NexusOracle başlatılıyor...")
    
    # Başlangıç performans raporu
    tracker.print_detailed_report()
    
    while True:
        try:
            # Tüm marketleri analiz et
            opportunities = analyzer.analyze_all_markets()
            
            # Her fırsat için AI tahminini al
            for opp in opportunities:
                ai_probability = ai.predict_probability(opp)
                opp['ai_score'] = ai_probability
                
            # İşlem önerilerini al
            suggestions = tracker.history.get_position_suggestions()
            if suggestions:
                # Önerilen sembollere öncelik ver
                opportunities = sorted(opportunities, 
                    key=lambda x: (x['symbol'] in suggestions['recommended_symbols'], x['score']), 
                    reverse=True
                )
                
            # Fırsatları değerlendir ve işlem aç
            new_trades = executor.execute_trades(opportunities)
            
            # Yeni işlemleri kaydet
            for trade in new_trades:
                tracker.add_trade(trade)
            
            # Her 24 saatte bir AI modelini güncelle ve rapor oluştur
            if time.time() % 86400 < 300:  # Her gün başında
                trade_history = tracker.get_trade_history()
                if trade_history:
                    ai.train_model(trade_history)
                    tracker.print_detailed_report()
                    tracker.history.get_position_suggestions()
            
            # 5 dakika bekle
            time.sleep(300)
            
        except Exception as e:
            logging.error(f"Ana döngüde hata: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()

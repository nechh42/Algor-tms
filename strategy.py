import numpy as np
import pandas as pd
import ta
import logging
from datetime import datetime

class TradingStrategy:
    def __init__(self, config):
        self.config = config
        
    def calculate_signals(self, df):
        # RSI Hesaplama
        df['RSI'] = ta.momentum.RSIIndicator(
            df['close'], 
            window=self.config.RSI_PERIOD
        ).rsi()
        
        # MACD Hesaplama
        macd = ta.trend.MACD(
            df['close'],
            window_slow=self.config.EMA_SLOW,
            window_fast=self.config.EMA_FAST,
            window_sign=self.config.MACD_SIGNAL
        )
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        return self.generate_signals(df)
    
    def generate_signals(self, df):
        signals = []
        
        # Long Pozisyon Sinyalleri
        long_condition = (
            (df['RSI'] < self.config.RSI_OVERSOLD) &
            (df['MACD'] > df['MACD_Signal']) &
            (df['close'] < df['BB_lower'])
        )
        
        # Short Pozisyon Sinyalleri
        short_condition = (
            (df['RSI'] > self.config.RSI_OVERBOUGHT) &
            (df['MACD'] < df['MACD_Signal']) &
            (df['close'] > df['BB_upper'])
        )
        
        for i in range(len(df)):
            if long_condition.iloc[i]:
                signals.append('LONG')
            elif short_condition.iloc[i]:
                signals.append('SHORT')
            else:
                signals.append('HOLD')
                
        return signals

class GridTradingStrategy:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.grid_levels = []
        self.active_orders = {}
        
    def setup_grid_strategy(self, current_price, grid_size=10, price_range_percent=2.0):
        """Grid seviyeleri oluştur"""
        try:
            upper_price = current_price * (1 + price_range_percent/100)
            lower_price = current_price * (1 - price_range_percent/100)
            
            # Grid seviyeleri hesapla
            self.grid_levels = np.linspace(lower_price, upper_price, grid_size)
            
            # Her seviye için alış ve satış emirleri oluştur
            for i, price in enumerate(self.grid_levels):
                self.active_orders[f"buy_order_{i}"] = {
                    'type': 'BUY',
                    'price': price,
                    'status': 'active'
                }
                self.active_orders[f"sell_order_{i}"] = {
                    'type': 'SELL',
                    'price': price * 1.005,  # %0.5 kar marjı
                    'status': 'active'
                }
            
            self.logger.info(f"Grid strateji kuruldu - {grid_size} seviye")
            return True
            
        except Exception as e:
            self.logger.error(f"Grid strateji kurulumu hatası: {str(e)}")
            return False
            
    def update_grid_orders(self, current_price):
        """Grid emirlerini güncelle"""
        try:
            executed_orders = []
            
            # Her emri kontrol et
            for order_id, order in self.active_orders.items():
                if order['status'] != 'active':
                    continue
                    
                # Alış emirleri için
                if order['type'] == 'BUY' and current_price <= order['price']:
                    executed_orders.append({
                        'order_id': order_id,
                        'type': 'BUY',
                        'price': order['price'],
                        'timestamp': datetime.now()
                    })
                    order['status'] = 'executed'
                    
                # Satış emirleri için
                elif order['type'] == 'SELL' and current_price >= order['price']:
                    executed_orders.append({
                        'order_id': order_id,
                        'type': 'SELL',
                        'price': order['price'],
                        'timestamp': datetime.now()
                    })
                    order['status'] = 'executed'
            
            # Yeni emirler oluştur
            if executed_orders:
                self._create_new_grid_orders(current_price)
                
            return executed_orders
            
        except Exception as e:
            self.logger.error(f"Grid emir güncelleme hatası: {str(e)}")
            return []
            
    def _create_new_grid_orders(self, current_price):
        """Yeni grid emirleri oluştur"""
        try:
            # Aktif olmayan emirleri temizle
            self.active_orders = {k: v for k, v in self.active_orders.items() 
                                if v['status'] == 'active'}
            
            # Yeni grid seviyeleri oluştur
            self.setup_grid_strategy(current_price)
            
        except Exception as e:
            self.logger.error(f"Yeni grid emir oluşturma hatası: {str(e)}")
            
    def calculate_grid_profits(self):
        """Grid işlem karlarını hesapla"""
        total_profit = 0
        executed_buys = []
        executed_sells = []
        
        for order_id, order in self.active_orders.items():
            if order['status'] == 'executed':
                if order['type'] == 'BUY':
                    executed_buys.append(order['price'])
                else:
                    executed_sells.append(order['price'])
        
        # Eşleşen alış-satışları bul
        for buy_price in executed_buys:
            if executed_sells:
                sell_price = executed_sells.pop(0)
                profit = sell_price - buy_price
                total_profit += profit
        
        return total_profit

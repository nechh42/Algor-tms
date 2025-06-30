import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from binance.client import Client
import threading
import time

class EmergencySystem:
    def __init__(self, client: Client, max_drawdown: float = 20,
                 volatility_threshold: float = 3, volume_threshold: float = 5):
        """
        Acil durum sistemi
        :param client: Binance client
        :param max_drawdown: Maximum izin verilen drawdown yüzdesi
        :param volatility_threshold: Volatilite eşiği (standart sapma çarpanı)
        :param volume_threshold: Hacim eşiği (normal hacmin katı)
        """
        self.client = client
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.logger = logging.getLogger(__name__)
        
        # Market durumu
        self.market_status = {
            'is_crash': False,
            'high_volatility': False,
            'abnormal_volume': False,
            'last_check': None
        }
        
        # Monitoring thread'i başlat
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_market)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def check_emergency_conditions(self, positions: Dict,
                                 balance_history: List[Dict]) -> bool:
        """
        Acil durum koşullarını kontrol et
        :return: True if emergency conditions exist
        """
        try:
            emergency_conditions = []
            
            # Market çöküşü kontrolü
            if self._check_market_crash():
                emergency_conditions.append('market_crash')
                
            # Drawdown kontrolü
            if self._check_drawdown(balance_history):
                emergency_conditions.append('excessive_drawdown')
                
            # Yüksek volatilite kontrolü
            if self._check_high_volatility():
                emergency_conditions.append('high_volatility')
                
            # Anormal hacim kontrolü
            if self._check_abnormal_volume():
                emergency_conditions.append('abnormal_volume')
                
            # Pozisyon riski kontrolü
            if self._check_position_risk(positions):
                emergency_conditions.append('position_risk')
                
            if emergency_conditions:
                self.logger.warning(f"Acil durum tespit edildi: {emergency_conditions}")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Acil durum kontrolü hatası: {str(e)}")
            return False
            
    def execute_emergency_closure(self, positions: Dict) -> bool:
        """
        Acil durum kapatma işlemini gerçekleştir
        :return: True if all positions are closed successfully
        """
        try:
            success = True
            
            # Tüm pozisyonları kapat
            for symbol, position in positions.items():
                try:
                    # Market emri ile pozisyonu kapat
                    side = 'SELL' if position['side'] == 'BUY' else 'BUY'
                    order = self.client.create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=position['size']
                    )
                    
                    self.logger.info(f"Acil durum: {symbol} pozisyonu kapatıldı")
                    
                except Exception as e:
                    self.logger.error(f"Pozisyon kapatma hatası ({symbol}): {str(e)}")
                    success = False
                    
            return success
            
        except Exception as e:
            self.logger.error(f"Acil durum kapatma hatası: {str(e)}")
            return False
            
    def _monitor_market(self):
        """Sürekli market durumunu izle"""
        while self.monitoring_active:
            try:
                # Market durumunu güncelle
                self._update_market_status()
                
                # 1 dakika bekle
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Market izleme hatası: {str(e)}")
                time.sleep(60)
                
    def _update_market_status(self):
        """Market durumunu güncelle"""
        try:
            # BTC/USDT durumunu kontrol et (market göstergesi olarak)
            btc_stats = self.client.get_ticker(symbol='BTCUSDT')
            
            # Son 24 saatlik değişim
            price_change = float(btc_stats['priceChangePercent'])
            
            # Volatilite (ATR kullanarak)
            atr = self._calculate_atr('BTCUSDT', '1h', 14)
            
            # 24 saatlik hacim değişimi
            volume_change = self._calculate_volume_change('BTCUSDT')
            
            # Market durumunu güncelle
            self.market_status = {
                'is_crash': price_change < -10,  # %10'dan fazla düşüş
                'high_volatility': atr > self.volatility_threshold,
                'abnormal_volume': volume_change > self.volume_threshold,
                'last_check': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Market durum güncelleme hatası: {str(e)}")
            
    def _check_market_crash(self) -> bool:
        """Market çöküşü kontrolü"""
        return self.market_status['is_crash']
        
    def _check_drawdown(self, balance_history: List[Dict]) -> bool:
        """Drawdown kontrolü"""
        try:
            if len(balance_history) < 2:
                return False
                
            df = pd.DataFrame(balance_history)
            df['balance'] = pd.to_numeric(df['balance'])
            
            # Rolling maximum hesapla
            rolling_max = df['balance'].expanding().max()
            
            # Current drawdown hesapla
            current_drawdown = (df['balance'].iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1] * 100
            
            return abs(current_drawdown) > self.max_drawdown
            
        except Exception as e:
            self.logger.error(f"Drawdown kontrolü hatası: {str(e)}")
            return False
            
    def _check_high_volatility(self) -> bool:
        """Yüksek volatilite kontrolü"""
        return self.market_status['high_volatility']
        
    def _check_abnormal_volume(self) -> bool:
        """Anormal hacim kontrolü"""
        return self.market_status['abnormal_volume']
        
    def _check_position_risk(self, positions: Dict) -> bool:
        """Pozisyon riski kontrolü"""
        try:
            if not positions:
                return False
                
            # Toplam pozisyon büyüklüğü kontrolü
            total_exposure = sum(abs(float(pos['size'])) for pos in positions.values())
            
            # Hesap bakiyesini al
            account = self.client.get_account()
            total_balance = sum(
                float(asset['free']) * float(self.client.get_symbol_ticker(
                    symbol=f"{asset['asset']}USDT"
                )['price'])
                for asset in account['balances']
                if float(asset['free']) > 0
            )
            
            # Risk oranı kontrolü (pozisyon/bakiye)
            risk_ratio = total_exposure / total_balance
            
            return risk_ratio > 2  # 2x'ten fazla kaldıraç tehlikeli
            
        except Exception as e:
            self.logger.error(f"Pozisyon riski kontrolü hatası: {str(e)}")
            return False
            
    def _calculate_atr(self, symbol: str, interval: str, period: int) -> float:
        """ATR hesapla"""
        try:
            # Kline verilerini al
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=period
            )
            
            high_prices = [float(k[2]) for k in klines]
            low_prices = [float(k[3]) for k in klines]
            close_prices = [float(k[4]) for k in klines]
            
            # True Range hesapla
            tr_prices = []
            for i in range(len(close_prices)):
                if i != 0:
                    tr = max(
                        high_prices[i] - low_prices[i],
                        abs(high_prices[i] - close_prices[i-1]),
                        abs(low_prices[i] - close_prices[i-1])
                    )
                else:
                    tr = high_prices[i] - low_prices[i]
                    
                tr_prices.append(tr)
                
            # ATR hesapla
            atr = np.mean(tr_prices)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR hesaplama hatası: {str(e)}")
            return 0
            
    def _calculate_volume_change(self, symbol: str) -> float:
        """Hacim değişimini hesapla"""
        try:
            # Son 24 saatlik hacim
            ticker = self.client.get_ticker(symbol=symbol)
            current_volume = float(ticker['volume'])
            
            # Önceki 24 saatlik hacim
            klines = self.client.get_klines(
                symbol=symbol,
                interval='1d',
                limit=2
            )
            
            if len(klines) < 2:
                return 0
                
            previous_volume = float(klines[0][5])
            
            # Hacim değişimi (%)
            volume_change = (current_volume - previous_volume) / previous_volume * 100
            
            return volume_change
            
        except Exception as e:
            self.logger.error(f"Hacim değişimi hesaplama hatası: {str(e)}")
            return 0
            
    def stop_monitoring(self):
        """Market izlemeyi durdur"""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join()

from src.utils.logger import setup_logger
import pandas as pd
import asyncio
import json
from typing import Dict, List
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.patterns.detector import PatternDetector
from src.ml.price_predictor import PricePredictor

logger = setup_logger(__name__)

class ArbitrageStrategy:
    def __init__(self, min_profit_threshold=0.001, min_volume=10, 
                 max_leverage=20, max_positions=5, aggressive_mode=True):
        self.min_profit_threshold = min_profit_threshold
        self.min_volume = min_volume
        self.max_leverage = max_leverage
        self.max_positions = max_positions
        self.aggressive_mode = aggressive_mode
        
        # Alt sistemleri başlat
        self.indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.price_predictor = PricePredictor()
        
        # Minimum işlem tutarları
        self.min_trade_amount = 10  # $10
        self.max_trade_amount = 1000  # $1000
        
        logger.info(
            f"Arbitrage strategy initialized with:\n"
            f"Min Profit: {min_profit_threshold:.2%}\n"
            f"Min Volume: ${min_volume}\n"
            f"Max Leverage: {max_leverage}x"
        )

    async def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Market analizi yap ve arbitraj fırsatlarını bul"""
        try:
            signals = []
            
            # Temel analiz
            for symbol, data in market_data.items():
                # Teknik indikatörleri hesapla
                rsi = self.indicators.calculate_rsi(data)
                macd = self.indicators.calculate_macd(data)
                bb = self.indicators.calculate_bollinger_bands(data)
                
                # Pattern analizi
                patterns = self.pattern_detector.detect(data)
                
                # Fiyat tahmini
                prediction = self.price_predictor.predict(data)
                
                # Sinyal üret
                if patterns['strength'] > 0.7:  # Güçlü pattern
                    if 'double_bottom' in patterns['patterns'] or 'head_shoulders' in patterns['patterns']:
                        signals.append(self._generate_long_signal(symbol, data, prediction))
                    elif 'double_top' in patterns['patterns']:
                        signals.append(self._generate_short_signal(symbol, data, prediction))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            raise

    def find_triangular_arbitrage(self, prices: Dict[str, float], volumes: Dict[str, float]) -> List[Dict]:
        """Üçgen arbitraj fırsatlarını bul"""
        try:
            signals = []
            
            # USDT-BTC-ETH üçgeni
            if all(k in prices for k in ['BTCUSDT', 'ETHUSDT', 'ETHBTC']):
                btc_price = prices['BTCUSDT']
                eth_price = prices['ETHUSDT']
                eth_btc_price = prices['ETHBTC']
                
                # İleri yön: USDT -> ETH -> BTC -> USDT
                forward_rate = (1 / eth_price) * (1 / eth_btc_price) * btc_price
                
                # Geri yön: USDT -> BTC -> ETH -> USDT
                reverse_rate = (1 / btc_price) * eth_btc_price * eth_price
                
                # Minimum kâr kontrolü
                if forward_rate > (1 + self.min_profit_threshold):
                    # İşlem büyüklüğü hesapla
                    trade_size = self.calculate_optimal_size(prices, volumes)
                    
                    signals.append({
                        'type': 'TRIANGULAR',
                        'direction': 'FORWARD',
                        'profit_rate': forward_rate - 1,
                        'size': trade_size,
                        'expected_profit': trade_size * (forward_rate - 1),
                        'trades': [
                            {'symbol': 'ETHUSDT', 'side': 'BUY', 'price': eth_price, 'amount': trade_size / eth_price},
                            {'symbol': 'ETHBTC', 'side': 'SELL', 'price': eth_btc_price, 'amount': trade_size / eth_price},
                            {'symbol': 'BTCUSDT', 'side': 'SELL', 'price': btc_price, 'amount': trade_size / btc_price}
                        ]
                    })
                
                elif reverse_rate > (1 + self.min_profit_threshold):
                    trade_size = self.calculate_optimal_size(prices, volumes)
                    
                    signals.append({
                        'type': 'TRIANGULAR',
                        'direction': 'REVERSE',
                        'profit_rate': reverse_rate - 1,
                        'size': trade_size,
                        'expected_profit': trade_size * (reverse_rate - 1),
                        'trades': [
                            {'symbol': 'BTCUSDT', 'side': 'BUY', 'price': btc_price, 'amount': trade_size / btc_price},
                            {'symbol': 'ETHBTC', 'side': 'BUY', 'price': eth_btc_price, 'amount': trade_size / btc_price},
                            {'symbol': 'ETHUSDT', 'side': 'SELL', 'price': eth_price, 'amount': trade_size / eth_price}
                        ]
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error finding triangular arbitrage: {e}")
            raise

    def calculate_optimal_size(self, prices: Dict[str, float], volumes: Dict[str, float]) -> float:
        """Optimal işlem büyüklüğünü hesapla"""
        try:
            # En düşük hacimli paritenin %1'i kadar işlem yap
            min_volume = min(volumes.values())
            optimal_size = min(max(min_volume * 0.01, self.min_trade_amount), self.max_trade_amount)
            
            logger.debug(f"Optimal trade size: ${optimal_size:,.2f} USDT")
            return optimal_size
            
        except Exception as e:
            logger.error(f"Error calculating optimal size: {e}")
            return self.min_trade_amount

    def _generate_long_signal(self, symbol: str, data: pd.DataFrame, prediction: Dict) -> Dict:
        """Long sinyal oluştur"""
        try:
            current_price = data['close'].iloc[-1]
            atr = self.indicators.calculate_atr(data)
            
            # Stop loss ve take profit hesapla
            stop_loss = current_price - (atr * 2)  # 2 ATR altı
            take_profit = current_price + (atr * 3)  # 3 ATR üstü
            
            return {
                'symbol': symbol,
                'type': 'LONG',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': prediction['confidence'],
                'target_price': prediction['prediction']
            }
            
        except Exception as e:
            logger.error(f"Error generating long signal: {e}")
            raise

    def _generate_short_signal(self, symbol: str, data: pd.DataFrame, prediction: Dict) -> Dict:
        """Short sinyal oluştur"""
        try:
            current_price = data['close'].iloc[-1]
            atr = self.indicators.calculate_atr(data)
            
            # Stop loss ve take profit hesapla
            stop_loss = current_price + (atr * 2)  # 2 ATR üstü
            take_profit = current_price - (atr * 3)  # 3 ATR altı
            
            return {
                'symbol': symbol,
                'type': 'SHORT',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': prediction['confidence'],
                'target_price': prediction['prediction']
            }
            
        except Exception as e:
            logger.error(f"Error generating short signal: {e}")
            raise 
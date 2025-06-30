from binance.client import Client
import pandas as pd
import numpy as np
import ta
import config
import logging
from advanced_analysis import AdvancedAnalysis

class MultiMarketAnalyzer:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.setup_logging()
        self.advanced = AdvancedAnalysis()
        self.logger = logging.getLogger(__name__)
        self.colors = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'reset': '\033[0m'
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    def analyze_market(self, symbol):
        """Piyasa analizi yap"""
        try:
            # Temel verileri al
            klines = self.client.futures_klines(symbol=symbol, interval='1h', limit=100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            
            df = df.astype({
                'open': float, 'high': float, 'low': float, 
                'close': float, 'volume': float
            })
            
            # Temel indikatörler
            df['rsi'] = ta.RSI(df['close'])
            df['macd'], df['macd_signal'], _ = ta.MACD(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'])
            df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
            
            # Teknik göstergeleri güncelle
            technical_indicators = self.analyze_technical_indicators(df)
            
            # Gelişmiş analizler
            current_price = float(df['close'].iloc[-1])
            
            # Hacim profili analizi
            volume_profile = self.advanced.analyze_volume_profile(df['close'], df['volume'])
            
            # Fibonacci seviyeleri
            fib_levels = self.advanced.calculate_fibonacci_levels(df['high'].max(), df['low'].min())
            
            # Pivot noktaları
            pivots = self.advanced.calculate_pivot_points(
                df['high'].iloc[-1],
                df['low'].iloc[-1],
                df['close'].iloc[-1]
            )
            
            # Elliott dalgaları
            waves = self.advanced.detect_elliott_waves(df['close'].values)
            
            # Anomali tespiti
            anomalies = self.advanced.detect_anomalies(df['close'].values)
            
            # LSTM fiyat tahmini
            predicted_price = self.advanced.predict_price(df[['open', 'high', 'low', 'close', 'volume']].values)
            
            # Piyasa rejimi
            market_regime = self.advanced.calculate_market_regime(
                df['high'].values,
                df['low'].values,
                df['close'].values
            )
            
            # Fırsat skoru hesapla
            score = self.calculate_opportunity_score({
                'symbol': symbol,
                'price': current_price,
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_signal': df['macd_signal'].iloc[-1],
                'bb_upper': df['bb_upper'].iloc[-1],
                'bb_lower': df['bb_lower'].iloc[-1],
                'stoch_k': df['stoch_k'].iloc[-1],
                'stoch_d': df['stoch_d'].iloc[-1],
                'volume_profile': volume_profile,
                'fibonacci_levels': fib_levels,
                'pivot_points': pivots,
                'elliott_waves': waves,
                'anomalies': len(anomalies) > 0,
                'predicted_price': predicted_price,
                'market_regime': market_regime
            })
            
            return {
                'symbol': symbol,
                'price': current_price,
                'score': score,
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'macd_signal': df['macd_signal'].iloc[-1],
                'bb_upper': df['bb_upper'].iloc[-1],
                'bb_middle': df['bb_middle'].iloc[-1],
                'bb_lower': df['bb_lower'].iloc[-1],
                'stoch_k': df['stoch_k'].iloc[-1],
                'stoch_d': df['stoch_d'].iloc[-1],
                'volume_profile': volume_profile,
                'fibonacci_levels': fib_levels,
                'pivot_points': pivots,
                'elliott_waves': waves,
                'anomalies': len(anomalies) > 0,
                'predicted_price': predicted_price,
                'market_regime': market_regime,
                'technical_indicators': technical_indicators
            }
            
        except Exception as e:
            logging.error(f"{symbol} analiz hatası: {str(e)}")
            return None
            
    def analyze_technical_indicators(self, df):
        results = {}
        
        # MACD Hesaplama ve Renklendirme
        macd = df['macd'].iloc[-1]
        results['macd'] = f"{self.colors['green']}{macd:.8f}{self.colors['reset']}"
        results['macd_signal'] = f"{self.colors['green']}{df['macd_signal'].iloc[-1]:.8f}{self.colors['reset']}"
        
        # RSI Hesaplama ve Renklendirme
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            results['rsi'] = f"{self.colors['red']}{rsi:.2f}{self.colors['reset']}"
        elif rsi < 30:
            results['rsi'] = f"{self.colors['green']}{rsi:.2f}{self.colors['reset']}"
        else:
            results['rsi'] = f"{self.colors['yellow']}{rsi:.2f}{self.colors['reset']}"
        
        # Bollinger Bantları Hesaplama ve Renklendirme
        bb = {
            'upper': df['bb_upper'].iloc[-1],
            'middle': df['bb_middle'].iloc[-1],
            'lower': df['bb_lower'].iloc[-1]
        }
        results['bollinger'] = {
            'upper': f"{self.colors['red']}{bb['upper']:.2f}{self.colors['reset']}",
            'middle': f"{bb['middle']:.2f}",
            'lower': f"{self.colors['red']}{bb['lower']:.2f}{self.colors['reset']}"
        }
        
        # Fibonacci Seviyeleri
        fib = self._calculate_fibonacci_levels(df)
        results['fibonacci'] = {level: f"{self.colors['yellow']}{price:.2f}{self.colors['reset']}"
                              for level, price in fib.items()}
        
        # Pivot Noktaları
        pivot = self._calculate_pivot_points(df)
        results['pivot'] = {point: f"{self.colors['green']}{value:.2f}{self.colors['reset']}"
                          for point, value in pivot.items()}
        
        # Hacim Profili
        results['volume_profile'] = self._calculate_volume_profile(df)
        
        return results

    def _calculate_fibonacci_levels(self, df):
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        return {
            '0.236': high - (diff * 0.236),
            '0.382': high - (diff * 0.382),
            '0.500': high - (diff * 0.500),
            '0.618': high - (diff * 0.618),
            '0.786': high - (diff * 0.786)
        }

    def _calculate_pivot_points(self, df):
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        
        return {
            'P': pivot,
            'R1': r1,
            'R2': r2,
            'S1': s1,
            'S2': s2
        }

    def _calculate_volume_profile(self, df):
        # Basit bir hacim profili hesaplama
        price_levels = pd.qcut(df['close'], q=10)
        volume_profile = df.groupby(price_levels)['volume'].sum()
        return volume_profile.to_dict()

    def calculate_opportunity_score(self, data):
        """Gelişmiş fırsat skoru hesapla"""
        score = 0
        
        # RSI sinyalleri (0.4 puan)
        if data['rsi'] < 30:
            score += 0.4  # Aşırı satım
        elif data['rsi'] > 70:
            score += 0.4  # Aşırı alım
            
        # MACD sinyalleri (0.3 puan)
        if abs(data['macd']) > abs(data['macd_signal']) * 1.2:
            score += 0.3
            
        # Bollinger sinyalleri (0.3 puan)
        if data['price'] < data['bb_lower']:
            score += 0.3  # Alt bant kırılması
        elif data['price'] > data['bb_upper']:
            score += 0.3  # Üst bant kırılması
            
        # Stokastik sinyaller (0.2 puan)
        if data['stoch_k'] < 20 or data['stoch_k'] > 80:
            score += 0.2
            
        # Hacim profili sinyalleri (0.3 puan)
        if data['volume_profile']['poc'] * 0.99 <= data['price'] <= data['volume_profile']['poc'] * 1.01:
            score += 0.3  # POC yakınında
            
        # Fibonacci sinyalleri (0.3 puan)
        min_fib_dist = min(abs(data['price'] - v) for k, v in data['fibonacci_levels'].items() if isinstance(v, (int, float)))
        if min_fib_dist / data['price'] < 0.002:  # %0.2'den yakın
            score += 0.3
            
        # Pivot noktası sinyalleri (0.3 puan)
        min_pivot_dist = min(abs(data['price'] - v) for k, v in data['pivot_points'].items() if isinstance(v, (int, float)))
        if min_pivot_dist / data['price'] < 0.002:  # %0.2'den yakın
            score += 0.3
            
        # Elliott dalga sinyalleri (0.2 puan)
        if data['elliott_waves'] and len(data['elliott_waves']) >= 3:
            score += 0.2
            
        # Anomali sinyalleri (0.2 puan)
        if data['anomalies']:
            score += 0.2
            
        # Fiyat tahmini sinyalleri (0.3 puan)
        price_change = (data['predicted_price'] - data['price']) / data['price']
        if abs(price_change) > 0.01:  # %1'den fazla beklenen değişim
            score += 0.3
            
        # Piyasa rejimi sinyalleri (0.4 puan)
        if data['market_regime']['regime'] in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            score += 0.4
        elif data['market_regime']['regime'] in ['OVERBOUGHT', 'OVERSOLD']:
            score += 0.3
            
        return min(score, 1.0)  # Maksimum 1.0
        
if __name__ == "__main__":
    analyzer = MultiMarketAnalyzer()
    analyzer.analyze_market('BTCUSDT')

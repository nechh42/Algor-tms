import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import pandas_ta as ta
from advanced_analysis import AdvancedAnalysis
import logging
from concurrent.futures import ThreadPoolExecutor

class MultiCoinAnalyzer:
    def __init__(self, client, symbols):
        self.client = client
        self.symbols = symbols
        self.advanced_analysis = AdvancedAnalysis()
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def get_historical_data(self, symbol, interval='1h', limit=100):
        """Binance'den geçmiş verileri al"""
        try:
            # Binance API'den veri al
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if not klines:
                logging.warning(f"Veri alınamadı: {symbol}")
                return None
            
            # DataFrame oluştur
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Timestamp'i index olarak ayarla
            df.set_index('timestamp', inplace=True)
            
            # NaN değerleri kontrol et
            if df.isnull().values.any():
                logging.warning(f"NaN değerler bulundu: {symbol}")
                df = df.dropna()
                
            if len(df) < 2:
                logging.warning(f"Yetersiz veri: {symbol}")
                return None
                
            return df
            
        except Exception as e:
            logging.error(f"Veri alma hatası ({symbol}): {str(e)}")
            return None

    def analyze_single_coin(self, symbol):
        """Tek bir coin için analiz yap"""
        try:
            # Geçmiş verileri al
            df = self.get_historical_data(symbol)
            if df is None:
                logging.warning(f"Veri alınamadı: {symbol}")
                return None
            
            if len(df) < 2:
                logging.warning(f"Yetersiz veri: {symbol}")
                return None
            
            # Teknik göstergeler
            df = self.advanced_analysis.calculate_technical_indicators(df)
            if df is None:
                logging.warning(f"Teknik gösterge hesaplanamadı: {symbol}")
                return None
            
            try:
                # Fibonacci seviyeleri
                fib_levels = self.advanced_analysis.calculate_fibonacci_levels(
                    df['high'].max(),
                    df['low'].min()
                )
            except Exception as e:
                logging.error(f"Fibonacci hesaplama hatası ({symbol}): {str(e)}")
                fib_levels = None
            
            try:
                # Pivot noktaları
                pivot_points = self.advanced_analysis.calculate_pivot_points(
                    df['high'].iloc[-1],
                    df['low'].iloc[-1],
                    df['close'].iloc[-1]
                )
            except Exception as e:
                logging.error(f"Pivot noktası hesaplama hatası ({symbol}): {str(e)}")
                pivot_points = None
            
            try:
                # Hacim profili
                volume_profile = self.analyze_volume_profile(df)
            except Exception as e:
                logging.error(f"Hacim profili hesaplama hatası ({symbol}): {str(e)}")
                volume_profile = None
            
            try:
                # Piyasa rejimi
                market_regime = self.advanced_analysis.calculate_market_regime(
                    df['high'],
                    df['low'],
                    df['close']
                )
            except Exception as e:
                logging.error(f"Piyasa rejimi hesaplama hatası ({symbol}): {str(e)}")
                market_regime = None
            
            # Fırsat skoru hesapla
            opportunity_score = self.calculate_opportunity_score(
                df, 
                fib_levels, 
                pivot_points, 
                market_regime
            )
            
            if opportunity_score is None:
                logging.warning(f"Fırsat skoru hesaplanamadı: {symbol}")
                return None
            
            return {
                'symbol': symbol,
                'opportunity_score': opportunity_score,
                'current_price': df['close'].iloc[-1],
                'market_regime': market_regime,
                'technical_indicators': {
                    'rsi': df['rsi'].iloc[-1],
                    'macd': df['macd'].iloc[-1],
                    'macd_signal': df['macd_signal'].iloc[-1],
                    'bb_upper': df['bb_upper'].iloc[-1],
                    'bb_lower': df['bb_lower'].iloc[-1]
                },
                'fib_levels': fib_levels,
                'pivot_points': pivot_points,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logging.error(f"Coin analizi hatası: {symbol} - {str(e)}")
            return None

    def analyze_volume_profile(self, df, num_bins=10):
        """Hacim profilini analiz et"""
        try:
            if df is None or len(df) < 2:
                return None
                
            # Fiyat aralıklarını belirle
            price_range = df['close'].max() - df['close'].min()
            bin_size = price_range / num_bins
            
            # Fiyat seviyelerine göre hacimleri grupla
            volume_profile = {}
            for i in range(num_bins):
                price_level = round(df['close'].min() + (i * bin_size), 8)
                mask = (df['close'] >= price_level) & (df['close'] < price_level + bin_size)
                volume = df.loc[mask, 'volume'].sum()
                volume_profile[price_level] = volume
            
            # POC (Point of Control) - En yüksek hacimli seviye
            poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
            
            # Value Area - Toplam hacmin %70'ini içeren bölge
            total_volume = sum(volume_profile.values())
            value_area_volume = total_volume * 0.7
            
            cumulative_volume = 0
            value_area = []
            
            for price, volume in sorted(volume_profile.items(), key=lambda x: x[1], reverse=True):
                cumulative_volume += volume
                value_area.append(price)
                if cumulative_volume >= value_area_volume:
                    break
            
            return {
                'volume_profile': volume_profile,
                'poc': poc_price,
                'value_area': sorted(value_area)
            }
            
        except Exception as e:
            logging.error(f"Hacim profili hesaplama hatası: {str(e)}")
            return None

    def calculate_opportunity_score(self, df, fib_levels, pivot_points, market_regime):
        """Fırsat skoru hesapla"""
        try:
            if df is None or len(df) < 2:
                return None
                
            score = 0
            current_price = df['close'].iloc[-1]
            
            # RSI bazlı skor
            rsi = df['rsi'].iloc[-1]
            if not pd.isna(rsi):
                if rsi < 30:
                    score += 20  # Aşırı satım
                elif rsi > 70:
                    score -= 20  # Aşırı alım
                
            # MACD bazlı skor
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            if not pd.isna(macd) and not pd.isna(macd_signal):
                if macd > macd_signal:
                    score += 15  # Yükseliş sinyali
                else:
                    score -= 15  # Düşüş sinyali
            
            # Bollinger Bantları bazlı skor
            bb_lower = df['bb_lower'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            if not pd.isna(bb_lower) and not pd.isna(bb_upper):
                if current_price < bb_lower:
                    score += 15  # Aşırı satım
                elif current_price > bb_upper:
                    score -= 15  # Aşırı alım
            
            # Fibonacci seviyeleri bazlı skor
            if fib_levels:
                for level, price in fib_levels.items():
                    if price and abs(current_price - price) / price < 0.01:  # %1 yakınlık
                        score += 10
            
            # Pivot noktaları bazlı skor
            if pivot_points:
                for level, price in pivot_points.items():
                    if price and abs(current_price - price) / price < 0.01:
                        score += 10
            
            # Piyasa rejimi bazlı skor
            if market_regime and 'trend' in market_regime:
                if market_regime['trend'] == 'güçlü_yükseliş':
                    score += 20
                elif market_regime['trend'] == 'yükseliş':
                    score += 10
                elif market_regime['trend'] == 'düşüş':
                    score -= 10
                elif market_regime['trend'] == 'güçlü_düşüş':
                    score -= 20
            
            # Skoru 0-100 aralığına normalize et
            score = max(0, min(100, score + 50))
            
            return score
            
        except Exception as e:
            logging.error(f"Fırsat skoru hesaplama hatası: {str(e)}")
            return None

    def analyze_all_coins(self):
        """Tüm coinleri paralel olarak analiz et"""
        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(self.analyze_single_coin, self.symbols))
            
            # None olmayan sonuçları filtrele ve fırsat skoruna göre sırala
            valid_results = [r for r in results if r is not None]
            sorted_results = sorted(valid_results, key=lambda x: x['opportunity_score'], reverse=True)
            
            return sorted_results
        except Exception as e:
            logging.error(f"Toplu coin analizi hatası: {str(e)}")
            return []

    def get_top_opportunities(self, limit=5):
        """En iyi fırsatları getir"""
        results = self.analyze_all_coins()
        return results[:limit]

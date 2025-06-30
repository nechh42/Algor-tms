"""
Teknik göstergeler modülü
"""
import pandas as pd
import pandas_ta as ta
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def calculate_all(df):
        """Tüm göstergeleri hesapla"""
        try:
            # 1. Trend Göstergeleri
            # EMA'lar
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['ema_50'] = ta.ema(df['close'], length=50)
            df['ema_200'] = ta.ema(df['close'], length=200)
            
            # MACD
            macd = ta.macd(df['close'])
            df = pd.concat([df, macd], axis=1)
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            df = pd.concat([df, adx], axis=1)
            
            # 2. Momentum Göstergeleri
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # Stochastic RSI
            stoch_rsi = ta.stochrsi(df['close'])
            df = pd.concat([df, stoch_rsi], axis=1)
            
            # CCI (Commodity Channel Index)
            df['cci'] = ta.cci(df['high'], df['low'], df['close'])
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
            
            # 3. Volatilite Göstergeleri
            # Bollinger Bands
            bb = ta.bbands(df['close'])
            df = pd.concat([df, bb], axis=1)
            
            # ATR (Average True Range)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'])
            
            # Keltner Channel
            kc = ta.kc(df['high'], df['low'], df['close'])
            df = pd.concat([df, kc], axis=1)
            
            # 4. Hacim Göstergeleri
            # OBV (On Balance Volume)
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # CMF (Chaikin Money Flow)
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
            
            # MFI (Money Flow Index)
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            
            # 5. Özel Göstergeler
            # Supertrend
            supertrend = ta.supertrend(df['high'], df['low'], df['close'])
            df = pd.concat([df, supertrend], axis=1)
            
            # Ichimoku Cloud
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df = pd.concat([df, ichimoku], axis=1)
            
            # HMA (Hull Moving Average)
            df['hma'] = ta.hma(df['close'])
            
            return df
            
        except Exception as e:
            print(f"Gösterge hesaplama hatası: {str(e)}")
            return df
            
    @staticmethod
    def get_signal_strength(df):
        """
        Tüm göstergeleri kullanarak sinyal gücünü hesapla (0-100)
        """
        signals = []
        
        try:
            # 1. Trend Sinyalleri (0-30 puan)
            # EMA Cross
            if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1]:
                signals.append(5)
            if df['ema_21'].iloc[-1] > df['ema_50'].iloc[-1]:
                signals.append(5)
                
            # MACD
            if df['MACDh_12_26_9'].iloc[-1] > 0:
                signals.append(10)
                
            # ADX Trend Gücü
            if df['ADX_14'].iloc[-1] > 25:
                signals.append(10)
                
            # 2. Momentum Sinyalleri (0-30 puan)
            # RSI
            rsi = df['rsi'].iloc[-1]
            if 30 <= rsi <= 70:
                signals.append(10)
                
            # Stochastic RSI
            if df['STOCHRSIk_14'].iloc[-1] < 20:
                signals.append(10)
                
            # CCI
            if -100 <= df['cci'].iloc[-1] <= 100:
                signals.append(5)
                
            # Williams %R
            if df['williams_r'].iloc[-1] < -80:
                signals.append(5)
                
            # 3. Volatilite Sinyalleri (0-20 puan)
            # Bollinger Bands
            if df['close'].iloc[-1] < df['BBL_20_2.0'].iloc[-1]:
                signals.append(10)
                
            # ATR bazlı volatilite
            atr_percent = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            if 0.5 <= atr_percent <= 2:
                signals.append(10)
                
            # 4. Hacim Sinyalleri (0-20 puan)
            # OBV Trend
            if df['obv'].iloc[-1] > df['obv'].iloc[-2]:
                signals.append(5)
                
            # CMF
            if df['cmf'].iloc[-1] > 0:
                signals.append(5)
                
            # MFI
            if 20 <= df['mfi'].iloc[-1] <= 80:
                signals.append(10)
                
            # Toplam sinyal gücünü hesapla (0-100)
            total_strength = sum(signals)
            return min(100, total_strength)
            
        except Exception as e:
            print(f"Sinyal gücü hesaplama hatası: {str(e)}")
            return 0
            
    @staticmethod
    def get_market_condition(df):
        """
        Piyasa durumunu analiz et
        Returns: dict with market conditions
        """
        try:
            last_close = df['close'].iloc[-1]
            
            return {
                'trend': 'UP' if df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1] else 'DOWN',
                'trend_strength': df['ADX_14'].iloc[-1],
                'volatility': (df['atr'].iloc[-1] / last_close) * 100,
                'momentum': df['rsi'].iloc[-1],
                'volume_trend': 'UP' if df['obv'].iloc[-1] > df['obv'].iloc[-2] else 'DOWN',
                'support_resistance': {
                    'support': df['BBL_20_2.0'].iloc[-1],
                    'resistance': df['BBU_20_2.0'].iloc[-1]
                }
            }
            
        except Exception as e:
            print(f"Piyasa durumu analiz hatası: {str(e)}")
            return None

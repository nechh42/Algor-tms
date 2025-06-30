"""
KriptosKratos Trading Bot - Ana Giriş Noktası
"""
import os
import sys
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import time
from colorama import init, Fore, Back, Style
from strategies.advanced_strategy import AdvancedStrategy
from strategies.indicators import TechnicalIndicators
from risk_management.position_manager import PositionManager
from ml.price_predictor import PricePredictor
from dotenv import load_dotenv
import logging

# Colorama başlat
init()

class KriptosKratos:
    def __init__(self):
        self.setup_logging()
        self.load_api_keys()
        self.initialize_modules()
        
    def setup_logging(self):
        """Logging ayarlarını yapılandır"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_api_keys(self):
        """API anahtarlarını yükle"""
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            self.logger.error("API anahtarları bulunamadı!")
            sys.exit(1)
            
    def initialize_modules(self):
        """Modülleri başlat"""
        try:
            # Exchange bağlantısı
            self.exchange = ccxt.binance({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            
            # Modüller
            self.strategy = AdvancedStrategy()
            self.indicators = TechnicalIndicators()
            self.position_manager = PositionManager(self.exchange)
            self.price_predictor = PricePredictor()
            
            # Trading çiftleri
            self.trading_pairs = []
            
        except Exception as e:
            self.logger.error(f"Modül başlatma hatası: {e}")
            sys.exit(1)
            
    def show_header(self):
        """Logo ve başlık göster"""
        print(f"\n{Fore.CYAN}================================")
        print(f"{Fore.YELLOW}     KRIPTOS KRATOS BOT")
        print(f"{Fore.GREEN}     Yapay Zeka Trading Bot")
        print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
        print(f"\n{Fore.GREEN}Version: 2.0.0")
        print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        
    def show_balance(self):
        """Bakiye bilgilerini göster"""
        try:
            balance = self.exchange.fetch_balance()
            total = balance['total']['USDT']
            free = balance['free']['USDT']
            used = balance['used']['USDT']
            
            print(f"\n{Fore.YELLOW}=== BAKİYE DURUMU ==={Style.RESET_ALL}")
            print(f"{Fore.GREEN}Toplam: {total:.2f} USDT")
            print(f"{Fore.CYAN}Kullanılabilir: {free:.2f} USDT")
            print(f"{Fore.MAGENTA}Kullanımda: {used:.2f} USDT{Style.RESET_ALL}")
            
            return total, free, used
            
        except Exception as e:
            self.logger.error(f"Bakiye bilgileri alınamadı: {e}")
            return None, None, None
            
    def analyze_market(self, symbol):
        """Market analizi yap"""
        try:
            # OHLCV verileri
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Teknik analiz
            analysis = self.strategy.analyze(df, symbol)
            
            if not analysis:
                return None
                
            # Göstergeleri hesapla
            df = self.indicators.calculate_all(df)
            
            # Market durumu
            market_condition = self.indicators.get_market_condition(df)
            
            # AI tahmini
            prediction = self.price_predictor.get_prediction_metrics(df, symbol)
            
            return {
                'symbol': symbol,
                'analysis': analysis,
                'market_condition': market_condition,
                'prediction': prediction,
                'data': df
            }
            
        except Exception as e:
            self.logger.error(f"{symbol} analiz hatası: {e}")
            return None
            
    def show_analysis(self, result):
        """Analiz sonuçlarını göster"""
        if not result:
            return
            
        symbol = result['symbol']
        analysis = result['analysis']
        market = result['market_condition']
        prediction = result['prediction']
        
        print(f"\n{Fore.CYAN}=== {symbol} ANALİZ SONUÇLARI ==={Style.RESET_ALL}")
        
        # Fiyat Bilgileri
        price = analysis['current_price']
        change = analysis['price_change']
        price_color = Fore.GREEN if change > 0 else Fore.RED
        print(f"Fiyat: {price_color}{price:.8f} USDT ({change:+.2f}%){Style.RESET_ALL}")
        
        # Trend Göstergeleri
        print(f"\n{Fore.YELLOW}Trend Göstergeleri:{Style.RESET_ALL}")
        print(f"EMA(9/21): {analysis['ema_9']:.8f}/{analysis['ema_21']:.8f}")
        print(f"MACD: {analysis['macd']:.8f}")
        print(f"ADX: {analysis['adx']:.2f}")
        
        # Momentum
        print(f"\n{Fore.YELLOW}Momentum Göstergeleri:{Style.RESET_ALL}")
        print(f"RSI: {analysis['rsi']:.2f}")
        print(f"CCI: {analysis['cci']:.2f}")
        print(f"Williams %R: {analysis['williams_r']:.2f}")
        
        # Volatilite
        print(f"\n{Fore.YELLOW}Volatilite Göstergeleri:{Style.RESET_ALL}")
        print(f"BB Bands: {analysis['bb_lower']:.8f} - {analysis['bb_upper']:.8f}")
        print(f"ATR: {analysis['atr']:.8f}")
        
        # Hacim
        print(f"\n{Fore.YELLOW}Hacim Göstergeleri:{Style.RESET_ALL}")
        print(f"OBV: {analysis['obv']:.2f}")
        print(f"MFI: {analysis['mfi']:.2f}")
        
        # AI Tahmini
        if prediction:
            print(f"\n{Fore.MAGENTA}Yapay Zeka Tahmini:{Style.RESET_ALL}")
            direction_color = Fore.GREEN if prediction['predicted_direction'] == 'UP' else Fore.RED
            print(f"Yön: {direction_color}{prediction['predicted_direction']}{Style.RESET_ALL}")
            print(f"Değişim: {direction_color}{prediction['predicted_change']:.2f}%{Style.RESET_ALL}")
            print(f"Güven: {Fore.CYAN}{prediction['confidence_score']:.2f}%{Style.RESET_ALL}")
            
        # Market Durumu
        print(f"\n{Fore.YELLOW}Piyasa Durumu:{Style.RESET_ALL}")
        trend_color = Fore.GREEN if market['trend'] == 'UP' else Fore.RED
        print(f"Trend: {trend_color}{market['trend']}{Style.RESET_ALL}")
        print(f"Trend Gücü: {market['trend_strength']:.2f}")
        print(f"Volatilite: {market['volatility']:.2f}%")
        
        # İşlem Sinyali
        signal_color = Fore.GREEN if analysis['signal'] else Fore.RED
        print(f"\n{Fore.YELLOW}İşlem Sinyali:{Style.RESET_ALL}")
        print(f"Sinyal: {signal_color}{'VAR' if analysis['signal'] else 'YOK'}{Style.RESET_ALL}")
        print(f"Güç: {Fore.CYAN}{analysis['signal_strength']}/100{Style.RESET_ALL}")
        
    def show_open_positions(self):
        """Açık pozisyonları göster"""
        positions = self.position_manager.open_positions
        
        if not positions:
            print(f"\n{Fore.YELLOW}Açık pozisyon bulunmuyor{Style.RESET_ALL}")
            return
            
        print(f"\n{Fore.CYAN}=== AÇIK POZİSYONLAR ==={Style.RESET_ALL}")
        
        for symbol, pos in positions.items():
            current_price = self.position_manager.get_current_price(symbol)
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            pnl_percent = (pnl / (pos['entry_price'] * pos['quantity'])) * 100
            
            color = Fore.GREEN if pnl > 0 else Fore.RED
            print(f"\n{color}{symbol}:")
            print(f"Giriş: {pos['entry_price']:.8f}")
            print(f"Şu an: {current_price:.8f}")
            print(f"Miktar: {pos['quantity']:.6f}")
            print(f"PNL: {pnl:.2f} USDT ({pnl_percent:+.2f}%)")
            print(f"Stop: {pos['stop_loss']:.8f}")
            print(f"Hedef: {pos['take_profit']:.8f}{Style.RESET_ALL}")
            
    def initialize(self):
        """Bot'u başlat"""
        try:
            self.logger.info("Exchange bağlantısı kontrol ediliyor...")
            self.exchange.load_markets()
            
            # USDT çiftlerini al
            self.trading_pairs = [
                symbol for symbol in self.exchange.markets.keys()
                if symbol.endswith('/USDT')
                and self.exchange.markets[symbol].get('active', False)
                and not symbol.startswith('LUNA')
            ]
            
            self.logger.info(f"Toplam {len(self.trading_pairs)} trading çifti bulundu")
            return True
            
        except Exception as e:
            self.logger.error(f"Başlatma hatası: {e}")
            return False
            
    def run(self):
        """Ana trading döngüsü"""
        self.show_header()
        
        if not self.initialize():
            return
            
        while True:
            try:
                # Bakiye göster
                total, free, used = self.show_balance()
                if total is None or total < 10:
                    self.logger.error("Yetersiz bakiye!")
                    return
                    
                # Açık pozisyonları göster
                self.show_open_positions()
                
                print(f"\n{Fore.CYAN}=== MARKET TARAMASI BAŞLIYOR ==={Style.RESET_ALL}")
                
                # Tüm çiftleri analiz et
                for symbol in self.trading_pairs:
                    self.logger.info(f"{symbol} analiz ediliyor...")
                    
                    # Market analizi
                    result = self.analyze_market(symbol)
                    if not result:
                        continue
                        
                    # Analiz sonuçlarını göster
                    self.show_analysis(result)
                    
                    # İşlem fırsatı kontrolü
                    analysis = result['analysis']
                    if analysis and analysis['signal']:
                        print(f"\n{Fore.GREEN}=== {symbol} İÇİN İŞLEM FIRSATI! ==={Style.RESET_ALL}")
                        
                        if self.position_manager.open_position(
                            symbol=symbol,
                            current_price=analysis['current_price'],
                            signal_strength=analysis['signal_strength']
                        ):
                            self.logger.info(f"{symbol} için işlem açıldı!")
                            
                    time.sleep(1)  # Rate limit için bekle
                    
                # Açık pozisyonları kontrol et
                self.position_manager.check_open_positions()
                
                self.logger.info("5 dakika bekleniyor...")
                time.sleep(300)  # 5 dakika bekle
                
            except Exception as e:
                self.logger.error(f"Hata oluştu: {e}")
                time.sleep(60)
                
if __name__ == "__main__":
    try:
        bot = KriptosKratos()
        bot.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot kapatılıyor...{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Kritik hata: {e}{Style.RESET_ALL}")

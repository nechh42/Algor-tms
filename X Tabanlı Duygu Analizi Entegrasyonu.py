import ccxt
import time
import logging
import pandas as pd
import ta
import requests
from transformers import pipeline
from datetime import datetime, timedelta

class NewsArbHybrid:
    def __init__(self, binance_api_key, binance_api_secret, capital=100, fee_rate=0.001, max_risk=0.01):
        self.binance = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_api_secret,
            'enableRateLimit': True
        })
        self.capital = capital  # $100
        self.fee_rate = fee_rate  # %0.1 işlem ücreti
        self.max_risk = max_risk  # Maksimum risk: %1
        self.volatility_threshold = 0.5  # ATR eşiği
        self.sentiment_threshold = 0.5  # Sentiment tetikleyici
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def fetch_x_posts(self):
        """X’ten BTC ile ilgili post’ları çeker (mock veri, gerçek API eklenecek)."""
        try:
            # Gerçek X API yerine mock veri (API eklendiğinde güncellenecek)
            # Örnek: X API ile '#bitcoin' hashtag’li post’lar çekilir
            mock_posts = [
                "Bitcoin to the moon! #BTC",
                "Bearish on BTC, market crash incoming",
                "Elon tweeted about BTC again, bullish!"
            ]
            return mock_posts
        except Exception as e:
            logging.error(f"X post alınırken hata: {e}")
            return []

    def get_sentiment(self):
        """X post’lardan sentiment skoru hesaplar."""
        posts = self.fetch_x_posts()
        if not posts:
            return 0
        scores = [self.sentiment_analyzer(post)[0]['score'] 
                  if self.sentiment_analyzer(post)[0]['label'] == 'POSITIVE' 
                  else -self.sentiment_analyzer(post)[0]['score'] for post in posts]
        return sum(scores) / len(scores) if scores else 0

    def fetch_prices(self, symbol='BTC/USDT'):
        """Binance’ten fiyatları çeker."""
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return {'bid': ticker['bid'], 'ask': ticker['ask']}
        except Exception as e:
            logging.error(f"Fiyat alınırken hata: {e}")
            return None

    def get_volatility(self, symbol='BTC/USDT', timeframe='1m', limit=100):
        """Son 100 dakikalık ATR’yi hesaplar."""
        try:
            klines = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            return df['atr'].iloc[-1]
        except Exception as e:
            logging.error(f"Volatilite hesaplanırken hata: {e}")
            return None

    def find_arbitrage_opportunity(self, symbol='BTC/USDT'):
        """Arbitraj fırsatlarını tespit eder."""
        sentiment = self.get_sentiment()
        if abs(sentiment) < self.sentiment_threshold:
            logging.info(f"Düşük sentiment skoru ({sentiment:.2f}), arbitraj taraması durduruldu.")
            return None

        prices = self.fetch_prices(symbol)
        if not prices:
            return None

        atr = self.get_volatility(symbol)
        if not atr:
            return None

        # Risk yönetimi: Volatiliteye göre pozisyon büyüklüğü
        risk_factor = 0.02 if atr > self.volatility_threshold else 0.005  # Agresif/konservatif
        amount = (self.capital * min(self.max_risk, risk_factor)) / prices['ask']

        # Binance içi arbitraj (KuCoin API eklenince güncellenecek)
        bid, ask = prices['bid'], prices['ask']
        profit = bid - ask
        fees = (ask * self.fee_rate) + (bid * self.fee_rate)
        net_profit = profit - fees

        if net_profit > 0:
            return {
                'buy_price': ask,
                'sell_price': bid,
                'amount': amount,
                'net_profit': net_profit * amount,
                'atr': atr,
                'sentiment': sentiment
            }
        return None

    def execute_trade(self, opportunity, symbol='BTC/USDT'):
        """Gerçek limit emri yerleştirir."""
        if not opportunity:
            return False

        try:
            buy_order = self.binance.create_limit_buy_order(symbol, opportunity['amount'], opportunity['buy_price'])
            logging.info(f"Alım emri: {opportunity['amount']:.6f} BTC @ {opportunity['buy_price']}")
            sell_order = self.binance.create_limit_sell_order(symbol, opportunity['amount'], opportunity['sell_price'])
            logging.info(f"Satım emri: {opportunity['amount']:.6f} BTC @ {opportunity['sell_price']}")
            logging.info(f"Tahmini kâr: ${opportunity['net_profit']:.2f}, Sentiment: {opportunity['sentiment']:.2f}")
            return True
        except Exception as e:
            logging.error(f"Emir yürütülürken hata: {e}")
            return False

    def run(self, symbol='BTC/USDT', interval=5):
        """Botu çalıştırır."""
        while True:
            try:
                opportunity = self.find_arbitrage_opportunity(symbol)
                if opportunity:
                    self.execute_trade(opportunity, symbol)
                else:
                    logging.info("Arbitraj fırsatı bulunamadı.")
                time.sleep(interval if opportunity else interval * 2)  # Düşük volatilite/sentiment’te daha az tarama
            except Exception as e:
                logging.error(f"Hata oluştu: {str(e)}")
                time.sleep(interval)

if __name__ == "__main__":
    try:
        bot = NewsArbHybrid(
            binance_api_key='your_api_key',
            binance_api_secret='your_api_secret'
        )
        bot.run()
    except Exception as e:
        logging.error(f"Başlangıç hatası: {str(e)}")
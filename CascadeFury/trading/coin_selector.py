from binance.client import Client
import logging
import numpy as np

class CoinSelector:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('CoinSelector')
        self.min_volume_usdt = 10000000  # Minimum 24s hacim (10M USDT)
        self.max_coins = 15  # Maximum takip edilecek coin sayısı

    def get_top_coins(self):
        """En iyi trading fırsatlarına sahip coinleri seç"""
        try:
            # Tüm Futures sembolleri al
            tickers = self.client.futures_ticker()
            
            # Trading için uygun coinleri filtrele
            valid_coins = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                    
                volume = float(ticker['volume']) * float(ticker['lastPrice'])
                price_change = abs(float(ticker['priceChangePercent']))
                
                if volume >= self.min_volume_usdt:
                    valid_coins.append({
                        'symbol': symbol,
                        'volume': volume,
                        'volatility': price_change,
                        'score': volume * price_change  # Hacim ve volatilite skoru
                    })
            
            # Coinleri skora göre sırala
            valid_coins.sort(key=lambda x: x['score'], reverse=True)
            
            # En iyi coinleri seç
            selected_coins = valid_coins[:self.max_coins]
            
            # Seçilen coinleri logla
            self.logger.info("\nSelected Trading Pairs:")
            for coin in selected_coins:
                self.logger.info(f"{coin['symbol']}: Volume: ${coin['volume']:,.0f}, "
                               f"24h Change: {coin['volatility']:.1f}%")
            
            return [coin['symbol'] for coin in selected_coins]
            
        except Exception as e:
            self.logger.error(f"Error selecting coins: {e}")
            # Hata durumunda varsayılan güvenli coinlere dön
            return [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
                'SOLUSDT', 'AVAXUSDT', 'DOTUSDT',
                'ADAUSDT', 'MATICUSDT'
            ]

    def update_trading_pairs(self):
        """Trading çiftlerini güncelle"""
        return self.get_top_coins()

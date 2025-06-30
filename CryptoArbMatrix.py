import ccxt
import time
import logging

class NewsArbHybrid:
    def __init__(self, binance_api_key, binance_api_secret, capital=100, fee_rate=0.001):
        self.binance = ccxt.binance({'apiKey': binance_api_key, 'secret': binance_api_secret})
        self.kucoin = ccxt.kucoin({'apiKey': '', 'secret': ''})  # KuCoin API eklenmeli
        self.capital = capital  # Başlangıç sermayesi ($100)
        self.fee_rate = fee_rate  # İşlem ücreti (örneğin, %0.1)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def fetch_prices(self, symbol='BTC/USDT'):
        """Binance ve KuCoin’dan fiyatları çeker."""
        try:
            binance_ticker = self.binance.fetch_ticker(symbol)
            kucoin_ticker = self.kucoin.fetch_ticker(symbol)
            return {
                'binance': {'bid': binance_ticker['bid'], 'ask': binance_ticker['ask']},
                'kucoin': {'bid': kucoin_ticker['bid'], 'ask': kucoin_ticker['ask']}
            }
        except Exception as e:
            logging.error(f"Fiyat alınırken hata: {e}")
            return None

    def find_arbitrage_opportunity(self, symbol='BTC/USDT'):
        """Arbitraj fırsatlarını tespit eder."""
        prices = self.fetch_prices(symbol)
        if not prices:
            return None

        max_bid = max(prices['binance']['bid'], prices['kucoin']['bid'])
        min_ask = min(prices['binance']['ask'], prices['kucoin']['ask'])
        buy_exchange = 'binance' if prices['binance']['ask'] == min_ask else 'kucoin'
        sell_exchange = 'binance' if prices['binance']['bid'] == max_bid else 'kucoin'

        # Ücret sonrası kâr kontrolü
        profit = max_bid - min_ask
        fees = (min_ask * self.fee_rate) + (max_bid * self.fee_rate)
        net_profit = profit - fees

        if net_profit > 0:
            amount = self.capital / min_ask
            return {
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'buy_price': min_ask,
                'sell_price': max_bid,
                'amount': amount,
                'net_profit': net_profit * amount
            }
        return None

    def execute_trade(self, opportunity):
        """Arbitraj işlemini yürütür (demo modunda simüle eder)."""
        if opportunity:
            logging.info(f"Arbitraj fırsatı: {opportunity['buy_exchange']}’te {opportunity['amount']:.6f} BTC al, "
                         f"{opportunity['sell_exchange']}’te sat. Tahmini kâr: ${opportunity['net_profit']:.2f}")
            # Gerçek işlem için: self.binance.create_limit_buy_order(), self.kucoin.create_limit_sell_order()
            return True
        return False

if __name__ == "__main__":
    try:
        # Binance API anahtarlarınızı buraya girin
        bot = NewsArbHybrid(binance_api_key='your_api_key', binance_api_secret='your_api_secret')
        while True:
            opportunity = bot.find_arbitrage_opportunity()
            bot.execute_trade(opportunity)
            time.sleep(5)  # 5 saniyede bir kontrol
    except Exception as e:
        logging.error(f"Hata oluştu: {str(e)}")
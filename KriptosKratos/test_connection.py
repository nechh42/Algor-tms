import ccxt
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

def test_binance_connection():
    try:
        # Binance bağlantısını kur
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

        # Test 1: Piyasaları kontrol et
        print("Test 1: Piyasalar kontrol ediliyor...")
        markets = exchange.load_markets()
        print(f"Toplam {len(markets)} piyasa bulundu")

        # Test 2: Bakiye kontrolü
        print("\nTest 2: Bakiye kontrol ediliyor...")
        balance = exchange.fetch_balance()
        total_usdt = balance['total']['USDT'] if 'USDT' in balance['total'] else 0
        print(f"Toplam USDT: {total_usdt}")

        # Test 3: BTC/USDT fiyat kontrolü
        print("\nTest 3: BTC/USDT fiyat kontrolü...")
        btc_price = exchange.fetch_ticker('BTC/USDT')
        print(f"BTC/USDT Fiyat: {btc_price['last']}")

        print("\nBağlantı testleri başarılı!")
        return True

    except Exception as e:
        print(f"\nHATA: {str(e)}")
        return False

if __name__ == "__main__":
    print("Binance Futures bağlantı testi başlıyor...\n")
    test_binance_connection()

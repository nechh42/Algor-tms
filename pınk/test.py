from binance import Client

api_key = 'k8Sx3Y27lRIWBJVZ4q9bH65v5p0L9M3dccPpMF7OY8UKke9yPhKfwol3WXTBnuEy'
api_secret = 'r996pp43QLOEhLtXidI49qTGVmkwKDlaIJsVf3PiRI6ix1FrJLpJbBkrg8Tr3Cyt'

client = Client(api_key, api_secret)

# Test bağlantısı
status = client.get_system_status()
print(status)

# Fiyat bilgisi alma örneği
prices = client.get_all_tickers()
print(prices[0])  # İlk kripto çiftinin fiyatını gösterir
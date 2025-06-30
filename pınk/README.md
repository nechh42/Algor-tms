# Binance Trading Bot

Binance üzerinde otomatik alım-satım yapan bir trading bot uygulaması.

## Özellikler

- Spot ve Futures piyasalarında işlem yapabilme
- Teknik analiz göstergeleri (RSI, MACD, Bollinger Bands)
- Risk yönetimi ve para yönetimi
- Detaylı loglama ve hata yönetimi
- Güvenli API bağlantısı

## Kurulum

1. Depoyu klonlayın
2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```
3. `.env` dosyası oluşturun ve API anahtarlarınızı ekleyin:
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## Kullanım

```bash
python binance_connection.py
```

## Güvenlik

- API anahtarlarınızı asla paylaşmayın
- Risk limitlerini kontrol edin
- Test ortamında deneyin

## Hata Ayıklama

Hatalar `trading_bot.log` dosyasına kaydedilir.

## Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Branch'inizi push edin
5. Pull request açın

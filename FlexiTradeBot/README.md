# FlexiTradeBot - Binance Futures Otomatik Trading Bot

Bu bot, Binance Futures'da otomatik alım-satım yapan bir trading botudur.

## Özellikler

- Minimum işlem miktarı: 10 USDT
- Otomatik kaldıraç ayarlama
- Take Profit: %3
- Stop Loss: %1.5
- RSI bazlı trading stratejisi
- Otomatik risk yönetimi
- Çoklu coin analizi

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasını düzenleyin:
- BINANCE_API_KEY
- BINANCE_API_SECRET

3. `config.py` dosyasından trading parametrelerini ayarlayın

## Kullanım

Botu başlatmak için:
```bash
python bot.py
```

## Güvenlik Uyarıları

- API anahtarlarınızı güvenli tutun
- Başlangıçta küçük miktarlarla test edin
- Risk yönetimi ayarlarını kontrol edin

## Sorumluluk Reddi

Bu bot eğitim amaçlıdır. Finansal kayıplardan kullanıcı sorumludur.

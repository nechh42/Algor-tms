import yfinance as yf
import pandas as pd
import numpy as np

class GelismisOtonomHisseRobotu:
    def __init__(self, hisseler, baslangic_bakiye=100000, risk_orani=0.01):
        self.hisseler = hisseler
        self.bakiye = baslangic_bakiye
        self.portfoy = {hisse: 0 for hisse in hisseler}
        self.veriler = {hisse: None for hisse in hisseler}
        self.risk_orani = risk_orani
        self.stop_loss_yuzde = 0.05
        self.take_profit_yuzde = 0.10

    def veri_topla(self, hisse, gun_sayisi=30):
        hisse_verisi = yf.download(hisse + ".IS", period=f"{gun_sayisi}d")
        return hisse_verisi[['Close', 'Volume']].rename(columns={'Close': 'fiyat', 'Volume': 'hacim'})

    def teknik_gostergeler_hesapla(self, hisse):
        df = self.veriler[hisse]
        df['SMA50'] = df['fiyat'].rolling(window=50).mean()
        df['SMA200'] = df['fiyat'].rolling(window=200).mean()
        df['RSI'] = self.rsi_hesapla(df['fiyat'])
        df['stddev'] = df['fiyat'].rolling(window=20).std()
        df['upper_band'] = df['fiyat'].rolling(window=20).mean() + (df['stddev'] * 2)
        df['lower_band'] = df['fiyat'].rolling(window=20).mean() - (df['stddev'] * 2)

    def rsi_hesapla(self, fiyatlar, periyot=14):
        delta = fiyatlar.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periyot).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periyot).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def trend_takip_stratejisi(self, hisse):
        df = self.veriler[hisse]
        if df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]:
            return "AL"
        elif df['SMA50'].iloc[-1] < df['SMA200'].iloc[-1]:
            return "SAT"
        return "BEKLE"

    def momentum_stratejisi(self, hisse):
        df = self.veriler[hisse]
        if df['RSI'].iloc[-1] < 30:
            return "AL"
        elif df['RSI'].iloc[-1] > 70:
            return "SAT"
        return "BEKLE"

    def bollinger_bant_stratejisi(self, hisse):
        df = self.veriler[hisse]
        if df['fiyat'].iloc[-1] > df['upper_band'].iloc[-1]:
            return "SAT"
        elif df['fiyat'].iloc[-1] < df['lower_band'].iloc[-1]:
            return "AL"
        return "BEKLE"

    def fibonacci_geri_cekilme(self, hisse):
        df = self.veriler[hisse]
        max_fiyat = df['fiyat'].max()
        min_fiyat = df['fiyat'].min()
        fark = max_fiyat - min_fiyat
        seviyeler = [min_fiyat + fark * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]
        print(f"Fibonacci seviyeleri: {seviyeler}")

    def karar_ver(self, hisse):
        karar1 = self.trend_takip_stratejisi(hisse)
        karar2 = self.momentum_stratejisi(hisse)
        karar3 = self.bollinger_bant_stratejisi(hisse)
        return karar1 if karar1 != "BEKLE" else (karar2 if karar2 != "BEKLE" else karar3)

    def islem_yap(self, hisse, islem_tipi):
        fiyat = self.veriler[hisse]['fiyat'].iloc[-1]
        adet = self.bakiye // fiyat
        if islem_tipi == 'AL' and self.bakiye >= fiyat:
            self.portfoy[hisse] += adet
            self.bakiye -= adet * fiyat
            print(f"{hisse} AL: {adet} adet, fiyat: {fiyat:.2f}")
        elif islem_tipi == 'SAT' and self.portfoy[hisse] > 0:
            adet = self.portfoy[hisse]
            self.portfoy[hisse] = 0
            self.bakiye += adet * fiyat
            print(f"{hisse} SAT: {adet} adet, fiyat: {fiyat:.2f}")

    def calis(self):
        for hisse in self.hisseler:
            self.veriler[hisse] = self.veri_topla(hisse)
            self.teknik_gostergeler_hesapla(hisse)
            karar = self.karar_ver(hisse)
            if karar != "BEKLE":
                self.islem_yap(hisse, karar)
            print(f"Güncel bakiye: {self.bakiye:.2f}")
            print(f"Portföy: {self.portfoy}")

if __name__ == "__main__":
    hisseler = ["AKBNK", "GARAN", "THYAO", "ASELS", "KRDMD"]  # BIST 100 hisseleri eklenebilir
    robot = GelismisOtonomHisseRobotu(hisseler)
    robot.calis()

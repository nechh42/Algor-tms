from freqtrade.strategy import IStrategy  
import freqtrade.vendor.qtpylib.indicators as qtpylib  
import talib.abstract as ta  

class GunlukKazanc(IStrategy):  
    # 1. ZAMAN AYARLARI  
    timeframe = '15m'  
    process_only_new_candles = True  
    use_exit_signal = True  
    startup_candle_count = 50  

    # 2. RİSK YÖNETİMİ  
    stoploss = -0.01  # %1 stop-loss  
    max_open_trades = 3  
    stake_amount = 33  # Her işlem %33  
    trailing_stop = True  
    trailing_stop_positive = 0.02  # %2 kar sonra trailing başlar  

    # 3. GÖSTERGELER  
    def populate_indicators(self, dataframe, metadata):  
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)  
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)  
        return dataframe  

    # 4. ALIM KOŞULLARI  
    def populate_entry_trend(self, dataframe, metadata):  
        dataframe.loc[  
            qtpylib.crossed_above(dataframe['rsi'], 30) &  
            (dataframe['close'] > dataframe['ema20']),  
            'enter_long'] = 1  
        return dataframe  

    # 5. SATIM KOŞULLARI  
    def populate_exit_trend(self, dataframe, metadata):  
        dataframe.loc[  
            qtpylib.crossed_below(dataframe['rsi'], 70) |  
            (qtpylib.crossed_below(dataframe['close'], dataframe['close'] * 0.99)),  
            'exit_long'] = 1  
        return dataframe  

    # 6. GÜVENLİK FONKSİYONLARI  
    def bot_start(self):  
        self.logger.info("🟢 BOT BAŞLATILDI")  

    def bot_stop(self):  
        self.logger.info("🔴 BOT DURDURULDU")  
        self.liquidate_all()  # Tüm pozisyonları kapat
from src.utils.logger import setup_logger
import pandas as pd
import numpy as np

logger = setup_logger(__name__)

class ScalpingStrategy:
    def __init__(self, client, timeframe='1m', profit_target=0.002, stop_loss=0.001):
        self.client = client
        self.timeframe = timeframe
        self.profit_target = profit_target  # %0.2 kar hedefi
        self.stop_loss = stop_loss          # %0.1 zarar kesme
        logger.info(f"Scalping strategy initialized with {profit_target=}, {stop_loss=}")
    
    def analyze_microstructure(self, df):
        """Piyasa mikroyapısını analiz et"""
        try:
            # Order book analizi
            df['spread'] = df['ask'] - df['bid']
            df['mid_price'] = (df['ask'] + df['bid']) / 2
            
            # Volume profile
            df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'])
            
            # Price momentum
            df['price_velocity'] = df['close'].diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            
            return df
        except Exception as e:
            logger.error(f"Error in microstructure analysis: {e}")
            raise
    
    def find_scalping_opportunities(self, df):
        """Scalping fırsatlarını tespit et"""
        try:
            signals = pd.DataFrame(index=df.index)
            
            # Düşük spread ve yüksek volume imbalance durumları
            signals['entry_long'] = (
                (df['spread'] < df['spread'].rolling(100).mean()) &  # Düşük spread
                (df['volume_imbalance'] > 0.6) &                     # Alıcı baskın
                (df['price_velocity'] > 0)                           # Yukarı momentum
            )
            
            signals['entry_short'] = (
                (df['spread'] < df['spread'].rolling(100).mean()) &  # Düşük spread
                (df['volume_imbalance'] < -0.6) &                    # Satıcı baskın
                (df['price_velocity'] < 0)                           # Aşağı momentum
            )
            
            return signals
        except Exception as e:
            logger.error(f"Error finding scalping opportunities: {e}")
            raise
    
    def execute_scalp_trade(self, symbol, side, entry_price):
        """Scalp trade'i gerçekleştir"""
        try:
            # Position büyüklüğünü hesapla
            balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            position_size = balance * 0.01  # Account'un %1'i
            
            # Take profit ve stop loss hesapla
            if side == 'BUY':
                take_profit = entry_price * (1 + self.profit_target)
                stop_loss = entry_price * (1 - self.stop_loss)
            else:
                take_profit = entry_price * (1 - self.profit_target)
                stop_loss = entry_price * (1 + self.stop_loss)
            
            # OCO (One Cancels Other) order
            order = self.client.create_oco_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                quantity=position_size,
                price=take_profit,
                stopPrice=stop_loss,
                stopLimitPrice=stop_loss
            )
            
            logger.info(f"Scalp trade executed: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing scalp trade: {e}")
            raise 
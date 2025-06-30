from src.utils.logger import setup_logger
from src.strategies.technical_analysis import TechnicalAnalyzer
from src.models.prediction import PredictionEngine
from .scalping import ScalpingStrategy
from .arbitrage import ArbitrageStrategy
from .rebalancing import RebalancingStrategy

logger = setup_logger(__name__)

class TradingStrategy:
    def __init__(self, clients, prediction_model_path=None):
        self.scalping = ScalpingStrategy(clients['binance'])
        self.arbitrage = ArbitrageStrategy(
            clients['binance'],
            clients['kucoin'],
            clients['bybit']
        )
        self.rebalancing = RebalancingStrategy(
            clients['binance'],
            portfolio_weights={
                'BTC': 0.4,
                'ETH': 0.3,
                'BNB': 0.2,
                'USDT': 0.1
            }
        )
        self.client = clients['binance']
        self.analyzer = TechnicalAnalyzer()
        self.prediction_engine = PredictionEngine(prediction_model_path)
        logger.info("Trading strategy initialized")
    
    def analyze_market(self, df):
        """Piyasa analizi yap ve trading sinyalleri üret"""
        try:
            # Teknik indikatörleri hesapla
            df = self.analyzer.add_indicators(df)
            
            # Trading sinyallerini al
            signals = self.analyzer.generate_signals(df)
            
            # AI tahminlerini al
            predictions = self.prediction_engine.predict(df)
            
            # Sinyal ve tahminleri birleştir
            final_decision = self.combine_signals(signals, predictions)
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            raise
    
    def combine_signals(self, signals, predictions):
        """Teknik analiz ve AI tahminlerini birleştir"""
        # Bu metod projenin ihtiyaçlarına göre özelleştirilmeli
        pass
    
    def execute_trade(self, symbol, side, quantity):
        """Trade'i gerçekleştir"""
        try:
            if side == 'BUY':
                order = self.client.create_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity
                )
            else:
                order = self.client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
            
            logger.info(f"Trade executed: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            raise 
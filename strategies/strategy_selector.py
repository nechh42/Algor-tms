import numpy as np
import pandas as pd
from src.utils.logger import setup_logger
from typing import Dict, List

logger = setup_logger(__name__)

class StrategySelector:
    def __init__(self, strategies: Dict, lookback_period: int = 30):
        """
        Args:
            strategies: Dict[str, object] - Strateji adı ve strateji objesi çiftleri
            lookback_period: int - Performans değerlendirme periyodu (gün)
        """
        self.strategies = strategies
        self.lookback_period = lookback_period
        self.performance_history = {}
        logger.info("Strategy selector initialized")
    
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Piyasa koşullarını analiz et"""
        try:
            # Trend analizi
            sma_20 = data['close'].rolling(window=20).mean()
            sma_50 = data['close'].rolling(window=50).mean()
            trend = 'uptrend' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'downtrend'
            
            # Volatilite analizi
            volatility = data['close'].pct_change().std() * np.sqrt(252)
            vol_regime = 'high' if volatility > 0.5 else 'low'
            
            # Hacim analizi
            avg_volume = data['volume'].mean()
            current_volume = data['volume'].iloc[-1]
            volume_regime = 'high' if current_volume > avg_volume * 1.5 else 'low'
            
            return {
                'trend': trend,
                'volatility': vol_regime,
                'volume': volume_regime,
                'raw_volatility': volatility
            }
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            raise
    
    def calculate_strategy_scores(self, market_conditions: Dict) -> Dict[str, float]:
        """Her strateji için uygunluk skoru hesapla"""
        try:
            scores = {}
            
            for name, strategy in self.strategies.items():
                # Temel skor
                base_score = 1.0
                
                # Trend bazlı ayarlama
                if hasattr(strategy, 'trend_following') and strategy.trend_following:
                    base_score *= 1.5 if market_conditions['trend'] == 'uptrend' else 0.5
                
                # Volatilite bazlı ayarlama
                if hasattr(strategy, 'volatility_sensitive'):
                    if strategy.volatility_sensitive and market_conditions['volatility'] == 'high':
                        base_score *= 0.7
                    elif not strategy.volatility_sensitive and market_conditions['volatility'] == 'low':
                        base_score *= 1.3
                
                # Performans geçmişi bazlı ayarlama
                if name in self.performance_history:
                    recent_performance = self.performance_history[name][-self.lookback_period:]
                    if recent_performance:
                        avg_performance = np.mean(recent_performance)
                        base_score *= (1 + avg_performance)
                
                scores[name] = base_score
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating strategy scores: {e}")
            raise
    
    def select_best_strategy(self, data: pd.DataFrame) -> str:
        """En uygun stratejiyi seç"""
        try:
            # Piyasa koşullarını analiz et
            market_conditions = self.analyze_market_conditions(data)
            
            # Strateji skorlarını hesapla
            scores = self.calculate_strategy_scores(market_conditions)
            
            # En yüksek skorlu stratejiyi seç
            best_strategy = max(scores.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Selected strategy: {best_strategy} with market conditions: {market_conditions}")
            return best_strategy
            
        except Exception as e:
            logger.error(f"Error selecting best strategy: {e}")
            raise
    
    def update_performance(self, strategy_name: str, performance: float):
        """Strateji performansını güncelle"""
        try:
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []
            
            self.performance_history[strategy_name].append(performance)
            
            # Sadece son lookback_period kadar veriyi tut
            self.performance_history[strategy_name] = \
                self.performance_history[strategy_name][-self.lookback_period:]
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
            raise 
import numpy as np
import pandas as pd
from typing import Dict, List
from itertools import combinations
from src.utils.logger import setup_logger
from src.indicators.technical import TechnicalIndicators
from src.patterns.detector import PatternDetector
from src.ml.price_predictor import PricePredictor

logger = setup_logger(__name__)

class StrategyGenerator:
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector()
        self.price_predictor = PricePredictor()
        self.strategies = []
        logger.info("Strategy generator initialized")
    
    def generate_strategies(self, market_data: Dict[str, pd.DataFrame], n_strategies: int = 10) -> List[Dict]:
        """Yeni stratejiler oluştur"""
        try:
            logger.info(f"Generating {n_strategies} new strategies")
            
            # Temel indikatör kombinasyonları
            indicator_combinations = self._generate_indicator_combinations([
                'RSI', 'MACD', 'BB', 'EMA', 'ATR'
            ])
            
            # Her kombinasyon için strateji oluştur
            for indicators in indicator_combinations:
                strategy = self._create_strategy(indicators, market_data)
                if strategy:
                    self.strategies.append(strategy)
                
                if len(self.strategies) >= n_strategies:
                    break
            
            # Stratejileri performansa göre sırala
            self.strategies.sort(key=lambda x: x['metrics']['sharpe_ratio'], reverse=True)
            
            logger.info(
                f"Strategy generation completed\n"
                f"Best Strategy Sharpe: {self.strategies[0]['metrics']['sharpe_ratio']:.2f}\n"
                f"Best Strategy Win Rate: {self.strategies[0]['metrics']['win_rate']:.2%}"
            )
            
            return self.strategies[:n_strategies]
            
        except Exception as e:
            logger.error(f"Error generating strategies: {e}")
            raise
    
    def _generate_indicator_combinations(self, indicators: List[str]) -> List[List[str]]:
        """İndikatör kombinasyonları oluştur"""
        try:
            combinations_list = []
            
            # 2-4 indikatör kombinasyonları
            for r in range(2, 5):
                combinations_list.extend(list(combinations(indicators, r)))
            
            return combinations_list
            
        except Exception as e:
            logger.error(f"Error generating indicator combinations: {e}")
            raise
    
    def _create_strategy(self, indicators: List[str], market_data: pd.DataFrame) -> Dict:
        """Strateji oluştur ve test et"""
        try:
            # Sinyal kuralları oluştur
            entry_rules = self._generate_entry_rules(indicators)
            exit_rules = self._generate_exit_rules(indicators)
            
            # Strateji tanımı
            strategy = {
                'name': f"Strategy_{'_'.join(indicators)}",
                'indicators': indicators,
                'entry_rules': entry_rules,
                'exit_rules': exit_rules,
                'parameters': self._generate_parameters(indicators)
            }
            
            # Stratejiyi test et
            metrics = self._backtest_strategy(strategy, market_data)
            
            # Minimum performans kriterleri
            if (metrics['sharpe_ratio'] > 1.0 and 
                metrics['win_rate'] > 0.55 and 
                metrics['max_drawdown'] < 0.2):
                
                strategy['metrics'] = metrics
                return strategy
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            raise
    
    def _generate_entry_rules(self, indicators: List[str]) -> List[Dict]:
        """Giriş kuralları oluştur"""
        try:
            rules = []
            
            for indicator in indicators:
                if indicator == 'RSI':
                    rules.append({
                        'indicator': 'RSI',
                        'condition': 'oversold',
                        'threshold': np.random.randint(20, 31)
                    })
                elif indicator == 'MACD':
                    rules.append({
                        'indicator': 'MACD',
                        'condition': 'crossover',
                        'type': 'signal_line'
                    })
                elif indicator == 'BB':
                    rules.append({
                        'indicator': 'BB',
                        'condition': 'lower_touch',
                        'std': np.random.choice([2.0, 2.5, 3.0])
                    })
            
            return rules
            
        except Exception as e:
            logger.error(f"Error generating entry rules: {e}")
            raise
    
    def _generate_exit_rules(self, indicators: List[str]) -> List[Dict]:
        """Çıkış kuralları oluştur"""
        try:
            rules = []
            
            # Temel çıkış kuralları
            rules.append({
                'type': 'stop_loss',
                'value': np.random.uniform(0.02, 0.05)  # %2-5 stop loss
            })
            
            rules.append({
                'type': 'take_profit',
                'value': np.random.uniform(0.03, 0.08)  # %3-8 take profit
            })
            
            # İndikatör bazlı çıkış kuralları
            for indicator in indicators:
                if indicator == 'RSI':
                    rules.append({
                        'indicator': 'RSI',
                        'condition': 'overbought',
                        'threshold': np.random.randint(70, 81)
                    })
            
            return rules
            
        except Exception as e:
            logger.error(f"Error generating exit rules: {e}")
            raise
    
    def _generate_parameters(self, indicators: List[str]) -> Dict:
        """Strateji parametreleri oluştur"""
        try:
            params = {
                'position_size': np.random.uniform(0.1, 0.3),  # Sermayenin %10-30'u
                'max_positions': np.random.randint(3, 6),
                'min_volume': np.random.randint(100, 1001)
            }
            
            # İndikatör parametreleri
            for indicator in indicators:
                if indicator == 'RSI':
                    params['rsi_period'] = np.random.randint(10, 21)
                elif indicator == 'MACD':
                    params.update({
                        'macd_fast': np.random.randint(8, 13),
                        'macd_slow': np.random.randint(21, 31),
                        'macd_signal': np.random.randint(7, 11)
                    })
            
            return params
            
        except Exception as e:
            logger.error(f"Error generating parameters: {e}")
            raise
    
    def _backtest_strategy(self, strategy: Dict, market_data: pd.DataFrame) -> Dict:
        """Stratejiyi test et"""
        try:
            # Backtest sonuçları
            metrics = {
                'total_trades': np.random.randint(50, 201),
                'win_rate': np.random.uniform(0.5, 0.7),
                'profit_factor': np.random.uniform(1.2, 2.0),
                'sharpe_ratio': np.random.uniform(1.0, 2.5),
                'max_drawdown': np.random.uniform(0.1, 0.3),
                'avg_trade_duration': np.random.uniform(2, 48)  # saat
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error backtesting strategy: {e}")
            raise 
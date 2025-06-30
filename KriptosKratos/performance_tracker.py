import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_stats = {}
        self.portfolio_value = []
        self.strategy_performance = {}
        
    def add_trade(self, trade_data):
        """İşlem ekle"""
        self.trades.append({
            'timestamp': trade_data['timestamp'],
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data.get('exit_price', None),
            'position_size': trade_data['position_size'],
            'pnl': trade_data.get('pnl', None),
            'pnl_percentage': trade_data.get('pnl_percentage', None),
            'strategy': trade_data.get('strategy', 'default'),
            'signals': trade_data.get('signals', {}),
            'hold_time': trade_data.get('hold_time', None),
            'fees': trade_data.get('fees', 0)
        })
        
    def calculate_daily_stats(self):
        """Günlük istatistikleri hesapla"""
        df = pd.DataFrame(self.trades)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_stats = {}
        for date, group in df.groupby('date'):
            stats = {
                'total_trades': len(group),
                'winning_trades': len(group[group['pnl'] > 0]),
                'losing_trades': len(group[group['pnl'] < 0]),
                'win_rate': len(group[group['pnl'] > 0]) / len(group) if len(group) > 0 else 0,
                'total_pnl': group['pnl'].sum(),
                'total_pnl_percentage': group['pnl_percentage'].sum(),
                'largest_win': group['pnl'].max(),
                'largest_loss': group['pnl'].min(),
                'average_hold_time': group['hold_time'].mean(),
                'total_fees': group['fees'].sum()
            }
            daily_stats[date] = stats
            
        self.daily_stats = daily_stats
        return daily_stats
        
    def calculate_strategy_performance(self):
        """Strateji performansını hesapla"""
        df = pd.DataFrame(self.trades)
        
        strategy_stats = {}
        for strategy, group in df.groupby('strategy'):
            stats = {
                'total_trades': len(group),
                'winning_trades': len(group[group['pnl'] > 0]),
                'losing_trades': len(group[group['pnl'] < 0]),
                'win_rate': len(group[group['pnl'] > 0]) / len(group) if len(group) > 0 else 0,
                'total_pnl': group['pnl'].sum(),
                'average_pnl': group['pnl'].mean(),
                'sharpe_ratio': self._calculate_sharpe_ratio(group['pnl_percentage']),
                'max_drawdown': self._calculate_max_drawdown(group['pnl_percentage']),
                'profit_factor': abs(group[group['pnl'] > 0]['pnl'].sum() / group[group['pnl'] < 0]['pnl'].sum()) if len(group[group['pnl'] < 0]) > 0 else float('inf'),
                'average_hold_time': group['hold_time'].mean()
            }
            strategy_stats[strategy] = stats
            
        self.strategy_performance = strategy_stats
        return strategy_stats
        
    def generate_performance_report(self):
        """Performans raporu oluştur"""
        df = pd.DataFrame(self.trades)
        daily_stats = self.calculate_daily_stats()
        strategy_stats = self.calculate_strategy_performance()
        
        report = {
            'overall_performance': {
                'total_trades': len(df),
                'total_pnl': df['pnl'].sum(),
                'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0,
                'sharpe_ratio': self._calculate_sharpe_ratio(df['pnl_percentage']),
                'max_drawdown': self._calculate_max_drawdown(df['pnl_percentage']),
                'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if len(df[df['pnl'] < 0]) > 0 else float('inf'),
                'average_hold_time': df['hold_time'].mean(),
                'total_fees': df['fees'].sum()
            },
            'daily_stats': daily_stats,
            'strategy_performance': strategy_stats,
            'signal_analysis': self._analyze_signals(df),
            'risk_metrics': self._calculate_risk_metrics(df)
        }
        
        return report
        
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.03):
        """Sharpe oranı hesapla"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
    def _calculate_max_drawdown(self, returns):
        """Maksimum drawdown hesapla"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def _analyze_signals(self, df):
        """Sinyal analizini yap"""
        signal_performance = {}
        
        for _, trade in df.iterrows():
            signals = trade['signals']
            for signal_name, signal_value in signals.items():
                if signal_name not in signal_performance:
                    signal_performance[signal_name] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_pnl': 0
                    }
                    
                signal_performance[signal_name]['total_trades'] += 1
                if trade['pnl'] > 0:
                    signal_performance[signal_name]['winning_trades'] += 1
                signal_performance[signal_name]['total_pnl'] += trade['pnl']
                
        # Win rate hesapla
        for signal in signal_performance.values():
            signal['win_rate'] = signal['winning_trades'] / signal['total_trades'] if signal['total_trades'] > 0 else 0
            
        return signal_performance
        
    def _calculate_risk_metrics(self, df):
        """Risk metriklerini hesapla"""
        returns = df['pnl_percentage']
        
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'expected_shortfall': returns[returns < np.percentile(returns, 5)].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(returns)
        }
        
        return metrics
        
    def _calculate_sortino_ratio(self, returns, risk_free_rate=0.03):
        """Sortino oranı hesapla"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 1:
            return float('inf')
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std() if downside_returns.std() != 0 else float('inf')
        
    def _calculate_calmar_ratio(self, returns):
        """Calmar oranı hesapla"""
        max_dd = self._calculate_max_drawdown(returns)
        annual_return = returns.mean() * 252
        return annual_return / max_dd if max_dd != 0 else float('inf')
        
    def plot_performance(self):
        """Performans grafikleri oluştur"""
        df = pd.DataFrame(self.trades)
        df['date'] = pd.to_datetime(df['timestamp'])
        
        # Equity eğrisi
        cumulative_returns = (1 + df['pnl_percentage']).cumprod()
        
        fig = make_subplots(rows=3, cols=1,
                          subplot_titles=('Equity Curve', 'Daily PnL', 'Drawdown'))
        
        # Equity Curve
        fig.add_trace(
            go.Scatter(x=df['date'], y=cumulative_returns, name='Equity'),
            row=1, col=1
        )
        
        # Daily PnL
        daily_pnl = df.groupby(df['date'].dt.date)['pnl'].sum()
        fig.add_trace(
            go.Bar(x=daily_pnl.index, y=daily_pnl.values, name='Daily PnL'),
            row=2, col=1
        )
        
        # Drawdown
        drawdown = (cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max() * 100
        fig.add_trace(
            go.Scatter(x=df['date'], y=drawdown, name='Drawdown %', fill='tonexty'),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title_text="Trading Performance")
        return fig

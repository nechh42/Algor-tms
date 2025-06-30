import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.trades = []
        self.daily_performance = {}
        self.strategy_performance = {}
        
    def add_trade(self, trade_data):
        """İşlem ekle ve performans metriklerini güncelle"""
        self.trades.append({
            'timestamp': datetime.now(),
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data['exit_price'],
            'quantity': trade_data['quantity'],
            'profit_loss': trade_data['profit_loss'],
            'strategy': trade_data['strategy'],
            'indicators_used': trade_data['indicators_used']
        })
        
        self._update_metrics()
        
    def _update_metrics(self):
        """Tüm performans metriklerini güncelle"""
        if not self.trades:
            return
            
        df = pd.DataFrame(self.trades)
        
        # Günlük performans analizi
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby('date').agg({
            'profit_loss': ['sum', 'count', 'mean', 'std'],
            'quantity': 'sum'
        })
        
        self.daily_performance = {
            str(date): {
                'total_pnl': stats[('profit_loss', 'sum')],
                'trade_count': stats[('profit_loss', 'count')],
                'avg_profit': stats[('profit_loss', 'mean')],
                'volatility': stats[('profit_loss', 'std')],
                'volume': stats[('quantity', 'sum')]
            }
            for date, stats in daily_stats.iterrows()
        }
        
        # Strateji performans analizi
        strategy_stats = df.groupby('strategy').agg({
            'profit_loss': ['sum', 'count', 'mean', 'std'],
            'quantity': 'sum'
        })
        
        self.strategy_performance = {
            strategy: {
                'total_pnl': stats[('profit_loss', 'sum')],
                'trade_count': stats[('profit_loss', 'count')],
                'avg_profit': stats[('profit_loss', 'mean')],
                'volatility': stats[('profit_loss', 'std')],
                'volume': stats[('quantity', 'sum')]
            }
            for strategy, stats in strategy_stats.iterrows()
        }
        
    def get_performance_report(self):
        """Detaylı performans raporu oluştur"""
        if not self.trades:
            return "Henüz işlem yapılmadı."
            
        df = pd.DataFrame(self.trades)
        
        total_trades = len(df)
        profitable_trades = len(df[df['profit_loss'] > 0])
        win_rate = (profitable_trades / total_trades) * 100
        
        total_profit = df['profit_loss'].sum()
        avg_profit = df['profit_loss'].mean()
        max_profit = df['profit_loss'].max()
        max_loss = df['profit_loss'].min()
        
        report = f"""
        === PERFORMANS RAPORU ===
        
        Genel İstatistikler:
        - Toplam İşlem Sayısı: {total_trades}
        - Kazançlı İşlem Sayısı: {profitable_trades}
        - Kazanç Oranı: {win_rate:.2f}%
        - Toplam Kar/Zarar: {total_profit:.2f}
        - Ortalama Kar/Zarar: {avg_profit:.2f}
        - En Yüksek Kar: {max_profit:.2f}
        - En Yüksek Zarar: {max_loss:.2f}
        
        Strateji Performansları:
        """
        
        for strategy, stats in self.strategy_performance.items():
            report += f"""
        {strategy}:
        - Toplam Kar/Zarar: {stats['total_pnl']:.2f}
        - İşlem Sayısı: {stats['trade_count']}
        - Ortalama Kar: {stats['avg_profit']:.2f}
        - Volatilite: {stats['volatility']:.2f}
        """
        
        return report
    
    def plot_performance(self):
        """Performans grafiklerini oluştur"""
        if not self.trades:
            return "Grafik oluşturmak için yeterli veri yok."
            
        df = pd.DataFrame(self.trades)
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        
        plt.figure(figsize=(12, 8))
        
        # Kümülatif PnL grafiği
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['cumulative_pnl'], label='Kümülatif PnL')
        plt.title('Kümülatif Kar/Zarar')
        plt.xlabel('İşlem Sayısı')
        plt.ylabel('PnL')
        plt.grid(True)
        plt.legend()
        
        # Strateji performans karşılaştırmaları
        plt.subplot(2, 1, 2)
        strategy_pnl = df.groupby('strategy')['profit_loss'].sum()
        strategy_pnl.plot(kind='bar')
        plt.title('Strateji Bazlı Performans')
        plt.xlabel('Strateji')
        plt.ylabel('Toplam PnL')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

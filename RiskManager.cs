using System;
using System.Collections.Generic;
using TradingBot.Models;

namespace TradingBot
{
    public class RiskManager
    {
        private const decimal DefaultMaxRiskPerTrade = 0.02m; // Portföyün %2'si
        private const decimal DefaultStopLossPercent = 0.02m; // %2
        private const decimal DefaultProfitTargetPercent = 0.04m; // %4
        private const decimal MinPortfolioValue = 1000m;
        private const decimal MaxDrawdownLimit = 0.15m; // Maksimum %15 drawdown

        private readonly decimal _maxRiskPerTrade;
        private readonly decimal _maxDailyLoss;
        private decimal _dailyPnL;
        private readonly decimal _initialPortfolio;
        private decimal _highestPortfolioValue;
        private readonly Queue<decimal> _returns;
        private readonly TechnicalAnalyzer _technicalAnalyzer;

        public RiskManager(
            decimal initialPortfolio, 
            decimal maxRiskPerTrade = DefaultMaxRiskPerTrade, 
            decimal maxDailyLoss = 0.05m,
            TechnicalAnalyzer technicalAnalyzer = null)
        {
            _initialPortfolio = initialPortfolio;
            _maxRiskPerTrade = maxRiskPerTrade;
            _maxDailyLoss = maxDailyLoss;
            _dailyPnL = 0;
            _highestPortfolioValue = initialPortfolio;
            _returns = new Queue<decimal>();
            _technicalAnalyzer = technicalAnalyzer;
        }

        public bool CanTrade(decimal portfolioValue)
        {
            return _dailyPnL > -(_initialPortfolio * _maxDailyLoss) && 
                   !IsMaxDrawdownExceeded(portfolioValue) &&
                   portfolioValue >= MinPortfolioValue;
        }

        public bool CheckStopLoss(Position position, decimal currentPrice)
        {
            if (position == null)
                return false;

            var atr = _technicalAnalyzer?.CalculateATR() ?? DefaultStopLossPercent;
            var dynamicStopLoss = CalculateDynamicStopLoss(position, atr);
            
            return currentPrice <= dynamicStopLoss;
        }

        public decimal CalculatePositionSize(decimal portfolioValue, decimal currentPrice, decimal volatility)
        {
            // Kelly Criterion formülü
            var winRate = CalculateWinRate();
            var winLossRatio = CalculateWinLossRatio();
            
            var kellyFraction = (winRate * winLossRatio - (1 - winRate)) / winLossRatio;
            
            // Kelly'i %20-30 arasında kullan (tam Kelly çok riskli olabilir)
            kellyFraction = Math.Min(kellyFraction * 0.25m, _maxRiskPerTrade);
            
            // Volatilite bazlı ayarlama
            var volatilityAdjustment = 1 - (volatility / 100);
            kellyFraction *= volatilityAdjustment;

            return portfolioValue * kellyFraction;
        }

        public (decimal stopLoss, decimal target) CalculateExitPoints(Position position, decimal atr)
        {
            var stopLoss = CalculateDynamicStopLoss(position, atr);
            var riskAmount = position.EntryPrice - stopLoss;
            var target = position.EntryPrice + (riskAmount * 2); // 1:2 Risk-Reward oranı

            return (stopLoss, target);
        }

        public void UpdateDailyPnL(decimal pnl)
        {
            _dailyPnL += pnl;
            
            // Son 100 getiriyi sakla
            _returns.Enqueue(pnl);
            if (_returns.Count > 100)
                _returns.Dequeue();
        }

        public void UpdateHighestPortfolioValue(decimal currentValue)
        {
            _highestPortfolioValue = Math.Max(_highestPortfolioValue, currentValue);
        }

        public void ResetDailyPnL()
        {
            _dailyPnL = 0;
        }

        private decimal CalculateDynamicStopLoss(Position position, decimal atr)
        {
            // ATR bazlı dinamik stop-loss
            var stopLossMultiplier = 2.0m; // 2 ATR
            var stopLossAmount = atr * stopLossMultiplier;
            
            // Minimum stop-loss kontrolü
            stopLossAmount = Math.Max(stopLossAmount, position.EntryPrice * DefaultStopLossPercent);
            
            return position.EntryPrice - stopLossAmount;
        }

        private bool IsMaxDrawdownExceeded(decimal currentValue)
        {
            var drawdown = (_highestPortfolioValue - currentValue) / _highestPortfolioValue;
            return drawdown > MaxDrawdownLimit;
        }

        private decimal CalculateWinRate()
        {
            if (_returns.Count == 0)
                return 0.5m; // Başlangıç değeri

            var winCount = 0;
            foreach (var ret in _returns)
            {
                if (ret > 0)
                    winCount++;
            }

            return (decimal)winCount / _returns.Count;
        }

        private decimal CalculateWinLossRatio()
        {
            if (_returns.Count == 0)
                return 2.0m; // Başlangıç değeri

            decimal totalWins = 0;
            decimal totalLosses = 0;
            int winCount = 0;
            int lossCount = 0;

            foreach (var ret in _returns)
            {
                if (ret > 0)
                {
                    totalWins += ret;
                    winCount++;
                }
                else if (ret < 0)
                {
                    totalLosses -= ret;
                    lossCount++;
                }
            }

            if (lossCount == 0 || totalLosses == 0)
                return 2.0m;

            var avgWin = totalWins / winCount;
            var avgLoss = totalLosses / lossCount;

            return avgWin / avgLoss;
        }
    }
}

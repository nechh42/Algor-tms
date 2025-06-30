using System;
using System.Collections.Generic;
using System.Linq;
using TradingBot.Models;

namespace TradingBot
{
    public class TechnicalAnalyzer
    {
        private readonly Queue<decimal> _prices = new Queue<decimal>();
        private readonly Queue<decimal> _volumes = new Queue<decimal>();
        private readonly int _rsiPeriod;
        private readonly int _emaPeriod;
        private readonly int _atrPeriod;
        private readonly int _macdFastPeriod;
        private readonly int _macdSlowPeriod;
        private readonly int _macdSignalPeriod;
        private readonly int _bBandsPeriod;
        private readonly decimal _bBandsMultiplier;
        private readonly int _stochPeriod;
        private readonly int _maxPeriod;

        private List<decimal> _gains = new List<decimal>();
        private List<decimal> _losses = new List<decimal>();
        private decimal? _lastEma;
        private decimal? _lastTr;
        private decimal? _lastVwap;
        private decimal _cumulativeVolume;
        private decimal _cumulativePriceVolume;

        public TechnicalAnalyzer(
            int rsiPeriod = 14, 
            int emaPeriod = 20, 
            int atrPeriod = 14,
            int macdFastPeriod = 12,
            int macdSlowPeriod = 26,
            int macdSignalPeriod = 9,
            int bBandsPeriod = 20,
            decimal bBandsMultiplier = 2.0m,
            int stochPeriod = 14)
        {
            _rsiPeriod = rsiPeriod;
            _emaPeriod = emaPeriod;
            _atrPeriod = atrPeriod;
            _macdFastPeriod = macdFastPeriod;
            _macdSlowPeriod = macdSlowPeriod;
            _macdSignalPeriod = macdSignalPeriod;
            _bBandsPeriod = bBandsPeriod;
            _bBandsMultiplier = bBandsMultiplier;
            _stochPeriod = stochPeriod;
            _maxPeriod = new[] { rsiPeriod, emaPeriod, atrPeriod, macdSlowPeriod + macdSignalPeriod, bBandsPeriod, stochPeriod }.Max();
        }

        public void UpdateData(HistoricalData data)
        {
            _prices.Enqueue(data.Close);
            _volumes.Enqueue(data.Volume);
            
            // VWAP hesaplama
            _cumulativeVolume += data.Volume;
            _cumulativePriceVolume += data.Close * data.Volume;
            _lastVwap = _cumulativePriceVolume / _cumulativeVolume;

            if (_prices.Count > _maxPeriod)
            {
                _prices.Dequeue();
                _volumes.Dequeue();
            }

            if (_prices.Count > 1)
            {
                var priceChange = data.Close - _prices.ElementAt(_prices.Count - 2);
                if (priceChange > 0)
                {
                    _gains.Add(priceChange);
                    _losses.Add(0);
                }
                else
                {
                    _gains.Add(0);
                    _losses.Add(-priceChange);
                }

                if (_gains.Count > _maxPeriod)
                {
                    _gains.RemoveAt(0);
                    _losses.RemoveAt(0);
                }
            }
        }

        public decimal CalculateRSI()
        {
            if (_prices.Count < _rsiPeriod) return 50;

            var avgGain = _gains.TakeLast(_rsiPeriod).Average();
            var avgLoss = _losses.TakeLast(_rsiPeriod).Average();

            if (avgLoss == 0) return 100;
            var rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        public decimal CalculateEMA()
        {
            if (_prices.Count < _emaPeriod) return _prices.Last();

            if (_lastEma == null)
            {
                _lastEma = _prices.Take(_emaPeriod).Average();
                return _lastEma.Value;
            }

            var multiplier = 2.0m / (_emaPeriod + 1);
            _lastEma = (_prices.Last() - _lastEma.Value) * multiplier + _lastEma.Value;
            return _lastEma.Value;
        }

        public decimal CalculateATR()
        {
            if (_prices.Count < 2) return 0;

            var high = _prices.Max();
            var low = _prices.Min();
            var close = _prices.ElementAt(_prices.Count - 2);

            var tr = Math.Max(high - low, Math.Max(Math.Abs(high - close), Math.Abs(low - close)));

            if (_lastTr == null)
            {
                _lastTr = tr;
                return tr;
            }

            _lastTr = ((_lastTr * (_atrPeriod - 1)) + tr) / _atrPeriod;
            return _lastTr.Value;
        }

        public (decimal macd, decimal signal, decimal histogram) CalculateMACD()
        {
            if (_prices.Count < _macdSlowPeriod + _macdSignalPeriod)
                return (0, 0, 0);

            var fastEMA = CalculateCustomEMA(_prices.ToList(), _macdFastPeriod);
            var slowEMA = CalculateCustomEMA(_prices.ToList(), _macdSlowPeriod);
            var macd = fastEMA - slowEMA;

            var macdValues = new List<decimal>();
            for (int i = 0; i < _macdSignalPeriod; i++)
            {
                macdValues.Add(macd);
            }

            var signal = CalculateCustomEMA(macdValues, _macdSignalPeriod);
            var histogram = macd - signal;

            return (macd, signal, histogram);
        }

        public (decimal upper, decimal middle, decimal lower) CalculateBollingerBands()
        {
            if (_prices.Count < _bBandsPeriod)
                return (0, 0, 0);

            var prices = _prices.TakeLast(_bBandsPeriod).ToList();
            var sma = prices.Average();
            var stdDev = CalculateStandardDeviation(prices);

            var upper = sma + (_bBandsMultiplier * stdDev);
            var lower = sma - (_bBandsMultiplier * stdDev);

            return (upper, sma, lower);
        }

        public (decimal k, decimal d) CalculateStochastic()
        {
            if (_prices.Count < _stochPeriod)
                return (0, 0);

            var prices = _prices.TakeLast(_stochPeriod).ToList();
            var high = prices.Max();
            var low = prices.Min();
            var current = prices.Last();

            var k = ((current - low) / (high - low)) * 100;
            var d = CalculateCustomEMA(new List<decimal> { k }, 3); // 3-period SMA of %K

            return (k, d);
        }

        public decimal? GetVWAP()
        {
            return _lastVwap;
        }

        private decimal CalculateCustomEMA(List<decimal> values, int period)
        {
            if (values.Count < period)
                return values.Last();

            var sma = values.Take(period).Average();
            var multiplier = 2.0m / (period + 1);
            var ema = sma;

            for (int i = period; i < values.Count; i++)
            {
                ema = (values[i] - ema) * multiplier + ema;
            }

            return ema;
        }

        private decimal CalculateStandardDeviation(List<decimal> values)
        {
            var avg = values.Average();
            var sumOfSquares = values.Sum(x => (x - avg) * (x - avg));
            return (decimal)Math.Sqrt((double)(sumOfSquares / values.Count));
        }

        public void ClearData()
        {
            _prices.Clear();
            _volumes.Clear();
            _gains.Clear();
            _losses.Clear();
            _lastEma = null;
            _lastTr = null;
            _lastVwap = null;
            _cumulativeVolume = 0;
            _cumulativePriceVolume = 0;
        }
    }
}

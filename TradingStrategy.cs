using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TradingBot.Models;
using TradingBot.RiskManagement;

namespace TradingBot
{
    public enum SignalType
    {
        None,
        Buy,
        Sell
    }

    public class TradingStrategy
    {
        private readonly MLPredictor _mlPredictor;
        private readonly RiskManager _riskManager;
        private readonly List<HistoricalData> _historicalData;
        private readonly int _lookbackPeriod = 20;

        public TradingStrategy(MLPredictor mlPredictor, RiskManager riskManager)
        {
            _mlPredictor = mlPredictor;
            _riskManager = riskManager;
            _historicalData = new List<HistoricalData>();
        }

        public async Task<SignalType> GenerateSignal(HistoricalData currentData)
        {
            _historicalData.Add(currentData);

            if (_historicalData.Count < _lookbackPeriod)
            {
                return SignalType.None;
            }

            // Son n günlük veriyi al
            var recentData = _historicalData.GetRange(
                _historicalData.Count - _lookbackPeriod,
                _lookbackPeriod
            );

            // ML tahminini al
            var prediction = await _mlPredictor.PredictNextMove(recentData);

            // Güven kontrolü
            if (prediction.Confidence < 0.6m)
            {
                return SignalType.None;
            }

            // Alım/satım sinyali üret
            return prediction.PredictedDirection ? SignalType.Buy : SignalType.Sell;
        }
    }
}

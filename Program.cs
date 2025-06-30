using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TradingBot.RiskManagement;
using TradingBot.Models;

namespace TradingBot
{
    class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                Console.WriteLine("TradingBot başlatılıyor...");

                // API servisini başlat
                var apiService = new ApiService("YOUR_API_KEY");

                // Geçmiş verileri al
                var historicalData = await apiService.GetHistoricalDataAsync("THYAO.IS", 100);
                if (historicalData == null || historicalData.Count == 0)
                {
                    Console.WriteLine("Geçmiş veri alınamadı.");
                    return;
                }

                Console.WriteLine($"{historicalData.Count} adet geçmiş veri alındı.");

                // Teknik analiz aracını oluştur
                var technicalAnalyzer = new TechnicalAnalyzer();

                // ML modelini yükle
                var mlPredictor = new MLPredictor(technicalAnalyzer);
                await mlPredictor.LoadModel("model.zip");

                // Risk yöneticisini oluştur
                var riskManager = new RiskManager(100000m, 0.02m);

                // Trading stratejisini oluştur
                var strategy = new TradingStrategy(mlPredictor, riskManager);

                // Backtesting motorunu başlat
                var backtest = new Backtesting.BacktestEngine(strategy, riskManager);
                backtest.RunBacktest(historicalData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Hata oluştu: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }

            Console.WriteLine("\nDevam etmek için bir tuşa basın...");
            Console.ReadKey();
        }
    }
}
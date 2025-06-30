using System;
using System.Collections.Generic;
using MatriksIQ.API;
using MatriksIQ.Data;
using MatriksIQ.Order;
using System.Threading.Tasks;

namespace RealTradingSystem
{
    public class TradingSystem
    {
        private readonly IMatriksConnection _connection;
        private readonly IDataFeed _dataFeed;
        private readonly IOrderManager _orderManager;
        private readonly IRiskManager _riskManager;
        private readonly Dictionary<string, StockData> _stockData;
        
        public TradingSystem(string apiKey, string secretKey)
        {
            // Matriks IQ Bağlantısı
            _connection = new MatriksConnection(apiKey, secretKey);
            _dataFeed = new RealTimeDataFeed(_connection);
            _orderManager = new OrderManager(_connection);
            _riskManager = new RiskManager();
            _stockData = new Dictionary<string, StockData>();

            // Hisse Listesi
            var stocks = new[] { "THYAO", "GARAN", "ASELS", "KRDMD", "YKBNK", 
                               "SISE", "TUPRS", "AKBNK", "BIMAS", "KOZAL" };
            
            foreach (var symbol in stocks)
            {
                _stockData[symbol] = new StockData();
            }
        }

        public async Task Initialize()
        {
            try
            {
                await _connection.ConnectAsync();
                await SubscribeToData();
                InitializeRiskParameters();
                StartMonitoring();
            }
            catch (Exception ex)
            {
                throw new SystemException($"Başlatma hatası: {ex.Message}");
            }
        }

        private async Task SubscribeToData()
        {
            foreach (var symbol in _stockData.Keys)
            {
                // Gerçek-zamanlı veri aboneliği
                await _dataFeed.SubscribeToPrice(symbol);
                await _dataFeed.SubscribeToVolume(symbol);
                await _dataFeed.SubscribeToDepth(symbol);
            }

            _dataFeed.OnPriceUpdate += HandlePriceUpdate;
            _dataFeed.OnVolumeUpdate += HandleVolumeUpdate;
            _dataFeed.OnError += HandleError;
        }

        private void InitializeRiskParameters()
        {
            _riskManager.SetParameters(new RiskParameters
            {
                MaxPositionSize = 150000,
                MaxDailyLoss = -5000,
                MaxPositions = 5,
                StopLossPercent = 2.5m,
                TakeProfitPercent = 4.5m,
                UseATRStopLoss = true,
                EnableDynamicPositionSizing = true
            });
        }

        private void HandlePriceUpdate(string symbol, decimal price)
        {
            try
            {
                var stock = _stockData[symbol];
                stock.UpdatePrice(price);
                
                // Teknik Analiz
                var technicals = CalculateTechnicals(stock);
                
                // Sinyal Kontrolü
                CheckSignals(symbol, technicals);
                
                // Pozisyon Kontrolü
                CheckPositions(symbol, price);
            }
            catch (Exception ex)
            {
                LogError($"Fiyat güncelleme hatası - {symbol}: {ex.Message}");
            }
        }

        private TechnicalIndicators CalculateTechnicals(StockData stock)
        {
            return new TechnicalIndicators
            {
                RSI = stock.CalculateRSI(14),
                MACD = stock.CalculateMACD(12, 26, 9),
                BollingerBands = stock.CalculateBollingerBands(20, 2),
                ATR = stock.CalculateATR(14),
                OBV = stock.CalculateOBV(),
                VolumeProfile = stock.CalculateVolumeProfile()
            };
        }

        private async Task PlaceOrder(string symbol, OrderType type, decimal price, decimal quantity)
        {
            try
            {
                // Risk Kontrolü
                if (!_riskManager.ValidateOrder(symbol, type, price, quantity))
                {
                    LogWarning($"Risk limiti aşıldı - {symbol}");
                    return;
                }

                // Emir Doğrulama
                var order = new Order
                {
                    Symbol = symbol,
                    Type = type,
                    Price = price,
                    Quantity = quantity,
                    TimeInForce = TimeInForce.Day,
                    ValidationType = ValidationType.Strong
                };

                // Emir Gönderme
                var result = await _orderManager.PlaceOrderAsync(order);
                if (result.Success)
                {
                    LogInfo($"Emir başarılı - {symbol}: {type} @ {price}");
                }
                else
                {
                    LogError($"Emir hatası - {symbol}: {result.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                LogError($"Emir gönderme hatası - {symbol}: {ex.Message}");
            }
        }

        private async Task ClosePosition(string symbol, decimal price)
        {
            try
            {
                var position = _riskManager.GetPosition(symbol);
                if (position != null)
                {
                    await PlaceOrder(symbol, OrderType.Sell, price, position.Quantity);
                }
            }
            catch (Exception ex)
            {
                LogError($"Pozisyon kapatma hatası - {symbol}: {ex.Message}");
            }
        }

        private void StartMonitoring()
        {
            Task.Run(async () =>
            {
                while (true)
                {
                    try
                    {
                        // Bağlantı Kontrolü
                        if (!_connection.IsConnected)
                        {
                            await _connection.ReconnectAsync();
                        }

                        // Risk Kontrolü
                        _riskManager.CheckDailyLimits();
                        
                        // Pozisyon Kontrolü
                        CheckAllPositions();

                        await Task.Delay(1000); // 1 saniye bekle
                    }
                    catch (Exception ex)
                    {
                        LogError($"Monitoring hatası: {ex.Message}");
                    }
                }
            });
        }

        private void CheckAllPositions()
        {
            foreach (var position in _riskManager.GetAllPositions())
            {
                var stock = _stockData[position.Symbol];
                var currentPrice = stock.LastPrice;

                // Stop Loss Kontrolü
                if (currentPrice <= position.StopLossPrice)
                {
                    Task.Run(() => ClosePosition(position.Symbol, currentPrice));
                }
                // Take Profit Kontrolü
                else if (currentPrice >= position.TakeProfitPrice)
                {
                    Task.Run(() => ClosePosition(position.Symbol, currentPrice));
                }
            }
        }
    }
}

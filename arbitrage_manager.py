import numpy as np
import pandas as pd
from binance.client import Client
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import asyncio
import aiohttp

class ArbitrageManager:
    def __init__(self, client: Client, min_profit_percent: float = 0.5):
        """
        Arbitraj yöneticisi
        :param client: Binance client
        :param min_profit_percent: Minimum kâr yüzdesi
        """
        self.client = client
        self.min_profit_percent = min_profit_percent
        self.logger = logging.getLogger(__name__)
        
        # Desteklenen arbitraj tipleri
        self.arbitrage_types = {
            'triangular': self._check_triangular_arbitrage,
            'cross_exchange': self._check_cross_exchange_arbitrage,
            'futures_spot': self._check_futures_spot_arbitrage
        }
        
    async def find_arbitrage_opportunities(self) -> List[Dict]:
        """Arbitraj fırsatlarını bul"""
        try:
            opportunities = []
            
            # Tüm arbitraj tiplerini kontrol et
            for arb_type, check_func in self.arbitrage_types.items():
                arb_opps = await check_func()
                if arb_opps:
                    opportunities.extend(arb_opps)
                    
            # Fırsatları kâr potansiyeline göre sırala
            opportunities.sort(key=lambda x: x['potential_profit'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Arbitraj fırsatı arama hatası: {str(e)}")
            return []
            
    async def _check_triangular_arbitrage(self) -> List[Dict]:
        """Üçgen arbitraj fırsatlarını kontrol et"""
        try:
            # Tüm sembolleri al
            exchange_info = self.client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols']]
            
            # Üçgen oluşturabilecek sembolleri bul
            triangles = self._find_triangular_pairs(symbols)
            opportunities = []
            
            async with aiohttp.ClientSession() as session:
                for triangle in triangles:
                    # Her üçgen için fiyatları al
                    prices = await self._get_triangle_prices(session, triangle)
                    if not prices:
                        continue
                        
                    # Kâr potansiyelini hesapla
                    profit = self._calculate_triangular_profit(triangle, prices)
                    if profit > self.min_profit_percent:
                        opportunities.append({
                            'type': 'triangular',
                            'pairs': triangle,
                            'prices': prices,
                            'potential_profit': profit,
                            'timestamp': datetime.now()
                        })
                        
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Üçgen arbitraj kontrolü hatası: {str(e)}")
            return []
            
    async def _check_cross_exchange_arbitrage(self) -> List[Dict]:
        """Borsalar arası arbitraj fırsatlarını kontrol et"""
        try:
            # Binance ve diğer borsalardan fiyatları al
            opportunities = []
            
            async with aiohttp.ClientSession() as session:
                # Desteklenen borsalar için fiyatları al
                exchanges = ['binance', 'huobi', 'kucoin']  # Örnek borsalar
                prices = await self._get_cross_exchange_prices(session, exchanges)
                
                # Her sembol için fiyat farklarını kontrol et
                for symbol, exchange_prices in prices.items():
                    profit = self._calculate_cross_exchange_profit(exchange_prices)
                    if profit > self.min_profit_percent:
                        opportunities.append({
                            'type': 'cross_exchange',
                            'symbol': symbol,
                            'prices': exchange_prices,
                            'potential_profit': profit,
                            'timestamp': datetime.now()
                        })
                        
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Borsalar arası arbitraj kontrolü hatası: {str(e)}")
            return []
            
    async def _check_futures_spot_arbitrage(self) -> List[Dict]:
        """Vadeli-Spot arbitraj fırsatlarını kontrol et"""
        try:
            opportunities = []
            
            # Vadeli işlem sembolleri
            futures_symbols = [s['symbol'] for s in self.client.futures_exchange_info()['symbols']]
            
            for symbol in futures_symbols:
                # Spot ve vadeli fiyatları al
                spot_price = float(self.client.get_symbol_ticker(symbol=symbol)['price'])
                futures_price = float(self.client.futures_symbol_ticker(symbol=symbol)['price'])
                
                # Funding rate'i al
                funding_rate = float(self.client.futures_mark_price(symbol=symbol)['lastFundingRate'])
                
                # Kâr potansiyelini hesapla
                profit = self._calculate_futures_spot_profit(
                    spot_price, futures_price, funding_rate
                )
                
                if profit > self.min_profit_percent:
                    opportunities.append({
                        'type': 'futures_spot',
                        'symbol': symbol,
                        'spot_price': spot_price,
                        'futures_price': futures_price,
                        'funding_rate': funding_rate,
                        'potential_profit': profit,
                        'timestamp': datetime.now()
                    })
                    
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Vadeli-Spot arbitraj kontrolü hatası: {str(e)}")
            return []
            
    def _find_triangular_pairs(self, symbols: List[str]) -> List[Tuple[str, str, str]]:
        """Üçgen oluşturabilecek sembolleri bul"""
        try:
            triangles = []
            base_currencies = {'BTC', 'ETH', 'USDT', 'BNB'}
            
            for s1 in symbols:
                for s2 in symbols:
                    for s3 in symbols:
                        # Geçerli bir üçgen oluşturup oluşturmadığını kontrol et
                        if self._is_valid_triangle(s1, s2, s3, base_currencies):
                            triangles.append((s1, s2, s3))
                            
            return triangles
            
        except Exception as e:
            self.logger.error(f"Üçgen çiftleri bulma hatası: {str(e)}")
            return []
            
    async def _get_triangle_prices(self, session: aiohttp.ClientSession,
                                 triangle: Tuple[str, str, str]) -> Dict:
        """Üçgen için fiyatları al"""
        try:
            prices = {}
            for symbol in triangle:
                async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        prices[symbol] = float(data['price'])
                    else:
                        return None
                        
            return prices
            
        except Exception as e:
            self.logger.error(f"Üçgen fiyatları alma hatası: {str(e)}")
            return None
            
    def _calculate_triangular_profit(self, triangle: Tuple[str, str, str],
                                   prices: Dict) -> float:
        """Üçgen arbitraj kârını hesapla"""
        try:
            # İşlem maliyetlerini hesaba kat
            fee = 0.001  # %0.1 işlem ücreti
            
            # İlk işlem
            amount = 1.0
            amount *= prices[triangle[0]] * (1 - fee)
            
            # İkinci işlem
            amount *= prices[triangle[1]] * (1 - fee)
            
            # Son işlem
            amount *= prices[triangle[2]] * (1 - fee)
            
            # Kâr yüzdesini hesapla
            profit_percent = (amount - 1.0) * 100
            
            return profit_percent
            
        except Exception as e:
            self.logger.error(f"Üçgen kârı hesaplama hatası: {str(e)}")
            return 0.0
            
    async def _get_cross_exchange_prices(self, session: aiohttp.ClientSession,
                                       exchanges: List[str]) -> Dict:
        """Farklı borsalardan fiyatları al"""
        try:
            prices = {}
            
            # Her borsa için API endpoint'leri
            endpoints = {
                'binance': 'https://api.binance.com/api/v3/ticker/price',
                'huobi': 'https://api.huobi.pro/market/tickers',
                'kucoin': 'https://api.kucoin.com/api/v1/market/allTickers'
            }
            
            for exchange in exchanges:
                async with session.get(endpoints[exchange]) as response:
                    if response.status == 200:
                        data = await response.json()
                        prices[exchange] = self._parse_exchange_prices(exchange, data)
                        
            return prices
            
        except Exception as e:
            self.logger.error(f"Borsa fiyatları alma hatası: {str(e)}")
            return {}
            
    def _calculate_cross_exchange_profit(self, prices: Dict) -> float:
        """Borsalar arası arbitraj kârını hesapla"""
        try:
            # En düşük ve en yüksek fiyatları bul
            min_price = min(prices.values())
            max_price = max(prices.values())
            
            # İşlem maliyetlerini hesaba kat
            fee = 0.001  # %0.1 işlem ücreti
            transfer_fee = 0.002  # %0.2 transfer ücreti
            
            # Kâr potansiyelini hesapla
            profit = (max_price - min_price) / min_price * 100
            profit -= (fee * 2 + transfer_fee) * 100  # İşlem ve transfer ücretlerini düş
            
            return profit
            
        except Exception as e:
            self.logger.error(f"Borsalar arası kâr hesaplama hatası: {str(e)}")
            return 0.0
            
    def _calculate_futures_spot_profit(self, spot_price: float,
                                     futures_price: float,
                                     funding_rate: float) -> float:
        """Vadeli-Spot arbitraj kârını hesapla"""
        try:
            # Fiyat farkını hesapla
            price_diff = (futures_price - spot_price) / spot_price * 100
            
            # Funding rate etkisini hesapla (8 saatlik)
            funding_impact = funding_rate * 3 * 100  # Günlük etki
            
            # İşlem maliyetlerini hesaba kat
            fee = 0.001  # %0.1 işlem ücreti
            total_fee = fee * 4 * 100  # Açılış ve kapanış için toplam 4 işlem
            
            # Net kârı hesapla
            profit = abs(price_diff) - total_fee - abs(funding_impact)
            
            return profit
            
        except Exception as e:
            self.logger.error(f"Vadeli-Spot kâr hesaplama hatası: {str(e)}")
            return 0.0
            
    def _is_valid_triangle(self, s1: str, s2: str, s3: str,
                          base_currencies: set) -> bool:
        """Üçgenin geçerli olup olmadığını kontrol et"""
        try:
            # Sembolleri parçala
            pairs = [
                (s1[:3], s1[3:]),
                (s2[:3], s2[3:]),
                (s3[:3], s3[3:])
            ]
            
            # En az bir base currency içermeli
            has_base = any(p[0] in base_currencies or p[1] in base_currencies 
                         for p in pairs)
            if not has_base:
                return False
                
            # Üçgen oluşturup oluşturmadığını kontrol et
            currencies = set()
            for base, quote in pairs:
                currencies.add(base)
                currencies.add(quote)
                
            # Tam olarak 3 farklı currency olmalı
            return len(currencies) == 3
            
        except Exception as e:
            self.logger.error(f"Üçgen geçerlilik kontrolü hatası: {str(e)}")
            return False
            
    def _parse_exchange_prices(self, exchange: str, data: Dict) -> Dict:
        """Borsa verilerini parse et"""
        try:
            prices = {}
            
            if exchange == 'binance':
                for item in data:
                    prices[item['symbol']] = float(item['price'])
                    
            elif exchange == 'huobi':
                for item in data['data']:
                    prices[item['symbol'].upper()] = float(item['close'])
                    
            elif exchange == 'kucoin':
                for item in data['data']['ticker']:
                    prices[item['symbol'].replace('-', '')] = float(item['last'])
                    
            return prices
            
        except Exception as e:
            self.logger.error(f"Borsa veri parse hatası: {str(e)}")
            return {}

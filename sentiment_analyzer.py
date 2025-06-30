import tweepy
import requests
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

class SentimentAnalyzer:
    def __init__(self, twitter_api_key: str, twitter_api_secret: str,
                 twitter_access_token: str, twitter_access_secret: str,
                 news_api_key: str):
        """
        Duygu analizi sınıfı
        :param twitter_api_key: Twitter API anahtarı
        :param twitter_api_secret: Twitter API gizli anahtarı
        :param news_api_key: News API anahtarı
        """
        self.logger = logging.getLogger(__name__)
        
        # Twitter API kurulumu
        try:
            auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret)
            auth.set_access_token(twitter_access_token, twitter_access_secret)
            self.twitter_api = tweepy.API(auth)
        except Exception as e:
            self.logger.error(f"Twitter API bağlantı hatası: {str(e)}")
            self.twitter_api = None
            
        self.news_api_key = news_api_key
        
        # Kripto para anahtar kelimeleri
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', 'bitcoin crypto', 'btc price'],
            'ETH': ['ethereum', 'eth', 'ethereum crypto', 'eth price'],
            'BNB': ['binance coin', 'bnb', 'binance crypto', 'bnb price']
            # Diğer coinler için anahtar kelimeler eklenebilir
        }
        
    def analyze_sentiment(self, symbol: str, lookback_hours: int = 24) -> Dict:
        """
        Belirli bir sembol için duygu analizi yap
        :param symbol: Analiz edilecek sembol (örn: BTC)
        :param lookback_hours: Kaç saat geriye bakılacak
        :return: Duygu analizi sonuçları
        """
        try:
            # Twitter ve haber verilerini topla
            twitter_data = self._get_twitter_sentiment(symbol, lookback_hours)
            news_data = self._get_news_sentiment(symbol, lookback_hours)
            
            if not twitter_data and not news_data:
                return None
                
            # Sonuçları birleştir
            combined_sentiment = self._combine_sentiment_scores(twitter_data, news_data)
            
            # Trend analizi
            trend_analysis = self._analyze_sentiment_trend(combined_sentiment)
            
            return {
                'current_sentiment': combined_sentiment['sentiment_score'],
                'sentiment_trend': trend_analysis['trend'],
                'trend_strength': trend_analysis['strength'],
                'confidence': combined_sentiment['confidence'],
                'source_metrics': {
                    'twitter_count': len(twitter_data['tweets']) if twitter_data else 0,
                    'news_count': len(news_data['articles']) if news_data else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Duygu analizi hatası: {str(e)}")
            return None
            
    def _get_twitter_sentiment(self, symbol: str, lookback_hours: int) -> Dict:
        """Twitter'dan duygu analizi"""
        try:
            if not self.twitter_api:
                return None
                
            tweets = []
            keywords = self.crypto_keywords.get(symbol, [symbol])
            
            for keyword in keywords:
                # Tweet'leri topla
                search_results = tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=keyword,
                    lang="en",
                    tweet_mode="extended",
                    since_id=self._get_since_id(lookback_hours)
                ).items(100)  # Her anahtar kelime için 100 tweet
                
                for tweet in search_results:
                    # Retweet'leri ve spam'leri filtrele
                    if not self._is_valid_tweet(tweet):
                        continue
                        
                    # Duygu analizi yap
                    sentiment = self._analyze_text(tweet.full_text)
                    
                    tweets.append({
                        'text': tweet.full_text,
                        'sentiment': sentiment['sentiment'],
                        'confidence': sentiment['confidence'],
                        'timestamp': tweet.created_at
                    })
                    
            if not tweets:
                return None
                
            return {
                'tweets': tweets,
                'avg_sentiment': np.mean([t['sentiment'] for t in tweets]),
                'confidence': np.mean([t['confidence'] for t in tweets])
            }
            
        except Exception as e:
            self.logger.error(f"Twitter veri toplama hatası: {str(e)}")
            return None
            
    def _get_news_sentiment(self, symbol: str, lookback_hours: int) -> Dict:
        """Haber kaynaklarından duygu analizi"""
        try:
            articles = []
            keywords = self.crypto_keywords.get(symbol, [symbol])
            
            for keyword in keywords:
                # News API'den haberleri al
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'from': (datetime.now() - timedelta(hours=lookback_hours)).isoformat()
                }
                
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    continue
                    
                news_data = response.json()
                
                for article in news_data.get('articles', []):
                    # Başlık ve içerik analizi
                    title_sentiment = self._analyze_text(article['title'])
                    content_sentiment = self._analyze_text(article.get('description', ''))
                    
                    # Başlığa daha fazla ağırlık ver
                    combined_sentiment = (title_sentiment['sentiment'] * 0.7 +
                                       content_sentiment['sentiment'] * 0.3)
                    combined_confidence = (title_sentiment['confidence'] * 0.7 +
                                        content_sentiment['confidence'] * 0.3)
                    
                    articles.append({
                        'title': article['title'],
                        'sentiment': combined_sentiment,
                        'confidence': combined_confidence,
                        'timestamp': article['publishedAt']
                    })
                    
            if not articles:
                return None
                
            return {
                'articles': articles,
                'avg_sentiment': np.mean([a['sentiment'] for a in articles]),
                'confidence': np.mean([a['confidence'] for a in articles])
            }
            
        except Exception as e:
            self.logger.error(f"Haber toplama hatası: {str(e)}")
            return None
            
    def _analyze_text(self, text: str) -> Dict:
        """Metin analizi yap"""
        try:
            # TextBlob ile duygu analizi
            analysis = TextBlob(text)
            
            # Polarite (-1 ile 1 arası) ve öznellik (0 ile 1 arası)
            sentiment_score = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            # Güven skorunu hesapla
            confidence = 1 - abs(subjectivity - 0.5) * 2
            
            return {
                'sentiment': sentiment_score,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Metin analizi hatası: {str(e)}")
            return {'sentiment': 0, 'confidence': 0}
            
    def _combine_sentiment_scores(self, twitter_data: Optional[Dict],
                                news_data: Optional[Dict]) -> Dict:
        """Farklı kaynaklardan gelen duygu skorlarını birleştir"""
        try:
            scores = []
            confidences = []
            weights = []
            
            if twitter_data:
                scores.append(twitter_data['avg_sentiment'])
                confidences.append(twitter_data['confidence'])
                weights.append(0.4)  # Twitter %40 ağırlık
                
            if news_data:
                scores.append(news_data['avg_sentiment'])
                confidences.append(news_data['confidence'])
                weights.append(0.6)  # Haberler %60 ağırlık
                
            if not scores:
                return {'sentiment_score': 0, 'confidence': 0}
                
            # Ağırlıklı ortalama hesapla
            weights = np.array(weights) / sum(weights)
            sentiment_score = np.average(scores, weights=weights)
            confidence = np.average(confidences, weights=weights)
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Skor birleştirme hatası: {str(e)}")
            return {'sentiment_score': 0, 'confidence': 0}
            
    def _analyze_sentiment_trend(self, sentiment_data: Dict) -> Dict:
        """Duygu trendi analizi"""
        try:
            sentiment_score = sentiment_data['sentiment_score']
            confidence = sentiment_data['confidence']
            
            # Trend belirleme
            if sentiment_score > 0.2:
                trend = 'bullish'
            elif sentiment_score < -0.2:
                trend = 'bearish'
            else:
                trend = 'neutral'
                
            # Trend gücü (0-1 arası)
            strength = abs(sentiment_score) * confidence
            
            return {
                'trend': trend,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Trend analizi hatası: {str(e)}")
            return {'trend': 'neutral', 'strength': 0}
            
    def _is_valid_tweet(self, tweet) -> bool:
        """Tweet'in geçerli olup olmadığını kontrol et"""
        try:
            # Retweet kontrolü
            if hasattr(tweet, 'retweeted_status'):
                return False
                
            # Spam kontrolü (basit)
            text = tweet.full_text.lower()
            spam_indicators = ['giveaway', 'airdrop', 'win free', 'click here']
            if any(indicator in text for indicator in spam_indicators):
                return False
                
            # Minimum kelime sayısı kontrolü
            if len(text.split()) < 5:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Tweet doğrulama hatası: {str(e)}")
            return False
            
    def _get_since_id(self, lookback_hours: int) -> Optional[int]:
        """Belirli saat öncesine ait tweet ID'sini bul"""
        try:
            if not self.twitter_api:
                return None
                
            # Şu anki zamandan geriye git
            target_time = datetime.now() - timedelta(hours=lookback_hours)
            
            # Yaklaşık bir tweet ID hesapla
            snowflake_time = int((target_time - datetime(2010, 11, 4)).total_seconds() * 1000)
            since_id = snowflake_time << 22
            
            return since_id
            
        except Exception as e:
            self.logger.error(f"Tweet ID hesaplama hatası: {str(e)}")
            return None

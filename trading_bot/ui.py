# ui.py - Kullanıcı arayüzü
import streamlit as st
import plotly.graph_objects as go

class UserInterface:
    def __init__(self, bot):
        self.bot = bot
        
    def render(self):
        st.title("Binance Trading Bot")
        
        # Üst menü
        menu = st.sidebar.selectbox(
            "Menü",
            ["Ana Sayfa", "Açık Pozisyonlar", "İşlem Geçmişi", "Ayarlar"]
        )
        
        if menu == "Ana Sayfa":
            self.render_dashboard()
        elif menu == "Açık Pozisyonlar":
            self.render_positions()
        elif menu == "İşlem Geçmişi":
            self.render_history()
        elif menu == "Ayarlar":
            self.render_settings()
            
    def render_dashboard(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Spot Bakiye", self.bot.get_spot_balance())
        with col2:
            st.metric("Futures Bakiye", self.bot.get_futures_balance())
        with col3:
            st.metric("24s Kar/Zarar", self.bot.get_daily_pnl())

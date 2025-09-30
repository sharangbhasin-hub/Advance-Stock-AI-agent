import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 Elite AI Trading Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Environment and API Configuration ---
warnings.filterwarnings('ignore')
load_dotenv()

# Load API keys from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Configure the Gemini API
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Google Gemini API: {e}")

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
        <style>
        /* Main page styling */
        .stApp {
            background-color: #121212;
            color: #E0E0E0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1E1E1E;
            border-right: 1px solid #333;
        }

        /* Metric styling */
        .stMetric {
            background-color: #2E2E2E;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #007BFF;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .stMetric > label {
            font-weight: bold;
            color: #B0B0B0;
        }

        .stMetric > div > div > p {
            font-size: 1.5rem;
            color: #FFFFFF;
        }

        /* Button styling */
        .stButton>button {
            border: 2px solid #007BFF;
            border-radius: 20px;
            color: #FFFFFF;
            background-color: #007BFF;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #FFFFFF;
            color: #007BFF;
            border-color: #007BFF;
        }
        
        /* Expander styling */
        .stExpander {
            background-color: #2E2E2E;
            border-radius: 10px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1E1E1E;
            border-radius: 8px 8px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
        }
        
        /* Custom containers */
        .custom-container {
            padding: 20px;
            border-radius: 10px;
            background-color: #2E2E2E;
            margin-bottom: 20px;
        }
        
        .result-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #007BFF;
            margin-bottom: 10px;
            border-bottom: 2px solid #007BFF;
            padding-bottom: 5px;
        }

        </style>
    """, unsafe_allow_html=True)

# --- Static Data & Mappings ---
STOCK_CATEGORIES = {
    "NIFTY 50": {
        "ticker": "^NSEI",
        "stocks": {
            "Reliance": "RELIANCE.NS", "TCS": "TCS.NS", "HDFC Bank": "HDFCBANK.NS",
            "Infosys": "INFY.NS", "ICICI Bank": "ICICIBANK.NS", "Bharti Airtel": "BHARTIARTL.NS",
            "State Bank of India": "SBIN.NS", "LIC": "LICI.NS", "ITC": "ITC.NS",
            "Hindustan Unilever": "HINDUNILVR.NS"
        }
    },
    "BANK NIFTY": {
        "ticker": "^NSEBANK",
        "stocks": {
            "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "State Bank of India": "SBIN.NS",
            "Kotak Mahindra Bank": "KOTAKBANK.NS", "Axis Bank": "AXISBANK.NS", "IndusInd Bank": "INDUSINDBK.NS",
            "Bank of Baroda": "BANKBARODA.NS", "Punjab National Bank": "PNB.NS",
        }
    },
     "NIFTY IT": {
        "ticker": "^CNXIT",
        "stocks": {
            "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Tech": "HCLTECH.NS",
            "Wipro": "WIPRO.NS", "LTIMindtree": "LTIM.NS", "Tech Mahindra": "TECHM.NS",
        }
    }
}

# ==============================================================================
# === CORE ANALYSIS CLASS ======================================================
# ==============================================================================

class AdvancedStockAnalyzer:
    """
    An advanced stock analysis agent that incorporates multi-timeframe analysis,
    advanced technical indicators, chart patterns, and AI-powered insights.
    """
    
    @st.cache_data(ttl=300) # Cache data for 5 minutes
    def get_stock_data(_self, ticker, period="3mo", interval="1d"):
        """Fetches historical stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            if data.empty:
                st.warning(f"No data found for {ticker} at {interval} interval.")
                return None, None
            info = stock.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data for '{ticker}': {e}")
            return None, None

    def calculate_indicators(self, data):
        """Calculates all required technical indicators."""
        if data is None or data.empty:
            return data
            
        # Standard Indicators
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

        # Advanced Indicators from PPT/Doc
        data['BB_Mid'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)

        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()

        return data

    @st.cache_data(ttl=300)
    def get_news(_self, query):
        """Fetches financial news using NewsAPI."""
        if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here":
            return []
        try:
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}&pageSize=10"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            return articles
        except Exception:
            return []

    @st.cache_data(ttl=900)
    def get_options_chain(_self, ticker):
        """Fetches the options chain for a given ticker."""
        try:
            stock = yf.Ticker(ticker)
            expiry_dates = stock.options
            if not expiry_dates:
                return None, None, None
            
            # Fetch chain for the nearest expiry
            chain = stock.option_chain(expiry_dates[0])
            return chain.calls, chain.puts, expiry_dates[0]
        except Exception:
            return None, None, None

    def analyze_candlesticks(self, data):
        """Identifies key candlestick patterns on the latest candle."""
        if len(data) < 2:
            return "Not enough data"
        
        last = data.iloc[-1]
        prev = data.iloc[-2]
        
        body = abs(last.Close - last.Open)
        upper_wick = last.High - max(last.Open, last.Close)
        lower_wick = min(last.Open, last.Close) - last.Low

        # Bullish Engulfing
        if last.Close > last.Open and prev.Close < prev.Open and \
           last.Close > prev.Open and last.Open < prev.Close:
            return "🟢 Bullish Engulfing"
        
        # Bearish Engulfing
        if last.Close < last.Open and prev.Close > prev.Open and \
           last.Close < prev.Open and last.Open > prev.Close:
            return "🔴 Bearish Engulfing"

        # Doji
        if body / (last.High - last.Low + 1e-6) < 0.1:
            return "🟡 Doji (Indecision)"
            
        # Hammer
        if lower_wick > 2 * body and upper_wick < body:
            return "🟢 Hammer (Potential Bullish Reversal)"

        # Shooting Star
        if upper_wick > 2 * body and lower_wick < body:
            return "🔴 Shooting Star (Potential Bearish Reversal)"

        return "⚪ No significant pattern"

    def get_ai_summary(self, ticker_name, analysis_results):
        """Generates an AI-powered trading strategy summary."""
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_api_key_here":
            return "Google API Key not configured. Please add it to your .env file."

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            As an expert intraday trading analyst, provide a concise, actionable trading strategy for {ticker_name} based on the following real-time data.
            Follow the provided trading notes strictly. The target audience is a trader who needs a clear, step-by-step plan.

            **Core Principles to Follow:**
            1.  **Trend is King:** Only suggest trades that align with the dominant trend identified from EMAs (5m and 15m charts).
            2.  **Confirmation is Crucial:** A trade signal requires confluence from at least 3 factors (e.g., candlestick pattern, support/resistance level, indicator confirmation). Never rely on a single indicator.
            3.  **Breakout and Retest:** For breakout trades, insist on waiting for a price retest of the broken level before suggesting entry. Avoid chasing initial breakouts.
            4.  **Risk Management:** Provide a clear Stop Loss (e.g., below the recent swing low for a long trade) and at least two Profit Targets (e.g., next resistance level, Fibonacci extension).
            5.  **Timeframe Analysis:** Synthesize insights from the 15-minute chart for context (trend, S/R levels) and the 5-minute chart for execution timing.

            **Real-time Analysis Data:**
            - **Overall Trend (15min):** {analysis_results['trend_15m']}
            - **Current Price:** {analysis_results['price']:.2f}
            - **Key Support Level (15min):** {analysis_results['support']:.2f}
            - **Key Resistance Level (15min):** {analysis_results['resistance']:.2f}

            **5-Minute Chart Execution Signals:**
            - **RSI:** {analysis_results['rsi_5m']:.2f}
            - **VWAP:** {analysis_results['vwap_5m']:.2f} (Price position relative to VWAP: {analysis_results['price_vs_vwap']})
            - **Latest Candlestick Pattern:** {analysis_results['candlestick_5m']}
            
            **News Headlines:**
            {analysis_results['news']}

            **Your Task:**
            Generate a trading plan in the following format:
            **1. Overall Market Bias:** (e.g., Bullish, Bearish, Neutral based on 15m trend).
            **2. Actionable Signal:** (e.g., "Look for a Long Entry", "Wait for a Short Setup", "Stay Sidelined").
            **3. Entry Strategy:** (e.g., "Enter above {price} after a bullish candle confirms the retest of the {level} support level.").
            **4. Stop Loss:** (e.g., "Place Stop Loss at {price}.").
            **5. Profit Targets:** (e.g., "Target 1: {price}, Target 2: {price}.").
            **6. Rationale:** (A brief justification combining trend, patterns, and indicators).
            """

            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"AI summary generation failed: {e}"
            
# ==============================================================================
# === UI & PLOTTING FUNCTIONS ==================================================
# ==============================================================================
def create_advanced_chart(data, ticker_name, interval):
    """Creates a comprehensive Plotly chart with multiple indicators."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=(f'Price Action ({ticker_name} - {interval})', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )

    # Price and EMAs
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='EMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    
    # Bollinger Bands and VWAP
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='magenta', width=1.5)), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='#00FF00')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')), row=3, col=1)
    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
    fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name='Histogram', marker_color=colors), row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

# ==============================================================================
# === MAIN APPLICATION LOGIC ===================================================
# ==============================================================================
def run_app():
    local_css()
    analyzer = AdvancedStockAnalyzer()
    
    st.sidebar.title(" Elite AI Trading Agent")
    st.sidebar.markdown("---")

    # --- Sidebar Controls ---
    st.sidebar.header("Market & Stock Selection")
    selected_category = st.sidebar.selectbox("Choose Market Index:", list(STOCK_CATEGORIES.keys()))
    
    analysis_type = st.sidebar.radio("Analysis Type:", ["Index Analysis", "Individual Stock"])

    if analysis_type == "Index Analysis":
        selected_stock_name = selected_category
        ticker = STOCK_CATEGORIES[selected_category]['ticker']
    else:
        stocks_in_category = STOCK_CATEGORIES[selected_category]['stocks']
        selected_stock_name = st.sidebar.selectbox("Choose Stock:", list(stocks_in_category.keys()))
        ticker = stocks_in_category[selected_stock_name]

    st.sidebar.success(f"**Selected:** {selected_stock_name} ({ticker})")
    st.sidebar.markdown("---")
    
    analyze_button = st.sidebar.button("🚀 Run Full Analysis", use_container_width=True)
    
    # --- Main Page Layout ---
    st.title(f"🤖 AI Trading Dashboard: {selected_stock_name}")

    if not GOOGLE_API_KEY or not NEWSAPI_KEY:
        st.error("API keys for Google Gemini or NewsAPI are missing. Please add them to your .env file.")
        st.stop()
        
    if not analyze_button:
        st.info("Select a stock or index from the sidebar and click 'Run Full Analysis' to begin.")
        st.stop()

    # --- Analysis Execution ---
    with st.spinner(f"Performing multi-timeframe analysis on {selected_stock_name}..."):
        # Fetch data for different timeframes
        data_15m, info = analyzer.get_stock_data(ticker, period="60d", interval="15m")
        data_5m, _ = analyzer.get_stock_data(ticker, period="5d", interval="5m")
        
        if data_15m is None or data_5m is None:
            st.error("Could not fetch sufficient data for analysis. The stock may not be available for intraday intervals.")
            st.stop()
            
        # Calculate indicators for both timeframes
        data_15m = analyzer.calculate_indicators(data_15m)
        data_5m = analyzer.calculate_indicators(data_5m)

        # Perform analysis
        news = analyzer.get_news(info.get('longName', selected_stock_name))
        calls, puts, expiry = analyzer.get_options_chain(ticker)

        # --- Display Results in Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 AI Strategy & Dashboard", "📈 Technical Charts", "⛓️ Options Chain", "📰 Latest News"])

        with tab1:
            st.header("💡 AI-Generated Trading Plan")
            
            # Prepare data for AI summary
            last_price = data_5m['Close'].iloc[-1]
            analysis_payload = {
                "trend_15m": "Bullish 📈" if data_15m['Close'].iloc[-1] > data_15m['EMA_50'].iloc[-1] else "Bearish 📉" if data_15m['Close'].iloc[-1] < data_15m['EMA_50'].iloc[-1] else "Sideways 횡",
                "price": last_price,
                "support": data_15m['Low'].rolling(20).min().iloc[-1],
                "resistance": data_15m['High'].rolling(20).max().iloc[-1],
                "rsi_5m": data_5m['RSI'].iloc[-1],
                "vwap_5m": data_5m['VWAP'].iloc[-1],
                "price_vs_vwap": "Above" if last_price > data_5m['VWAP'].iloc[-1] else "Below",
                "candlestick_5m": analyzer.analyze_candlesticks(data_5m),
                "news": "\n".join([f"- {n['title']}" for n in news[:3]])
            }

            with st.spinner("AI Agent is formulating a strategy..."):
                ai_summary = analyzer.get_ai_summary(selected_stock_name, analysis_payload)
                st.markdown(f"<div class='custom-container'>{ai_summary}</div>", unsafe_allow_html=True)
            
            st.header("📈 Live Market Dashboard")
            price_change = data_5m['Close'].iloc[-1] - data_5m['Close'].iloc[-2]
            price_change_pct = (price_change / data_5m['Close'].iloc[-2]) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"{last_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
            col2.metric("15m Trend", analysis_payload['trend_15m'])
            col3.metric("5m RSI", f"{analysis_payload['rsi_5m']:.2f}")
            col4.metric("5m Candlestick", analysis_payload['candlestick_5m'])
            
            st.header("📋 5-Step Intraday Checklist")
            st.markdown(f"1. **Support/Resistance:** Key support at **{analysis_payload['support']:.2f}**, resistance at **{analysis_payload['resistance']:.2f}**.")
            st.markdown(f"2. **Price Rejection:** Latest 5m candle shows **{analysis_payload['candlestick_5m']}**.")
            st.markdown(f"3. **Chart Pattern:** Price is currently **{analysis_payload['price_vs_vwap']}** the 5-min VWAP.")
            st.markdown(f"4. **Candlestick Confirmation:** Awaiting confirmation candle for a clear entry.")
            st.markdown(f"5. **Indicator Alignment:** RSI is at **{analysis_payload['rsi_5m']:.2f}**. Trend is **{analysis_payload['trend_15m']}**.")


        with tab2:
            st.header("Multi-Timeframe Technical Charts")
            st.info("Analyze the 15-minute chart for overall trend and the 5-minute chart for entry/exit signals.")
            
            st.plotly_chart(create_advanced_chart(data_15m, selected_stock_name, "15-Min"), use_container_width=True)
            st.plotly_chart(create_advanced_chart(data_5m, selected_stock_name, "5-Min"), use_container_width=True)

        with tab3:
            st.header(f"Options Chain Analysis (Expiry: {expiry})")
            if calls is None or puts is None:
                st.warning(f"Options data is not available for {selected_stock_name}.")
            else:
                pcr = puts['openInterest'].sum() / calls['openInterest'].sum()
                st.metric("Put-Call Ratio (OI)", f"{pcr:.2f}", "Bullish > 1, Bearish < 0.7")
                
                st.subheader("Calls - Blue Zone (In-the-Money)")
                itm_calls = calls[calls['inTheMoney']]
                st.dataframe(itm_calls[['lastTradeDate', 'strike', 'lastPrice', 'volume', 'openInterest']].style.applymap(lambda x: 'background-color: #003366'))
                
                st.subheader("Puts - Blue Zone (In-the-Money)")
                itm_puts = puts[puts['inTheMoney']]
                st.dataframe(itm_puts[['lastTradeDate', 'strike', 'lastPrice', 'volume', 'openInterest']].style.applymap(lambda x: 'background-color: #003366'))

        with tab4:
            st.header("Latest Financial News")
            if not news:
                st.warning("No recent news found.")
            else:
                for article in news:
                    with st.expander(f"{article['title']}"):
                        st.write(article['description'])
                        st.write(f"[Read More]({article['url']})")
                        st.caption(f"Published at: {article['publishedAt']}")

if __name__ == "__main__":
    run_app()

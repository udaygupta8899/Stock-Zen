import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os
import sys
from datetime import datetime, timedelta
from time import sleep
from dotenv import load_dotenv
from random import randint
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import optuna  # For hyperparameter tuning
from transformers import pipeline  # For FinBERT sentiment analysis
import feedparser
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the model directory to the path (if needed)
sys.path.append(os.path.abspath("model"))

# Load environment variables (e.g., for NEWS_API_KEY, GROQ_API_KEY)
load_dotenv()

st.set_page_config(
    page_title="StockZen",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary: #4FC3F7;
        --background: #0E1117;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-color: #ffffff;
        --hover-color: #4FC3F7;
    }
    .stApp {
        background: var(--background);
        color: var(--text-color);
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-card, .news-card, .prediction-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover, .prediction-card:hover, .news-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .prediction-up { color: #4CAF50; }
    .prediction-down { color: #F44336; }
    .prediction-neutral { color: #FFC107; }
    h1, h2, h3 {
        color: var(--hover-color) !important;
        margin-bottom: 1rem !important;
    }
    a {
        color: var(--hover-color);
        text-decoration: none;
    }
    a:hover { text-decoration: underline; }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--hover-color) 0%, transparent 100%);
        margin: 2rem 0;
    }
    .st-bb { background-color: transparent; }
    .st-at { background-color: var(--hover-color) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tuned models
if "tuned_models" not in st.session_state:
    st.session_state["tuned_models"] = {}

# Stock list
all_stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Oil & Natural Gas Corporation": "ONGC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "UPL": "UPL.NS"
    # ... add more stocks as needed.
}

def map_ticker_to_symbol(ticker):
    return all_stocks.get(ticker, None)

# Sidebar selections
st.sidebar.title("📈 StockZen")
st.sidebar.markdown("---")
selected_stock_name = st.sidebar.selectbox(
    "Select Company",
    list(all_stocks.keys()),
    format_func=lambda x: f"{x} ({all_stocks[x]})"
)
selected_stock = all_stocks[selected_stock_name]

st.sidebar.markdown("---")
selected_period = st.sidebar.selectbox(
    "Time Period (for display)",
    ["1d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=4
)

st.sidebar.markdown("---")
st.sidebar.caption("Chart Settings")
candlestick_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)
show_predictions = st.sidebar.checkbox("Show ML Predictions", value=True)

# Initialize the FinBERT sentiment analysis pipeline.
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="yiyanghkust/finbert-tone", 
                              tokenizer="yiyanghkust/finbert-tone")

@st.cache_data(ttl=600)
def fetch_stock_data(symbol, period):
    retry_count = 3
    for _ in range(retry_count):
        try:
            stock = yf.Ticker(symbol)
            if period == "1h":
                df = stock.history(period="1d", interval="1m")
                if df.empty:
                    st.warning("No data found for the last 1 hour. Trying with broader period.")
                    df = stock.history(period="1d", interval="5m")
            else:
                df = stock.history(period=period)
            df = df.reset_index()  # Ensure 'Date' is a column
            info = stock.info
            return df, info
        except Exception as e:
            st.warning(f"Error fetching data (attempting retry): {e}")
            sleep(randint(1, 3))
    st.error("Failed to fetch stock data after multiple attempts.")
    return None, None

@st.cache_data(ttl=600)
def fetch_full_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="max")
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"Error fetching full historical data: {e}")
        return None

@st.cache_data(ttl=1800)
def get_relevant_news(stock_name, ticker):
    query = "+".join(stock_name.split())
    rss_url = f"https://news.google.com/rss/search?q={query}+OR+{ticker}&hl=en-US&gl=US&ceid=US:en"
    try:
        news_feed = feedparser.parse(rss_url)
        filtered = []
        for entry in news_feed.entries:
            title = entry.get('title', '')
            desc = entry.get('summary', '')
            link = entry.get('link', '')
            pub_date = entry.get('published', '')
            if (any(word.lower() in title.lower() or word.lower() in desc.lower() for word in stock_name.split()) 
                or ticker.lower() in title.lower() 
                or ticker.lower() in desc.lower()):
                filtered.append({
                    'title': entry.title,
                    'description': entry.summary,
                    'link': link,
                    'published': pub_date
                })
        return filtered[:10]
    except Exception as e:
        return []

@st.cache_resource
def load_groq_api_key():
    return os.getenv("GROQ_API_KEY", "your_api_key_here")

GROQ_API_KEY = load_groq_api_key()

def get_news_sentiment_with_impact(text):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    HEADERS = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    prompt = f'''
    You are a financial analyst. Analyze the sentiment of the following stock market news.
    Classify it as **Positive, Negative, or Neutral** and provide an impact score from -5 to 5.
    
    News: {text}
    
    Return output in the format: 
    Sentiment: <Positive/Negative/Neutral>
    Impact: <impact_score>
    '''
    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        response = requests.post(API_URL, json=data, headers=HEADERS)
        response.raise_for_status()
        output = response.json()["choices"][0]["message"]["content"]
        sentiment, impact = None, 0.0
        for line in output.split("\n"):
            if "Sentiment:" in line:
                sentiment = line.split(":")[-1].strip()
            if "Impact:" in line:
                try:
                    impact = float(line.split(":")[-1].strip())
                except ValueError:
                    impact = 0.0
        return sentiment, impact
    except Exception as e:
        return "Error", 0.0

def sentiment_color(label):
    if label.lower() == "positive":
        return "#4CAF50"
    elif label.lower() == "negative":
        return "#F44336"
    else:
        return "#9E9E9E"

def compute_daily_sentiment(stock_name, ticker):
    news_articles = get_relevant_news(stock_name, ticker)
    if not news_articles:
        return pd.DataFrame(columns=["Date", "sentiment"])
    news_df = pd.DataFrame(news_articles)
    try:
        news_df["Date"] = pd.to_datetime(news_df["published"])
    except Exception as e:
        news_df["Date"] = pd.Timestamp(datetime.now().date())
    daily_sent = news_df.groupby(news_df["Date"].dt.date).apply(
        lambda df: np.mean([
            sentiment_pipeline(row)[0]["score"] if sentiment_pipeline(row)[0]["label"].lower() == "positive" 
            else -sentiment_pipeline(row)[0]["score"] if sentiment_pipeline(row)[0]["label"].lower() == "negative"
            else 0.0
            for row in df["title"]
        ])
    ).reset_index().rename(columns={0: "sentiment", "Date": "Date"})
    daily_sent["Date"] = pd.to_datetime(daily_sent["Date"])
    return daily_sent

def add_daily_sentiment_feature(data_df, stock_name, ticker):
    data_df['Date'] = pd.to_datetime(data_df['Date']).dt.date
    daily_sent = compute_daily_sentiment(stock_name, ticker)
    if daily_sent.empty:
        data_df['sentiment'] = 0.0
    else:
        daily_sent['Date'] = pd.to_datetime(daily_sent['Date']).dt.date
        data_df = data_df.merge(daily_sent, on='Date', how='left')
        data_df['sentiment'] = data_df['sentiment'].fillna(method='ffill').fillna(0.0)
    return data_df

def tune_and_train_model(stock_symbol, data_df):
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    data_df = add_daily_sentiment_feature(data_df, selected_stock_name, stock_symbol)
    
    data_df["lag1"] = data_df["Close"].shift(1)
    data_df["lag2"] = data_df["Close"].shift(2)
    data_df["ma5"] = data_df["Close"].rolling(window=5).mean()
    data_df["ma10"] = data_df["Close"].rolling(window=10).mean()
    data_df["day_of_week"] = pd.to_datetime(data_df["Date"]).dt.dayofweek
    data_df = data_df.dropna().reset_index(drop=True)
    if data_df.empty:
        st.error("Not enough data to generate features for training.")
        return None, None

    features = ["lag1", "lag2", "ma5", "ma10", "day_of_week", "sentiment"]
    X = data_df[features]
    y = data_df["Close"]

    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(X)

    n = len(y)
    split_index = int(n * 0.8)
    X_train, X_valid = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_valid = y[:split_index], y[split_index:]

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50)
        }
        model = lgb.LGBMRegressor(
            **params,
            objective='regression',
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50)],
            
        )
        y_pred = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_trial.params
    best_model = lgb.LGBMRegressor(
        **best_params,
        objective='regression',
        random_state=42
    )
    best_model.fit(X_scaled, y, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(50)])
    return best_model, scaler_local

def get_gb_predictions(stock_symbol, data_df, tuned_model, scaler_local):
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    data_df = add_daily_sentiment_feature(data_df, selected_stock_name, stock_symbol)
    data_df["lag1"] = data_df["Close"].shift(1)
    data_df["lag2"] = data_df["Close"].shift(2)
    data_df["ma5"] = data_df["Close"].rolling(window=5).mean()
    data_df["ma10"] = data_df["Close"].rolling(window=10).mean()
    data_df["day_of_week"] = pd.to_datetime(data_df["Date"]).dt.dayofweek
    data_df = data_df.dropna().reset_index(drop=True)
    if data_df.empty:
        st.error("Not enough data for prediction.")
        return None

    base_price = data_df["Close"].iloc[-1]
    last_row = data_df.iloc[-1].copy()
    current_date = pd.to_datetime(last_row["Date"])
    today_date = pd.Timestamp(datetime.now().date())
    if current_date < today_date:
        current_date = today_date
    current_close = last_row["Close"]
    history = data_df["Close"].values[-10:].tolist()

    pred_dates = []
    pred_prices = []
    for _ in range(7):
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() > 4:
            next_date += timedelta(days=1)
        lag1 = current_close
        lag2 = history[-2] if len(history) >= 2 else current_close
        ma5 = np.mean(history[-5:]) if len(history) >= 5 else current_close
        ma10 = np.mean(history[-10:]) if len(history) >= 10 else current_close
        day_of_week = next_date.weekday()
        sentiment_value = data_df[data_df["Date"] == next_date.date()]["sentiment"]
        if sentiment_value.empty:
            sentiment_value = data_df["sentiment"].iloc[-1]
        else:
            sentiment_value = sentiment_value.iloc[0]
        X_new = np.array([[lag1, lag2, ma5, ma10, day_of_week, sentiment_value]])
        X_new_scaled = scaler_local.transform(X_new)
        next_close = tuned_model.predict(X_new_scaled)[0]
        pred_dates.append(next_date)
        pred_prices.append(next_close)
        history.append(next_close)
        current_close = next_close
        current_date = next_date

    pred_df = pd.DataFrame({
        "Date": pred_dates,
        "Predicted_Close": pred_prices
    })
    if not pred_df.empty:
        returns = []
        for idx, price in enumerate(pred_df["Predicted_Close"]):
            if idx == 0:
                returns.append((price - base_price) / base_price)
            else:
                prev_price = pred_df["Predicted_Close"].iloc[idx - 1]
                returns.append((price - prev_price) / prev_price)
        pred_df["Predicted_Return"] = returns
    return pred_df

def generate_sentiment_from_predictions(predictions, base_price):
    if predictions is None or predictions.empty:
        return None
    last_price = predictions['Predicted_Close'].iloc[-1]
    overall_change = (last_price - base_price) / base_price * 100
    if overall_change > 2:
        return {"sentiment": "positive", "change": f"+{overall_change:.2f}%", "class": "prediction-up"}
    elif overall_change < -2:
        return {"sentiment": "negative", "change": f"{overall_change:.2f}%", "class": "prediction-down"}
    else:
        return {"sentiment": "neutral", "change": f"{overall_change:.2f}%", "class": "prediction-neutral"}

def evaluate_model(tuned_model, scaler_local, data_df, stock_name, ticker):
    data = data_df.copy()
    data = add_daily_sentiment_feature(data, stock_name, ticker)
    data["lag1"] = data["Close"].shift(1)
    data["lag2"] = data["Close"].shift(2)
    data["ma5"] = data["Close"].rolling(window=5).mean()
    data["ma10"] = data["Close"].rolling(window=10).mean()
    data["day_of_week"] = pd.to_datetime(data["Date"]).dt.dayofweek
    data = data.dropna().reset_index(drop=True)
    features = ["lag1", "lag2", "ma5", "ma10", "day_of_week", "sentiment"]
    X = data[features]
    y = data["Close"]
    X_scaled = scaler_local.transform(X)
    n = len(data)
    split_index = int(n * 0.8)
    X_valid = X_scaled[split_index:]
    y_valid = y[split_index:]
    y_pred = tuned_model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    valid_dates = data["Date"].iloc[split_index:].reset_index(drop=True)
    backtest_df = pd.DataFrame({
          "Date": valid_dates,
          "Actual": y_valid.values,
          "Predicted": y_pred
    })
    error_std = np.std(y_valid - y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "error_std": error_std}, backtest_df

def main():
    st.title(f"{selected_stock_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.spinner('Loading market data for display...'):
        df_display, info = fetch_stock_data(selected_stock, selected_period)
    if df_display is None or df_display.empty:
        st.warning("No data available for the selected stock")
        return

    with st.spinner('Loading full historical data for model training...'):
        df_full = fetch_full_stock_data(selected_stock)
    if df_full is None or df_full.empty:
        st.warning("No full historical data available for model training")
        return

    st.subheader("Key Metrics")
    cols = st.columns(4)
    metrics = [
        ("Current Price", f"₹{df_display['Close'].iloc[-1]:,.2f}"),
        ("Market Cap", f"₹{info.get('marketCap', 0)/1e7:,.1f} Cr"),
        ("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}"),
        ("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}")
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_display["Date"],
        open=df_display['Open'],
        high=df_display['High'],
        low=df_display['Low'],
        close=df_display['Close'],
        name='Price'
    ))
    if candlestick_ma:
        for days, color in [(20, '#FFA726'), (50, '#26C6DA')]:
            ma = df_display['Close'].rolling(days).mean()
            fig.add_trace(go.Scatter(
                x=df_display["Date"],
                y=ma,
                name=f'{days} MA',
                line=dict(color=color, width=2)
            ))
    if show_bollinger:
        window = 20
        sma = df_display['Close'].rolling(window).mean()
        std = df_display['Close'].rolling(window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        fig.add_trace(go.Scatter(
            x=df_display["Date"],
            y=sma,
            line=dict(color='#FF6F00', width=1.5),
            name='Bollinger Middle (20 SMA)'
        ))
        fig.add_trace(go.Scatter(
            x=df_display["Date"],
            y=upper_band,
            line=dict(color='#4CAF50', width=1.5),
            name='Upper Band (2σ)',
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=df_display["Date"],
            y=lower_band,
            line=dict(color='#F44336', width=1.5),
            name='Lower Band (2σ)',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))
    fig.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if show_rsi:
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        rsi = calculate_rsi(df_display)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_display["Date"],
            y=rsi,
            line=dict(color='#8A2BE2', width=2),
            name='RSI'
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.update_layout(
            height=400,
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="RSI"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.subheader("Trading Volume")
    fig_vol = go.Figure(go.Bar(
        x=df_display["Date"],
        y=df_display['Volume'],
        marker=dict(color='rgba(255, 99, 132, 0.6)'),
        name="Volume"
    ))
    fig_vol.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    if show_predictions:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Price Predictions (Next 7 Days)")
        if selected_stock not in st.session_state["tuned_models"]:
            with st.spinner("Tuning and training model (this may take some time)..."):
                tuned_model, scaler_local = tune_and_train_model(selected_stock, df_full)
                if tuned_model is None:
                    st.error("Model training failed.")
                    return
                st.session_state["tuned_models"][selected_stock] = (tuned_model, scaler_local)
        else:
            tuned_model, scaler_local = st.session_state["tuned_models"][selected_stock]
        
        eval_metrics, backtest_df = evaluate_model(tuned_model, scaler_local, df_full, selected_stock_name, selected_stock)
        st.subheader("Model Performance")
        st.markdown(f"""
        **Evaluation Metrics:**
        - **RMSE:** {eval_metrics['rmse']:.2f}
        - **MAE:** {eval_metrics['mae']:.2f}
        """)
        
        backtest_df["Upper"] = backtest_df["Predicted"] + eval_metrics["error_std"]
        backtest_df["Lower"] = backtest_df["Predicted"] - eval_metrics["error_std"]
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["Actual"], mode='lines', name="Actual Price"))
        fig_eval.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["Predicted"], mode='lines', name="Predicted Price"))
        fig_eval.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["Upper"], mode='lines', line=dict(color='lightgrey'), name="Upper Bound"))
        fig_eval.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["Lower"], mode='lines', line=dict(color='lightgrey'), name="Lower Bound", fill='tonexty'))
        fig_eval.update_layout(title="Backtesting: Actual vs Predicted Prices", template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_eval, use_container_width=True)
        
        
        
        with st.spinner("Generating predictions..."):
            predictions = get_gb_predictions(selected_stock, df_full, tuned_model, scaler_local)
        if predictions is not None and not predictions.empty:
            st.markdown("### Detailed Daily Predictions")
            predictions["Return Indicator"] = predictions["Predicted_Return"].apply(
                lambda x: f"▲ {x:.2%}" if x > 0 else (f"▼ {x:.2%}" if x < 0 else f"{x:.2%}")
            )
            predictions_table = pd.DataFrame({
                'Date': predictions['Date'].dt.strftime('%Y-%m-%d'),
                'Predicted Close': predictions['Predicted_Close'].map('₹{:,.2f}'.format),
                'Daily Return': predictions["Return Indicator"]
            })
            st.table(predictions_table)
            
            base_price = df_full["Close"].iloc[-1]
            expected_change = (predictions["Predicted_Close"].iloc[-1] - base_price) / base_price * 100
            sentiment_summary = generate_sentiment_from_predictions(predictions, base_price)
            sentiment_cols = st.columns(3)
            with sentiment_cols[0]:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Trend</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment_summary['class']}">
                        {sentiment_summary['sentiment'].title()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with sentiment_cols[1]:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Expected Change</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment_summary['class']}">
                        {expected_change:+.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with sentiment_cols[2]:
                target_price = predictions['Predicted_Close'].iloc[-1]
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>7-Day Target Price</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment_summary['class']}">
                        ₹{target_price:,.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.line_chart(predictions.set_index("Date")["Predicted_Close"])
        else:
            st.warning("Predictions are not available for this stock. Please check back later.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Latest News")
    with st.spinner("Loading news..."):
        news_articles = get_relevant_news(selected_stock_name, selected_stock)
    if news_articles:
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            sentiment_label, impact = get_news_sentiment_with_impact(title)
            color = sentiment_color(sentiment_label)
            st.markdown(f"""
            <div class="news-card">
                <h3><a href="{url}" target="_blank">{title}</a></h3>
                <p>{description}</p>
                <div style="border-radius: 8px; padding: 4px 8px; background-color: {color}; display: inline-block; color: white; margin-top: 4px;">
                    {sentiment_label} (Impact: {impact:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No news found for the selected stock.")

if __name__ == "__main__":
    main()
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

# Add the model directory to the path (if needed)
sys.path.append(os.path.abspath("model"))

# Load environment variables (e.g., for NEWS_API_KEY)
load_dotenv()

st.set_page_config(
    page_title="Stock Dashboard",
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

# Stock list (unchanged)
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
    "UPL": "UPL.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Adani Ports & SEZ": "ADANIPORTS.NS",
    "Grasim Industries": "GRASIM.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Shree Cement": "SHREECEM.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Titan Company": "TITAN.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Coal India": "COALINDIA.NS",
    "Bharat Petroleum": "BPCL.NS",
    "GAIL": "GAIL.NS",
    "Indian Oil Corporation": "IOC.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Total Gas": "ADANIGAS.NS",
    "Adani Power": "ADANIPOWER.NS",
    "Torrent Pharmaceuticals": "TORNTPHARM.NS",
    "Cipla": "CIPLA.NS",
    "Biocon": "BIOCON.NS",
    "Lupin": "LUPIN.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Dabur India": "DABUR.NS",
    "Godrej Consumer Products": "GODREJCP.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "ICICI Prudential Life Insurance": "ICICIPRULI.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Marico": "MARICO.NS",
    "Voltas": "VOLTAS.NS",
    "Siemens India": "SIEMENS.NS",
    "Ashok Leyland": "ASHOKLEY.NS",
    "Bharat Electronics": "BEL.NS",
    "Bharat Heavy Electricals": "BHEL.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "Vodafone Idea": "IDEA.NS",
    "Motherson Sumi Systems": "MOTHERSUMI.NS",
    "JSW Energy": "JSWENERGY.NS",
    "Container Corporation of India": "CONCOR.NS",
    "Canara Bank": "CANBK.NS",
    "Punjab National Bank": "PNB.NS",
    "Union Bank of India": "UNIONBANK.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "RBL Bank": "RBLBANK.NS",
    "Yes Bank": "YESBANK.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "DLF": "DLF.NS",
    "IRCTC": "IRCTC.NS",
    "M&M Financial Services": "M&MFIN.NS",
    "Tata Communications": "TATACOMM.NS",
    "Adani Wilmar": "ADANIWILMAR.NS",
    "Hindustan Petroleum": "HINDPETRO.NS",
    "Bharat Forge": "BHARATFORG.NS",
    "Crompton Greaves": "CGPOWER.NS",
    "Berger Paints": "BERGEPAINT.NS",
    "NHPC": "NHPC.NS",
    "Tata Power": "TATAPOWER.NS",
    "Can Fin Homes": "CANFINHOME.NS",
    "Dixon Technologies": "DIXON.NS",
    "Aarti Industries": "AARTIIND.NS",
    "Adani Transmission": "ADANITRANS.NS",
    "Shriram Transport Finance": "SHRIRAMTF.NS",
    "PVR": "PVR.NS",
    "Jubilant FoodWorks": "JUBLFOOD.NS",
    "GMR Infra": "GMRINFRA.NS",
    "Finolex Industries": "FINOLEXIND.NS",
    "Cummins India": "CUMMINSIND.NS",
    "Emami": "EMAMILTD.NS",
    "Future Retail": "FRETAIL.NS",
    "IndiaMART InterMESH": "INDIAMART.NS"
}


def map_ticker_to_symbol(ticker):
    symbol_map = {
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
        "UPL": "UPL.NS",
        "Bajaj Auto": "BAJAJ-AUTO.NS",
        "Adani Ports & SEZ": "ADANIPORTS.NS",
        "Grasim Industries": "GRASIM.NS",
        "Divi's Laboratories": "DIVISLAB.NS",
        "Apollo Hospitals": "APOLLOHOSP.NS",
        "Shree Cement": "SHREECEM.NS",
        "JSW Steel": "JSWSTEEL.NS",
        "Titan Company": "TITAN.NS",
        "Hindalco Industries": "HINDALCO.NS",
        "Coal India": "COALINDIA.NS",
        "Bharat Petroleum": "BPCL.NS",
        "GAIL": "GAIL.NS",
        "Indian Oil Corporation": "IOC.NS",
        "Adani Enterprises": "ADANIENT.NS",
        "Adani Green Energy": "ADANIGREEN.NS",
        "Adani Total Gas": "ADANIGAS.NS",
        "Adani Power": "ADANIPOWER.NS",
        "Torrent Pharmaceuticals": "TORNTPHARM.NS",
        "Cipla": "CIPLA.NS",
        "Biocon": "BIOCON.NS",
        "Lupin": "LUPIN.NS",
        "Britannia Industries": "BRITANNIA.NS",
        "Dabur India": "DABUR.NS",
        "Godrej Consumer Products": "GODREJCP.NS",
        "SBI Life Insurance": "SBILIFE.NS",
        "ICICI Prudential Life Insurance": "ICICIPRULI.NS",
        "LIC Housing Finance": "LICHSGFIN.NS",
        "Shriram Finance": "SHRIRAMFIN.NS",
        "Pidilite Industries": "PIDILITIND.NS",
        "Tata Consumer Products": "TATACONSUM.NS",
        "Marico": "MARICO.NS",
        "Voltas": "VOLTAS.NS",
        "Siemens India": "SIEMENS.NS",
        "Ashok Leyland": "ASHOKLEY.NS",
        "Bharat Electronics": "BEL.NS",
        "Bharat Heavy Electricals": "BHEL.NS",
        "Jindal Steel & Power": "JINDALSTEL.NS",
        "Vodafone Idea": "IDEA.NS",
        "Motherson Sumi Systems": "MOTHERSUMI.NS",
        "JSW Energy": "JSWENERGY.NS",
        "Container Corporation of India": "CONCOR.NS",
        "Canara Bank": "CANBK.NS",
        "Punjab National Bank": "PNB.NS",
        "Union Bank of India": "UNIONBANK.NS",
        "IDFC First Bank": "IDFCFIRSTB.NS",
        "RBL Bank": "RBLBANK.NS",
        "Yes Bank": "YESBANK.NS",
        "Bandhan Bank": "BANDHANBNK.NS",
        "DLF": "DLF.NS",
        "IRCTC": "IRCTC.NS",
        "M&M Financial Services": "M&MFIN.NS",
        "Tata Communications": "TATACOMM.NS",
        "Adani Wilmar": "ADANIWILMAR.NS",
        "Hindustan Petroleum": "HINDPETRO.NS",
        "Bharat Forge": "BHARATFORG.NS",
        "Crompton Greaves": "CGPOWER.NS",
        "Berger Paints": "BERGEPAINT.NS",
        "NHPC": "NHPC.NS",
        "Tata Power": "TATAPOWER.NS",
        "Can Fin Homes": "CANFINHOME.NS",
        "Dixon Technologies": "DIXON.NS",
        "Aarti Industries": "AARTIIND.NS",
        "Adani Transmission": "ADANITRANS.NS",
        "Shriram Transport Finance": "SHRIRAMTF.NS",
        "PVR": "PVR.NS",
        "Jubilant FoodWorks": "JUBLFOOD.NS",
        "GMR Infra": "GMRINFRA.NS",
        "Finolex Industries": "FINOLEXIND.NS",
        "Cummins India": "CUMMINSIND.NS",
        "Emami": "EMAMILTD.NS",
        "Future Retail": "FRETAIL.NS",
        "IndiaMART InterMESH": "INDIAMART.NS"
    }

    return all_stocks.get(ticker, None)

# Sidebar selections (unchanged)
st.sidebar.title("📈 Stock Dashboard")
st.sidebar.markdown("---")
selected_stock_name = st.sidebar.selectbox(
    "Select Company",
    list(all_stocks.keys()),
    format_func=lambda x: f"{x} ({all_stocks[x]})"
)
selected_stock = all_stocks[selected_stock_name]

st.sidebar.markdown("---")
# For display, let the user select a period (for charts/metrics)
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

# Caching stock data for display (using selected period)
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

# For model training, fetch all available data (i.e. period="max")
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
    news_api_key = os.getenv("NEWS_API_KEY", "your_news_api_key_here")
    full_name = stock_name
    query = f'"{full_name}" OR "{ticker}"'
    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': news_api_key,
        'from': date_from,
        'qInTitle': stock_name
    }
    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        filtered = []
        for article in articles:
            title = article.get('title', '').lower() if article.get('title') else ""
            desc = article.get('description', '').lower() if article.get('description') else ""
            if any([full_name.lower() in title, ticker.lower() in title, full_name.lower() in desc, ticker.lower() in desc]):
                filtered.append(article)
        return filtered[:5]
    except Exception as e:
        st.error(f"News API Error: {e}")
        return []

# Use st.session_state to cache tuned models per stock
if "tuned_models" not in st.session_state:
    st.session_state.tuned_models = {}

# Refresh button to clear the cached tuned model for the selected stock
if st.sidebar.button("Refresh Model Cache"):
    if selected_stock in st.session_state.tuned_models:
        del st.session_state.tuned_models[selected_stock]
        st.success(f"Model cache for {selected_stock} refreshed.")

def tune_and_train_model(stock_symbol, data_df):
    """
    Generate features, scale them, and use Optuna to tune a LightGBM model.
    Returns the tuned model and the fitted StandardScaler.
    """
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    # Feature Engineering
    data_df["lag1"] = data_df["Close"].shift(1)
    data_df["lag2"] = data_df["Close"].shift(2)
    data_df["ma5"] = data_df["Close"].rolling(window=5).mean()
    data_df["ma10"] = data_df["Close"].rolling(window=10).mean()
    data_df["day_of_week"] = pd.to_datetime(data_df["Date"]).dt.dayofweek
    data_df = data_df.dropna().reset_index(drop=True)
    if data_df.empty:
        st.error("Not enough data to generate features for training.")
        return None, None

    features = ["lag1", "lag2", "ma5", "ma10", "day_of_week"]
    X = data_df[features]
    y = data_df["Close"]

    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(X)

    # Use the first 80% of the data for training and the remaining 20% for validation
    n = len(y)
    split_index = int(n * 0.8)
    X_train, X_valid = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_valid = y[:split_index], y[split_index:]

    def objective(trial):
        num_leaves = trial.suggest_int("num_leaves", 20, 100)
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.2)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 10)

        model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            objective='regression',
            random_state=42
        )
        # Fit without early_stopping_rounds
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        y_pred = model.predict(X_valid)
        rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)  # Increase n_trials for better results if needed

    # Do not display best parameters on the website.
    best_model = lgb.LGBMRegressor(
        **study.best_trial.params,
        objective='regression',
        random_state=42
    )
    best_model.fit(X_scaled, y)
    return best_model, scaler_local

def get_gb_predictions(stock_symbol, data_df, tuned_model, scaler_local):
    """
    Using the tuned model and scaler, perform recursive forecasting for the next 7 days.
    For the first predicted day, compute return relative to the base price (last historical close),
    and for each subsequent day, compute the daily return as the percentage change from the previous predicted day.
    """
    data_df = data_df.sort_values("Date").reset_index(drop=True)
    data_df["lag1"] = data_df["Close"].shift(1)
    data_df["lag2"] = data_df["Close"].shift(2)
    data_df["ma5"] = data_df["Close"].rolling(window=5).mean()
    data_df["ma10"] = data_df["Close"].rolling(window=10).mean()
    data_df["day_of_week"] = pd.to_datetime(data_df["Date"]).dt.dayofweek
    data_df = data_df.dropna().reset_index(drop=True)
    if data_df.empty:
        st.error("Not enough data for prediction.")
        return None

    base_price = data_df["Close"].iloc[-1]  # Last historical close as baseline
    last_row = data_df.iloc[-1].copy()
    current_date = pd.to_datetime(last_row["Date"]).tz_localize(None)
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
        X_new = np.array([[lag1, lag2, ma5, ma10, day_of_week]])
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
    # Compute daily returns:
    # For the first day, return is relative to the base price.
    # For subsequent days, return is relative to the previous predicted price.
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
    """
    Compute the overall expected percentage change as:
    (Predicted 7th day price - current price) / current price * 100
    and generate a sentiment dictionary.
    """
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

def main():
    st.title(f"{selected_stock_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Fetch data for display using the user-selected period
    with st.spinner('Loading market data for display...'):
        df_display, info = fetch_stock_data(selected_stock, selected_period)
    if df_display is None or df_display.empty:
        st.warning("No data available for the selected stock")
        return

    # For model training and predictions, fetch all available historical data
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
    
    # Display Price Movement chart using display data
    st.subheader("Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_display.index,
        open=df_display['Open'],
        high=df_display['High'],
        low=df_display['Low'],
        close=df_display['Close'],
        name='Price',
    ))
    if candlestick_ma:
        for days, color in [(20, '#FFA726'), (50, '#26C6DA')]:
            ma = df_display['Close'].rolling(days).mean()
            fig.add_trace(go.Scatter(
                x=df_display.index,
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
            x=df_display.index,
            y=sma,
            line=dict(color='#FF6F00', width=1.5),
            name='Bollinger Middle (20 SMA)'
        ))
        fig.add_trace(go.Scatter(
            x=df_display.index,
            y=upper_band,
            line=dict(color='#4CAF50', width=1.5),
            name='Upper Band (2σ)',
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=df_display.index,
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
            x=df_display.index,
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
        x=df_display.index,
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
        # Retrieve or tune and train a model for the selected stock using Optuna on full historical data
        if selected_stock not in st.session_state.tuned_models:
            with st.spinner("Tuning and training model (this may take some time)..."):
                tuned_model, scaler_local = tune_and_train_model(selected_stock, df_full)
                if tuned_model is None:
                    st.error("Model training failed.")
                    return
                st.session_state.tuned_models[selected_stock] = (tuned_model, scaler_local)
        else:
            tuned_model, scaler_local = st.session_state.tuned_models[selected_stock]

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
            
            # Calculate overall expected change: (7th day predicted price - current price)/current price * 100
            base_price = df_full["Close"].iloc[-1]
            expected_change = (predictions["Predicted_Close"].iloc[-1] - base_price) / base_price * 100
            sentiment = generate_sentiment_from_predictions(predictions, base_price)
            sentiment_cols = st.columns(3)
            with sentiment_cols[0]:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Trend</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                        {sentiment['sentiment'].title()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with sentiment_cols[1]:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Expected Change</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                        {expected_change:+.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            with sentiment_cols[2]:
                target_price = predictions['Predicted_Close'].iloc[-1]
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>7-Day Target Price</h3>
                    <p style="font-size: 1.5rem; margin: 0.5rem 0;" class="{sentiment['class']}">
                        ₹{target_price:,.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.line_chart(predictions.set_index("Date")["Predicted_Close"])
        else:
            st.warning("Predictions are not available for this stock. Please check if there is enough data for training.")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.subheader("Latest News")
    with st.spinner("Loading news..."):
        news_articles = get_relevant_news(selected_stock_name, selected_stock)
    if news_articles:
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            st.markdown(f"""
            <div class="news-card">
                <h3><a href="{url}" target="_blank">{title}</a></h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No news found for the selected stock.")

if __name__ == "__main__":
    main()

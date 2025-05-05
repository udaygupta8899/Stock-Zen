import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import lightgbm as lgb
import xgboost as xgb
import optuna
from transformers import pipeline
import feedparser
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")

# Setup & CSS theme
load_dotenv()
st.set_page_config(page_title="StockZen", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    :root {
        --primary: #4FC3F7;
        --background: #0E1117;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-color: #ffffff;
        --hover-color: #4FC3F7;
    }
    .stApp { background: var(--background); color: var(--text-color); font-family: 'Segoe UI', sans-serif; }
    .metric-card, .news-card, .prediction-card { background: var(--card-bg); border-radius:16px; padding:1.5rem; margin:1rem 0; border:1px solid rgba(255,255,255,0.1); transition:transform .3s ease, box-shadow .3s ease; }
    .metric-card:hover, .prediction-card:hover, .news-card:hover { transform: translateY(-5px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .prediction-up { color: #4CAF50; } .prediction-down { color: #F44336; } .prediction-neutral { color: #FFC107; }
    h1,h2,h3 { color: var(--hover-color) !important; margin-bottom:1rem !important; }
    a { color: var(--hover-color); text-decoration:none; }
    a:hover { text-decoration:underline; }
    .divider { height:2px; background: linear-gradient(90deg, var(--hover-color) 0%, transparent 100%); margin:2rem 0; }
    .st-bb { background-color: transparent; } .st-at { background-color: var(--hover-color) !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar: stock selection & chart options
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
    "Asian Paints": "ASIANPAINTS.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Wipro": "WIPRO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "UPL": "UPL.NS"
}

st.sidebar.title("📈 StockZen")
st.sidebar.markdown("---")
selected_name = st.sidebar.selectbox(
    "Select Company",
    list(all_stocks.keys()),
    format_func=lambda x: f"{x} ({all_stocks[x]})"
)
ticker = all_stocks[selected_name]

st.sidebar.markdown("---")
period = st.sidebar.selectbox("Time Period", ["1d","1wk","1mo","3mo","6mo","1y","2y","5y","max"], index=4)

st.sidebar.markdown("---")
st.sidebar.caption("Chart Settings")
candlestick_ma = st.sidebar.checkbox("Show Moving Averages", True)
show_boll = st.sidebar.checkbox("Show Bollinger Bands", False)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_preds = st.sidebar.checkbox("Show ML Predictions", True)

# Sentiment pipelines & API
finbert = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone", tokenizer="yiyanghkust/finbert-tone")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def get_news_sentiment_with_impact(text: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    prompt = (
        f"You are a financial analyst. Classify sentiment and score impact [-5 to +5].\n"
        f"News: {text}\n"
        "Return:\n"
        "Sentiment: <Positive/Negative/Neutral>\n"
        "Impact: <number>\n"
    )
    try:
        r = requests.post(url, json={
            "model":"llama3-8b-8192",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.2
        }, headers=headers)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        sent, imp = "Neutral", 0.0
        for line in content.splitlines():
            if line.lower().startswith("sentiment"):
                sent = line.split(":",1)[1].strip()
            if line.lower().startswith("impact"):
                try: imp = float(line.split(":",1)[1].strip())
                except: imp = 0.0
        return sent, imp
    except:
        return "Neutral", 0.0

def sentiment_color(lbl: str):
    return {"positive":"#4CAF50","negative":"#F44336"}.get(lbl.lower(), "#9E9E9E")

# Fetch & enrich news
@st.cache_data(ttl=1800)
def fetch_and_enrich_news(name: str, sym: str) -> pd.DataFrame:
    query = "+".join(name.split())
    rss = f"https://news.google.com/rss/search?q={query}+OR+{sym}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss)
    rows = []
    for entry in feed.entries[:15]:
        pub = entry.get("published", "")
        ts = pd.to_datetime(pub, format="%a, %d %b %Y %H:%M:%S GMT", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(pub, errors="coerce", utc=True)
        title = entry.get("title", "")
        desc = entry.get("summary", "")
        link = entry.get("link", "")
        fin = finbert(title)[0]
        fin_score = fin["score"] if fin["label"]=="POSITIVE" else -fin["score"]
        groq_lbl, groq_imp = get_news_sentiment_with_impact(title)
        rows.append({
            "published": ts,
            "title": title,
            "description": desc,
            "link": link,
            "finbert_score": fin_score,
            "groq_impact": groq_imp
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["published"])

# Compute combined daily sentiment
def compute_daily_sentiment(name: str, sym: str, alpha: float = 0.6) -> pd.DataFrame:
    df = fetch_and_enrich_news(name, sym)
    if df.empty:
        return pd.DataFrame(columns=["Date","daily_score"])
    df["Date"] = df["published"].dt.date
    agg = df.groupby("Date").agg({
        "finbert_score":"mean",
        "groq_impact": "mean"
    }).reset_index()
    agg["daily_score"] = alpha*agg["finbert_score"] + (1-alpha)*agg["groq_impact"]
    agg["Date"] = pd.to_datetime(agg["Date"])
    return agg[["Date","daily_score"]]

# Fetch stock data & feature engineering
@st.cache_data(ttl=600)
def fetch_stock_data(sym: str, per: str):
    try:
        t = yf.Ticker(sym)
        df = t.history(period=per).reset_index()
        if df.empty:
            st.warning(f"No data retrieved for {sym} with period {per}. Please try a different ticker or period.")
            return pd.DataFrame(), {}
        info = t.info or {}
        return df, info
    except Exception as e:
        st.error(f"Error fetching data for {sym}: {str(e)}")
        return pd.DataFrame(), {}

@st.cache_data(ttl=600)
def fetch_full_history(sym: str):
    try:
        t = yf.Ticker(sym)
        df = t.history(period="max").reset_index()
        if df.empty:
            st.warning(f"No historical data retrieved for {sym}. Please select another ticker.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error fetching historical data for {sym}: {str(e)}")
        return pd.DataFrame()

def add_features(df: pd.DataFrame, name: str, sym: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["Date"] = df["Date"].dt.date
    sent = compute_daily_sentiment(name, sym)
    sent["Date"] = sent["Date"].dt.date
    df = df.merge(sent, on="Date", how="left")
    df["daily_score"] = df["daily_score"].fillna(method="ffill").fillna(0.0)
    df["lag1"] = df["Close"].shift(1)
    df["lag2"] = df["Close"].shift(2)
    df["ma5"] = df["Close"].shift(1).rolling(5).mean()
    df["ma10"] = df["Close"].shift(1).rolling(10).mean()
    df["volatility"] = df["Close"].shift(1).rolling(20).std()
    df["momentum"] = df["Close"].shift(1) / df["Close"].shift(10) - 1
    df["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek
    return df.dropna().reset_index(drop=True)

# Model tuning, evaluation & prediction
@st.cache_resource
def tune_model_with_cv(df_full: pd.DataFrame, n_splits: int = 10):
    df_feat = add_features(df_full, selected_name, ticker)
    if df_feat.empty:
        st.warning("No features generated for model training. Check data availability.")
        return None, None, None
    X = df_feat[["lag1","lag2","ma5","ma10","volatility","momentum","dow","daily_score"]].values
    y = df_feat["Close"].values
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=30, max_train_size=252*3)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 10, 30),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100)
        }
        r2_scores = []
        for train_idx, val_idx in tscv.split(X):
            train_idx = train_idx[train_idx < val_idx.min() - 5]
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            model = lgb.LGBMRegressor(**params, objective="regression", random_state=42)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(30, first_metric_only=True)]
            )
            preds = model.predict(Xval)
            r2_scores.append(r2_score(yval, preds))
        return np.mean(r2_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params
    scaler = StandardScaler().fit(X)
    Xs_full = scaler.transform(X)
    final_model = lgb.LGBMRegressor(**best_params, objective="regression", random_state=42)
    final_model.fit(Xs_full, y)
    return final_model, scaler, study.best_trial

@st.cache_resource
def tune_xgb_model_with_cv(df_full: pd.DataFrame, n_splits: int = 10):
    df_feat = add_features(df_full, selected_name, ticker)
    if df_feat.empty:
        st.warning("No features generated for model training. Check data availability.")
        return None, None, None
    X = df_feat[["lag1","lag2","ma5","ma10","volatility","momentum","dow","daily_score"]].values
    y = df_feat["Close"].values
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=30, max_train_size=252*3)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "gamma": trial.suggest_float("gamma", 0.1, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10),
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1
        }
        r2_scores = []
        for train_idx, val_idx in tscv.split(X):
            train_idx = train_idx[train_idx < val_idx.min() - 5]
            Xtr, Xval = X[train_idx], X[val_idx]
            ytr, yval = y[train_idx], y[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(
                Xtr, ytr,
                eval_set=[(Xval, yval)],
                verbose=False
            )
            preds = model.predict(Xval)
            r2_scores.append(r2_score(yval, preds))
        return np.mean(r2_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params
    scaler = StandardScaler().fit(X)
    Xs_full = scaler.transform(X)
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(Xs_full, y)
    return final_model, scaler, study.best_trial

def evaluate_model_with_test(df_full: pd.DataFrame, model, scaler, test_size: float = 0.2):
    if model is None or scaler is None:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}, pd.DataFrame()
    df_feat = add_features(df_full, selected_name, ticker)
    if df_feat.empty:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}, pd.DataFrame()
    X = df_feat[["lag1","lag2","ma5","ma10","volatility","momentum","dow","daily_score"]].values
    y = df_feat["Close"].values
    n = len(y)
    split = int((1 - test_size) * n)
    X_trainval, X_test = X[:split], X[split:]
    y_trainval, y_test = y[:split], y[split:]
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    dates = pd.to_datetime(df_feat["Date"].iloc[split:])
    backtest = pd.DataFrame({
        "Date": dates,
        "Actual": y_test,
        "Predicted": y_pred
    })
    backtest["Upper"] = backtest["Predicted"] + 2 * (y_test - y_pred).std()
    backtest["Lower"] = backtest["Predicted"] - 2 * (y_test - y_pred).std()
    return {"rmse": rmse, "mae": mae, "r2": r2}, backtest

def make_predictions(df_full: pd.DataFrame, model, scaler):
    if model is None or scaler is None:
        return pd.DataFrame()
    df_feat = add_features(df_full, selected_name, ticker)
    if df_feat.empty:
        return pd.DataFrame()
    last_close = df_feat["Close"].iloc[-1]
    history = df_feat["Close"].tolist()[-10:]
    today = pd.Timestamp(datetime.now().date())
    curr = last_close
    dates, preds = [], []
    for _ in range(7):
        nxt = today + timedelta(days=1)
        while nxt.weekday() > 4:
            nxt += timedelta(days=1)
        feat = [
            curr,
            history[-2] if len(history) > 1 else curr,
            np.mean(history[-5:]),
            np.mean(history[-10:]),
            df_feat["volatility"].iloc[-1],
            df_feat["momentum"].iloc[-1],
            nxt.weekday(),
            df_feat["daily_score"].iloc[-1]
        ]
        p = model.predict(scaler.transform([feat]))[0]
        dates.append(nxt)
        preds.append(p)
        history.append(p)
        curr = p
        today = nxt
    out = pd.DataFrame({"Date": dates, "Predicted_Close": preds})
    out["Predicted_Return"] = out["Predicted_Close"].pct_change().fillna((preds[0] - last_close) / last_close)
    return out

# Streamlit main()
def main():
    st.title(f"{selected_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    df_disp, info = fetch_stock_data(ticker, period)
    df_full = fetch_full_history(ticker)

    # Check if df_disp is empty before accessing data
    if df_disp.empty:
        st.error("Unable to display stock metrics due to missing data.")
        metrics = [
            ("Current Price", "N/A"),
            ("Market Cap", "N/A"),
            ("52W High", "N/A"),
            ("52W Low", "N/A")
        ]
    else:
        metrics = [
            ("Current Price", f"₹{df_disp['Close'].iloc[-1]:,.2f}"),
            ("Market Cap", f"₹{info.get('marketCap',0)/1e7:,.1f} Cr"),
            ("52W High", f"₹{info.get('fiftyTwoWeekHigh',0):,.2f}"),
            ("52W Low", f"₹{info.get('fiftyTwoWeekLow',0):,.2f}")
        ]

    cols = st.columns(4)
    for c, (lab, val) in zip(cols, metrics):
        c.markdown(f"<div class='metric-card'><h3>{lab}</h3><p style='font-size:1.5rem;margin:0.5rem 0;'>{val}</p></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Only display candlestick chart if data is available
    if not df_disp.empty:
        fig = go.Figure([go.Candlestick(
            x=df_disp["Date"], open=df_disp["Open"], high=df_disp["High"],
            low=df_disp["Low"], close=df_disp["Close"], name="Price"
        )])
        if candlestick_ma:
            for w, col in [(20, "#FFA726"), (50, "#26C6DA")]:
                fig.add_trace(go.Scatter(
                    x=df_disp["Date"], y=df_disp["Close"].rolling(w).mean(),
                    line=dict(color=col, width=2), name=f"{w} MA"
                ))
        if show_boll:
            sma = df_disp["Close"].rolling(20).mean()
            std = df_disp["Close"].rolling(20).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            fig.add_trace(go.Scatter(x=df_disp["Date"], y=sma, line=dict(color="#FF6F00"), name="BB Middle"))
            fig.add_trace(go.Scatter(x=df_disp["Date"], y=upper, line=dict(color="#4CAF50"), name="BB Upper"))
            fig.add_trace(go.Scatter(x=df_disp["Date"], y=lower, line=dict(color="#F44336"), fill="tonexty", fillcolor="rgba(76,175,80,0.1)", name="BB Lower"))
        fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", showlegend=True, xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    sent = compute_daily_sentiment(selected_name, ticker)
    if not sent.empty:
        st.subheader("Combined Daily Sentiment (FinBERT + Groq)")
        fig_s = go.Figure([go.Bar(x=sent["Date"], y=sent["daily_score"], name="Sentiment Score")])
        fig_s.update_layout(template="plotly_dark", height=200, margin=dict(l=20,r=20,t=20,b=20), yaxis_title="Score")
        st.plotly_chart(fig_s, use_container_width=True)
    
    if show_rsi and not df_disp.empty:
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        rsi = calc_rsi(df_disp["Close"])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Relative Strength Index (RSI)")
        fig_r = go.Figure([go.Scatter(x=df_disp["Date"], y=rsi, line=dict(width=2), name="RSI")])
        fig_r.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_r.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_r.update_layout(template="plotly_dark", height=400, showlegend=False, margin=dict(l=20,r=20,t=40,b=20), yaxis_title="RSI")
        st.plotly_chart(fig_r, use_container_width=True)
    
    if not df_disp.empty:
        st.subheader("Trading Volume")
        fig_v = go.Figure([go.Bar(x=df_disp["Date"], y=df_disp["Volume"], marker=dict(color="rgba(255,99,132,0.6)"))])
        fig_v.update_layout(template="plotly_dark", height=400, showlegend=False, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_v, use_container_width=True)
    
    if show_preds and not df_full.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Model Training & Backtest")
        lgb_model, lgb_scaler, lgb_best_trial = tune_model_with_cv(df_full, n_splits=10)
        xgb_model, xgb_scaler, xgb_best_trial = tune_xgb_model_with_cv(df_full, n_splits=10)
        lgb_metrics, lgb_backtest = evaluate_model_with_test(df_full, lgb_model, lgb_scaler, test_size=0.2)
        xgb_metrics, xgb_backtest = evaluate_model_with_test(df_full, xgb_model, xgb_scaler, test_size=0.2)
        
        if lgb_best_trial is not None and xgb_best_trial is not None:
            st.markdown("**Model Performance Metrics**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**LightGBM**")
                st.markdown(f"Best CV R²: {lgb_best_trial.value:.3f}")
                st.markdown(f"Test R²: {lgb_metrics['r2']:.3f}")
                st.markdown(f"RMSE: {lgb_metrics['rmse']:.2f}")
                st.markdown(f"MAE: {lgb_metrics['mae']:.2f}")
            with col2:
                st.markdown("**XGBoost**")
                st.markdown(f"Best CV R²: {xgb_best_trial.value:.3f}")
                st.markdown(f"Test R²: {xgb_metrics['r2']:.3f}")
                st.markdown(f"RMSE: {xgb_metrics['rmse']:.2f}")
                st.markdown(f"MAE: {xgb_metrics['mae']:.2f}")
            
            st.subheader("Backtest: Actual vs Predicted")
            st.markdown("**LightGBM**")
            if not lgb_backtest.empty:
                fig_lgb = go.Figure()
                fig_lgb.add_trace(go.Scatter(x=lgb_backtest["Date"], y=lgb_backtest["Actual"], mode="lines", name="Actual"))
                fig_lgb.add_trace(go.Scatter(x=lgb_backtest["Date"], y=lgb_backtest["Predicted"], mode="lines", name="Predicted"))
                fig_lgb.add_trace(go.Scatter(x=lgb_backtest["Date"], y=lgb_backtest["Upper"], mode="lines", line=dict(color="lightgrey"), name="Upper"))
                fig_lgb.add_trace(go.Scatter(x=lgb_backtest["Date"], y=lgb_backtest["Lower"], mode="lines", line=dict(color="lightgrey"), fill="tonexty", name="Lower"))
                fig_lgb.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_lgb, use_container_width=True)
            
            st.markdown("**XGBoost**")
            if not xgb_backtest.empty:
                fig_xgb = go.Figure()
                fig_xgb.add_trace(go.Scatter(x=xgb_backtest["Date"], y=xgb_backtest["Actual"], mode="lines", name="Actual"))
                fig_xgb.add_trace(go.Scatter(x=xgb_backtest["Date"], y=xgb_backtest["Predicted"], mode="lines", name="Predicted"))
                fig_xgb.add_trace(go.Scatter(x=xgb_backtest["Date"], y=xgb_backtest["Upper"], mode="lines", line=dict(color="lightgrey"), name="Upper"))
                fig_xgb.add_trace(go.Scatter(x=xgb_backtest["Date"], y=xgb_backtest["Lower"], mode="lines", line=dict(color="lightgrey"), fill="tonexty", name="Lower"))
                fig_xgb.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_xgb, use_container_width=True)
            
            st.subheader("7-Day Price Predictions")
            lgb_preds = make_predictions(df_full, lgb_model, lgb_scaler)
            xgb_preds = make_predictions(df_full, xgb_model, xgb_scaler)
            if not lgb_preds.empty and not xgb_preds.empty:
                preds = lgb_preds.merge(xgb_preds, on="Date", suffixes=("_LGB", "_XGB"))
                preds["Predicted_Close_Avg"] = (preds["Predicted_Close_LGB"] + preds["Predicted_Close_XGB"]) / 2
                preds["Predicted_Return_Avg"] = preds["Predicted_Close_Avg"].pct_change().fillna((preds["Predicted_Close_Avg"].iloc[0] - df_full["Close"].iloc[-1]) / df_full["Close"].iloc[-1])
                st.table(pd.DataFrame({
                    "Date": preds["Date"].dt.strftime("%Y-%m-%d"),
                    "LightGBM": preds["Predicted_Close_LGB"].map("₹{:,.2f}".format),
                    "XGBoost": preds["Predicted_Close_XGB"].map("₹{:,.2f}".format),
                    "Average": preds["Predicted_Close_Avg"].map("₹{:,.2f}".format),
                    "Return (Avg)": preds["Predicted_Return_Avg"].apply(lambda x: f"▲ {x:.2%}" if x > 0 else (f"▼ {abs(x):.2%}" if x < 0 else "0.00%"))
                }))
                base = df_full["Close"].iloc[-1]
                target = preds["Predicted_Close_Avg"].iloc[-1]
                change = (target - base) / base * 100
                cls = "prediction-up" if change > 2 else ("prediction-down" if change < -2 else "prediction-neutral")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"<div class='prediction-card'><h3>Predicted Trend</h3><p class='{cls}' style='font-size:1.5rem;margin:0.5rem 0;'>{cls.split('-')[-1].title()}</p></div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<div class='prediction-card'><h3>Expected Change</h3><p class='{cls}' style='font-size:1.5rem;margin:0.5rem 0;'>{change:+.2f}%</p></div>", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"<div class='prediction-card'><h3>7-Day Target Price</h3><p class='{cls}' style='font-size:1.5rem;margin:0.5rem 0;'>₹{target:,.2f}</p></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Latest News & Sentiment")
    news_df = fetch_and_enrich_news(selected_name, ticker)
    for _, row in news_df.head(5).iterrows():
        col_label = sentiment_color(row.groq_impact > 0 and "positive" or "negative")
        st.markdown(f"""
          <div class="news-card">
            <h4><a href="{row.link}" target="_blank">{row.title}</a></h4>
            <p>{row.description}</p>
            <span style="background:{col_label};padding:4px;border-radius:4px;color:#fff;">
              FinBERT: {row.finbert_score:+.2f} | Impact: {row.groq_impact:+.2f}
            </span>
          </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
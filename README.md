# StockZen

**StockZen** is a modern Indian stock dashboard designed to empower investors with live market data, advanced charting, and cutting-edge machine learning-based predictions for Indian stocks using Yahoo Finance data. Built with Streamlit, LightGBM, and Optuna, StockZen helps you make informed investment decisions by analyzing both recent and full historical data.

StockZen combines modern technology with local market expertise to bring you a seamless, user-friendly investing experience.

**Web Link** stockzen.streamlit.app

## Features

- **Real-Time Data Fetching:**  
  Get up-to-date stock information directly from Yahoo Finance.

- **Interactive Charting:**  
  Visualize market trends with interactive candlestick charts enhanced with moving averages, Bollinger Bands, RSI, and volume indicators. Hover over the chart to view the exact dates on the x-axis.

- **Machine Learning Predictions:**  
  Generate 7-day price forecasts using a LightGBM model tuned with Optuna on full historical data.
  - **Daily Returns:**  
    - The first predicted day’s return is calculated relative to the current (last historical) price.
    - Each subsequent day’s return is computed as the percentage change from the previous predicted day.
  - **Overall Expected Change:**  
    The expected change is calculated as the percentage difference between the 7th day predicted price and the current price.

- **News Integration:**  
  Stay updated with the latest news articles relevant to your selected stock.

- **Model Caching & Refresh:**  
  Tuned models are cached per stock for fast predictions. Use the "Refresh Model Cache" button to retrain the model if needed.

```markdown
# StockZen

**StockZen** is a modern Indian stock dashboard designed to empower investors with live market data, advanced charting, and cutting-edge machine learning-based predictions for Indian stocks using Yahoo Finance data. Built with Streamlit, LightGBM, and Optuna, StockZen helps you make informed investment decisions by analyzing both recent and full historical data.

StockZen combines modern technology with local market expertise to bring you a seamless, user-friendly investing experience.

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

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stockzen.git
   cd stockzen
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8+ installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

   **Key dependencies include:**
   - Streamlit
   - yfinance
   - Plotly
   - Pandas
   - NumPy
   - LightGBM
   - Optuna
   - scikit-learn
   - Requests
   - python-dotenv

3. **Set Up Environment Variables:**

   Create a `.env` file in the project root with your News API key:

   ```
   NEWS_API_KEY=your_news_api_key_here
   ```

## Usage

1. **Run the App:**

   ```bash
   streamlit run app.py
   ```

2. **Dashboard Overview:**

   - **Select a Stock:**  
     Use the sidebar to choose from a curated list of major Indian stocks (e.g., Nifty 100 constituents).
     
   - **Choose a Display Period:**  
     Set the period for charts and key metrics using the sidebar.
     
   - **Model Tuning & Predictions:**  
     StockZen fetches all available historical data for model training. It then displays 7-day predictions with:
       - Daily returns computed as the percentage change from the previous predicted day.
       - An overall expected change, calculated as the percentage difference between the 7th day predicted price and the current price.
       
   - **Refresh Model Cache:**  
     Use the "Refresh Model Cache" button in the sidebar to retrain the model for the selected stock if needed.
     
   - **Latest News:**  
     Stay informed with the latest news articles related to the selected stock.

## Customization

- **Model Tuning:**  
  StockZen uses Optuna to automatically tune LightGBM hyperparameters. To perform more exhaustive tuning, adjust the number of trials in the `tune_and_train_model()` function.

- **UI Customization:**  
  The interface is built using Streamlit. You can modify the embedded CSS in the markdown block to change the look and feel of the dashboard.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or feature requests.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Embrace smart investing with StockZen – where modern technology meets financial wisdom!*
```
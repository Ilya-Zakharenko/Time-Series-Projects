# <center> **PROJECT: Telegram Stock Forecast Bot**

Intelligent Telegram bot for stock price forecasting and trading recommendations using Time Series analysis.

---

### **Project Goal**

Develop a fully functional Telegram bot that allows users to get AI-powered stock price forecasts for the next 30 days, along with buy/sell recommendations and potential profit calculations based on their investment amount.

---

### **Key Features**

- Real-time stock data loading via `yfinance`
- Training multiple forecasting models:
  - Classical ML (`RandomForestRegressor` with lag features)
  - Statistical model (`Prophet`)
  - Neural Network (`LSTM` / GRU)
- Automatic selection of the best model based on RMSE / MAPE
- 30-day price forecast with visualization
- Trading recommendations (local minima/maxima)
- Estimated profit calculation
- Full logging system (`logs.txt`)
- User-friendly Telegram interface (`aiogram`)

---

### **Technologies Used**

- **Telegram Bot**: `aiogram` (async)
- **Data**: `yfinance`, `pandas`
- **Forecasting**:
  - `Prophet`
  - `RandomForestRegressor` (scikit-learn)
  - `LSTM` / GRU (PyTorch)
- **Visualization**: `plotly`
- **Utilities**: `joblib`, logging, FSM (Finite State Machine)

**Main Files**:
- `bot.py` — main Telegram bot logic
- `models.py` — training and model selection
- `forecast.py` — forecasting and recommendations
- `utils.py` — plotting and logging

---

### **Project Stages**

1. User interaction via Telegram bot
2. Automatic downloading of historical stock data (2 years)
3. Feature engineering for time series
4. Training and comparison of multiple models
5. Automatic selection of the best model
6. 30-day forecast generation
7. Trading recommendations and profit estimation
8. Logging all user requests

---

### **Conclusion**

This project combines Time Series Forecasting, Machine Learning, and Telegram Bot Development into a complete production-like service. The bot can analyze any popular stock ticker, select the best forecasting model automatically, and provide actionable trading recommendations with visualizations.
A great demonstration of practical application of time series analysis in a user-facing product.

---

### **How to run**

```bash
cd Telegram.Stock-Forecast-Bot

# Install dependencies
pip install -r requirements.txt

# Run the bot
python bot.py

---
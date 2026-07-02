# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from prophet import Prophet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')







# --- Создаём лаговые признаки для временного ряда ---
def create_features(data, lags=30):
    # data — это Series с датами в индексе
    df = pd.DataFrame(index=data.index)          # сохраняем индекс-дат
    df['y'] = data.values                        # значения отдельно
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df.dropna()                           # удаляем строки с пропусками










# --- Создаём и обучаем модель 'RandomForestRegressor()' ---
def train_rf(train):
    
    # Создаём признаки
    df = create_features(train)
    
    # Разделяем на признаки и целевую переменную
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Разделяем на обучающую и тестовую выборки
    split = int(0.8 * len(X))                          # берём 80% от полного размера DataFrame
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Создаём модель
    RFR_model = RandomForestRegressor(
        n_estimators = 100,
        random_state = 42
    )
    
    # Обучаем модель
    RFR_model.fit(X_train, y_train)
    
    # Делаем предсказания и вычисляем метрики
    y_predict = RFR_model.predict(X_test)
    MSE_metric = mean_squared_error(y_test, y_predict)
    RMSE_metric = np.sqrt(MSE_metric)
    MAPE_metric = mean_absolute_percentage_error(y_test, y_predict)
    
    # Возвращаем модель и метрики
    return RFR_model, RMSE_metric, MAPE_metric










# --- Создаём и обучаем модель 'Prophet()' ---
def train_prophet(train):
    df = pd.DataFrame({
        'ds': train.index,
        'y': train.values.ravel()                
    })
    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(train_df)
    future = m.make_future_dataframe(periods=len(test_df))
    forecast = m.predict(future)
    pred = forecast.iloc[split_idx:]['yhat'].values
    rmse = np.sqrt(mean_squared_error(test_df['y'], pred))
    mape = mean_absolute_percentage_error(test_df['y'], pred)
    return m, rmse, mape










# --- Создаём и обучаем модель 'TimeSeriesDataset()' (нейросетевая модель) ---

# Подготовка данных 
class TimeSeriesDataset(Dataset):
    
    # Инициализация класса
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len   # создаём последовательности по 30 дней
    
    # Задаём количество примеров   
    def __len__(self):
        return len(self.data) - self.seq_len
    
    
    # Задаём получение одного примера
    def __getitem__(self, idx):
        return (
            
            # Последовательность из 30 дней
            torch.tensor(
                self.data[idx: idx+self.seq_len], # вход
                dtype = torch.float32
            ),
            
            # Целевое значение - следующий день
            torch.tensor(
                self.data[idx+self.seq_len],      # выход
                dtype = torch.float32
            )
            
        )
        
        
        
# Создаём архитектуру нейросети
#class LSTMModel(nn.Module):
#    
#    # Инициализация класса
#    def __init__(self):
#        super().__init__()                             # инициализируем родительский класс
#        self.lstm = nn.LSTM(1, 50, batch_first=True)   # задаём 1 вход, 50 нейронов
#        self.fc = nn.Linear(50, 1)                     # задаём линейный слой
#        
#    # Берём последнее скрытое состояние
#    def forward(self, x):
#        _, (h_n, _) = self.lstm(x)    # получаем скрытые состояния
#        return self.fc(h_n[-1])       # передаём последнее скрытое состояние в линейный слой
    
    
    
## Обучаем модель
#def train_lstm(train, seq_len=30, epochs=30):
#    
#    # Масштабируем данные
#    MM_scaler = MinMaxScaler()
#    scaled_data = MM_scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
#    
#    # Создаём датасет и загрузчики данных
#    dataset = TimeSeriesDataset(scaled_data, seq_len)
#    split = int(0.8 * len(dataset))
#    train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
#    test_loader = DataLoader(dataset[split:], batch_size=1)
#    
#    # Инициализируем модель, оптимизатор и функцию потерь
#    LSTM_model = LSTMModel()
#    optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=0.001)
#    criterion = nn.MSELoss()
    
#    # Обучаем модель
#    LSTM_model.train()
#    for _ in range(epochs):
#        for x, y in train_loader:
#            x = x.unsqueeze(-1)                     # добавляем размерность
#            optimizer.zero_grad()                   # обнуляем градиенты
#            output = LSTM_model(x)                  # прямой проход
#            loss = criterion(output.squeeze(), y)   # вычисляем потерю
#            loss.backward()                         # обратный проход 
#            optimizer.step()                        # обновляем веса
    
#    # Оцениваем модель    
#    LSTM_model.eval()
#    predictions_lst, actuals_lst = [], []
#    with torch.no_grad():
#        for x, y in test_loader:
#            x = x.unsqueeze(-1)
#            prediction = LSTM_model(x).item()
#            predictions_lst.append(prediction)
#            actuals_lst.append(y.item())

#    # Обратное масштабирование
#    predictions = MM_scaler.inverse_transform(np.array(predictions_lst).reshape(-1, 1)).flatten()
#    actuals = MM_scaler.inverse_transform(np.array(actuals_lst).reshape(-1, 1)).flatten()

#    # Вычисляем метрики
#    RMSE_metric = mean_squared_error(actuals, predictions, squared=False)
#    MAPE_metric = mean_absolute_percentage_error(actuals, predictions)

#    # Возвращаем модель, скейлер и метрики
#    return LSTM_model, MM_scaler, seq_len, RMSE_metric, MAPE_metric
            

    
    






# --- Создаём функцию для обучения всех моделей ---
def train_models(close_prices):
    print(f"Количество точек данных для обучения: {len(close_prices)}")  # ← отладка!

    metrics = {}
    models = {}

    # Random Forest
    try:
        rf_model, rf_rmse, rf_mape = train_rf(close_prices)
        metrics['RandomForest'] = {'RMSE': rf_rmse, 'MAPE': rf_mape}
        models['RandomForest'] = ('rf', rf_model)
        print("RandomForest успешно обучена")
    except Exception as e:
        print(f"Ошибка в RandomForest: {str(e)}")

    # Prophet
    try:
        prophet_model, p_rmse, p_mape = train_prophet(close_prices)
        metrics['Prophet'] = {'RMSE': p_rmse, 'MAPE': p_mape}
        models['Prophet'] = ('prophet', prophet_model)
        print("Prophet успешно обучена")
    except Exception as e:
        print(f"Ошибка в Prophet: {str(e)}")

#    # LSTM
#    try:
#        lstm_model, scaler, seq_len, l_rmse, l_mape = train_lstm(close_prices)
#        metrics['LSTM'] = {'RMSE': l_rmse, 'MAPE': l_mape}
#        models['LSTM'] = ('lstm', lstm_model, scaler, seq_len)
#        print("LSTM успешно обучена")
#    except Exception as e:
#        print(f"Ошибка в LSTM: {str(e)}")

    if not metrics:
        raise ValueError("Ни одна модель не обучилась! См. ошибки выше.")
    
    best_model_name = min(metrics, key=lambda x: metrics[x]['RMSE'])
    best_model = models[best_model_name]
    return best_model_name, best_model, metrics
     
     

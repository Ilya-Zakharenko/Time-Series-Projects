# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
import torch





# Прогноз RandomForest 
def forecast_rf(model, last_values, days=30, lags=30):
    preds = []
    # Берём ТОЛЬКО числовые цены как numpy array (гарантированно float)
    prices = last_values.values[-lags:]  # numpy array shape (lags,)
    current = prices.copy()              # копия, чтобы не менять оригинал

    for _ in range(days):
        # Явно делаем 2D-массив: 1 строка, lags столбцов
        X = current.reshape(1, -1)       # shape (1, lags) — идеально для RF
        pred = model.predict(X)[0]
        preds.append(pred)
        # Сдвигаем окно: убираем первый, добавляем новый предикт
        current = np.roll(current, -1)   # сдвиг влево
        current[-1] = pred               # добавляем новый в конец

    return preds







# Прогноз Prophet
def forecast_prophet(model, last_date, days=30):
    
    '''
    Функция для прогноза с использованием Prophet модели.
    
    Параметры:
      model: Обученная модель Prophet
      last_date: Последняя дата в данных
      days: Количество дней для прогноза (по умолчанию 30)
      
    Возвращает:
      Список предсказанных значений на заданное количество дней
    '''
    
    # Создаём датафрейм с будущими датами
    future = model.make_future_dataframe(periods = days)
    
    # Прогнозируем значения
    forecast = model.predict(future)
    
    # Возвращаем последние предсказанные значения
    return forecast[-days:]['yhat'].values







# Прогноз LSTM
def forecast_lstm(model_scaler_seq, last_values, days=30):
    
    '''
    Функция для прогноза с использованием LSTM модели.
    
    Параметры:
      model_scaler_seq: Кортеж из модели, скейлера и длины последовательности
      last_values: Последние значения для инициализации прогноза
      days: Количество дней для прогноза (по умолчанию 30)
      
    Возвращает:
      Список предсказанных значений на заданное количество дней
    '''
    
    # Извлекаем модель, скейлер и длину последовательности
    model, scaler, seq_len = model_scaler_seq
    scaled = scaler.transform(last_values.values.reshape(-1, 1)).flatten()
    current = scaled[-seq_len:]
    predictions_lst = []
    
    # Прогнозируем значения на заданное количество дней
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            x = torch.tensor(current[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred_scaled = model(x).item()
            predictions_lst.append(pred_scaled)
            current = np.append(current, pred_scaled)
            
    # Обратное преобразование предсказаний
    predictions_lst = scaler.inverse_transform(np.array(predictions_lst).reshape(-1, 1)).flatten()
    
    # Возвращаем список предсказаний
    return predictions_lst







# Создаём основную функцию для прогноза
def make_forecast(best_model, model_name, close_prices, forecast_days=30):
    
    '''
    Функция для создания прогноза на основе выбранной модели.
    
    Параметры:
    
      best_model: Обученная модель для прогнозирования
      model_name: Название модели ('RandomForest', 'Prophet', 'LSTM')
      close_prices: Серия с ценами закрытия
      forecast_days: Количество дней для прогноза
      
    Возвращает:
      Словарь с прогнозом и процентным изменением
    '''
    
    # Проверяем, что цены закрытия имеют индекс datetime
    if not isinstance(close_prices.index, pd.DatetimeIndex):
        raise ValueError('Индекс close_prices должен быть типа pd.DatetimeIndex')
    
    # Получаем последнюю дату в данных
    last_date = close_prices.index[-1]
    
    # Создаём будущие даты для прогноза
    future_dates = pd.date_range(
        start = last_date + timedelta(days = 1),
        periods = forecast_days
    )
    
    
    # Вызываем соответствующую функцию прогноза в зависимости от модели
    if model_name == 'RandomForest':                                        # если модель RandomForest
        _, model = best_model                  
        predictions = forecast_rf(model, close_prices, forecast_days)
    elif model_name == 'Prophet':                                           # если модель Prophet
        _, model = best_model
        predictions = forecast_prophet(model, last_date, forecast_days)
    elif model_name == 'LSTM':                                              # если модель LSTM
        _, model, scaler, seq_len = best_model
        predictions = forecast_lstm((model, scaler, seq_len), close_prices, forecast_days)
    else:                                                                   # в случае неподдерживаемого имени модели вызываем ошибку 
        raise ValueError('Неподдерживаемое имя модели')
    
    # Создаём серию с прогнозными значениями
    forecast_series = pd.Series(
        predictions,
        index = future_dates,
        name = 'Close'
    )

    # Получаем текущую цену и финальную цену прогноза
    current_price = close_prices.iloc[-1]
    final_price = predictions[-1]
    
    # Рассчитываем процентное изменение
    change_pct = (final_price - current_price) / current_price * 100

    # Возвращаем словарь с прогнозом и процентным изменением
    return {
        'forecast': forecast_series,
        'change_pct': change_pct
    }
    
    
    
    
    
    
    
# Создаём функцию для получения рекомендаций по торговле
def get_recommendations(forecast_series, amount):
    
    '''
    Функция для получения торговых рекомендаций на основе прогноза.
    
    Параметры:
      forecast_series: Серия с прогнозными значениями
      amount: Сумма денег для инвестирования
      
    Возвращает:
      Краткое описание стратегии и ожидаемая прибыль
    '''
    
    # Извлекаем цены и даты из серии прогноза
    prices = forecast_series.values
    dates = forecast_series.index.strftime('%Y-%m-%d')
    
    # Инициализируем списки для покупок и продаж
    buys_lst = []
    sells_lst = []
    
    # Ищем локальные минимумы и максимумы
    for i in range(1, len(prices) - 1):
        if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            buys_lst.append((dates[i], prices[i]))
        elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            sells_lst.append((dates[i], prices[i]))
            
    # Формируем рекомендации на основе найденных минимумов и максимумов
    profit = 0
    summary = 'Стратегия: \n'
    if buys_lst and sells_lst:
        buy_date, buy_price = buys_lst[0]
        sells_date, sells_price = max(sells_lst, key=lambda x: x[1])  # самый высокий пик
        shares = amount // buy_price
        profit = shares * (sells_price - buy_price)
        summary += f'Купить {shares} акций {buy_date} по ${buy_price:.2f}\n'
        summary += f'Продать {sells_date} по ${sells_price:.2f}\n'
    else:
        summary += 'Рекомендаций нет — тренд неопределённый.\n'
        
    # Возвращаем краткое описание стратегии и ожидаемую прибыль
    return summary, profit
    
    
    
    
    
    
    
    
   
# Импортируем необходимые библиотеки
import plotly.graph_objects as go
from datetime import datetime
import os







# Функция для построения графика прогноза
def plot_forecast(df, ticker):
    
    '''
    Построение графика прогноза цен на основе исторических данных и прогноза.
    
    Параметры:
        df: Датафрейм с историческими данными и прогнозом.
        ticker: Тикер актива.
    
    Возвращает:
        fig: Объект графика Plotly.
    '''
    
    # Создаем фигуру графика
    fig = go.Figure()
    
    # Добавляем исторические данные
    fig.add_trace(
        go.Scatter(
            x = df.index,
            y = df['История'],
            name = 'История',
            line = dict(color = 'blue')
        )
    )
    
    # Добавляем прогнозные данные
    fig.add_trace(
        go.Scatter(
            x = df.index,
            y = df['Прогноз'],
            name = 'Прогноз',
            line = dict(color='red', dash='dot')
        )
    )
    
    # Настраиваем оформление графика
    fig.update_layout(
        title = f'Прогноз цен {ticker}',
        xaxis_title = 'Дата',
        yaxis_title = 'Цена, USD',
        template = 'plotly_white',
        width = 1200,    
        height = 800,  
        margin = dict(l=50, r=50, t=80, b=50)
    )
    
    # Возвращаем объект графика
    return fig







# Функция для логирования запросов пользователей
def log_request(user_id, ticker, amount, model, metric, profit):
    
    '''
    Логирование запроса пользователя в файл logs.txt.
    
    Параметры:
        user_id: Идентификатор пользователя.
        ticker: Тикер актива.
        amount: Сумма инвестиций.
        model: Используемая модель прогноза.
        metric: Метрика качества модели.
        profit: Полученная прибыль.
    '''
    
    # Форматируем строку для логирования
    log_line = f'{datetime.now().isoformat()}\t{user_id}\t{ticker}\t{amount}\t{model}\t{metric:.2f}\t{profit:.2f}\n'
    
    # Записываем строку в файл logs.txt
    with open('logs.txt', 'a', encoding='utf-8') as f:
        f.write(log_line)
    
    











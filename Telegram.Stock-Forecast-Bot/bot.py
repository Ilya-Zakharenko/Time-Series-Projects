# Импортируем необходимые библиотеки
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import yfinance as yf
from datetime import datetime
import pandas as pd
import os

from models import train_models
from forecast import make_forecast, get_recommendations
from utils import plot_forecast, log_request





# Инициализация бота
BOT_TOKEN = '8249417762:AAGjYaBgomuckv0ecW8_XPOC9dT8MA_y7iY'

bot = Bot(token = BOT_TOKEN)         # инициализация бота и диспетчера
storage = MemoryStorage()            # ипользуем память для хранения состояний
dp = Dispatcher(storage = storage)   # инициализация диспетчера с хранилищем состояний




# Создаём 2 состояния диалога
class Form(StatesGroup):
    waiting_for_ticker = State()   # ждём тикер
    waiting_for_amount = State()   # ждём сумму




# Создаём обработчик команды /start
@dp.message(Command('start'))
async def start(message: types.Message, state: FSMContext):                  # берём текст сообщения
    await message.answer('Привет! Введи тикер компании (например, AAPL):')   # просим ввести 'тикер'
    await state.set_state(Form.waiting_for_ticker)                           # переводим в состояние 'ждём тикер'




# Обработка ввода тикера
@dp.message(Form.waiting_for_ticker)
async def get_ticker(message: types.Message, state: FSMContext):                       # берём текст сообщения
    ticker = message.text.strip().upper()
    await state.update_data(ticker = ticker)                                           # сохраняем текст в состояние
    await message.answer(f'Тикер: {ticker}. Теперь введи сумму инвестиций (в USD):')   # просим ввести 'сумму'
    await state.set_state(Form.waiting_for_amount)                                     # переводим в состояние 'ждём сумму'






# Получаем сумму и запускаем анализ
@dp.message(Form.waiting_for_amount)
async def get_amount(message: types.Message, state: FSMContext):
    
    # Проверка корректности суммы
    try:
        amount = float(message.text.replace(',', '.'))                    # заменяем запятую на точку для корректного преобразования
        if amount <= 0:                                                   # сумма должна быть положительной
            raise ValueError
    except ValueError:                                                    # если не удалось преобразовать в число
        await message.answer('Некорректная сумма. Попробуйте ещё раз:')   # просим ввести 'сумму' ещё раз
        return
    
    # Получаем данные из состояния
    user_data = await state.get_data()
    ticker = user_data['ticker']
    user_id = message.from_user.id
    await message.answer('Начинаю анализ... Это может занять несколько минут. Пожалуйста, подождите.')

    # Загрузка данных
    data = yf.download(ticker, period='2y', interval='1d')
    if data.empty:
        await message.answer('Не удалось загрузить данные. Проверь тикер.')
        return
    
    
    
    # Получение цен закрытия
    close_prices = data['Close'].dropna()
    
    # Проверка достаточности данных
    if len(close_prices) < 100:
        await message.answer(
            f"Ошибка: для тикера {ticker} доступно только {len(close_prices)} дней данных. "
            "Нужно минимум 100 дней для обучения моделей. Попробуй другой тикер (например, AAPL, MSFT)."
        )
        await state.clear()
        return



    # Обучение моделей
    best_model_name, best_model, metrics = train_models(close_prices)

    # Получаем прогноз на 30 дней
    forecast_result = make_forecast(best_model, best_model_name, close_prices, forecast_days=30)
    full_df = pd.concat([close_prices, forecast_result['forecast']], axis=1)
    full_df.columns = ['История', 'Прогноз']

    # Построение графика
    fig = plot_forecast(full_df, ticker)
    plot_path = f'plot_{user_id}_{int(datetime.now().timestamp())}.png'
    fig.write_image(plot_path)

    # Получение рекомендаций и расчёт прибыли
    recommendations, profit = get_recommendations(forecast_result['forecast'], amount)

    # Отправка результата
    change_pct_value = forecast_result['change_pct'].item() if isinstance(forecast_result['change_pct'], pd.Series) else forecast_result['change_pct']
    caption = (
        f'📈 *Прогноз для {ticker}*\n\n'
        f"Изменение за 30 дней: {float(change_pct_value):+.2f}%\n"
        f"Лучшая модель: {best_model_name} (RMSE: {metrics[best_model_name]['RMSE']:.2f})\n\n"
        f'💡 *Рекомендации:*\n{recommendations}\n\n'
        f'💰 Потенциальная прибыль: ${profit:,.2f}'
    )
    
    # Отправляем график с подписью
    await message.answer_photo(
        photo = types.FSInputFile(plot_path),  
        caption = caption,
        parse_mode = 'Markdown'
    )

    # Логируем запрос
    log_request(
        user_id = user_id,
        ticker = ticker,
        amount = amount,
        model = best_model_name,
        metric = metrics[best_model_name]['RMSE'],
        profit = profit
    )

    # Удаляем временный файл и очищаем состояние
    os.remove(plot_path)
    await state.clear()






# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())





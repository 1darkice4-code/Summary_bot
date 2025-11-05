FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY telegram_summary_bot_daily_group_summaries_with_llm_python_aiogram.py .

# Создание директории для базы данных (если используется SQLite)
RUN mkdir -p /app/data

# Переменные окружения по умолчанию (можно переопределить через .env)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "telegram_summary_bot_daily_group_summaries_with_llm_python_aiogram.py"]


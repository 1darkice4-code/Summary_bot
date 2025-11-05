# Инструкция по запуску в Docker

## Быстрый старт

1. **Создайте файл `.env`** с необходимыми переменными (см. README.md)

2. **Создайте директорию для данных:**
   ```bash
   mkdir data
   ```

3. **Запустите бота:**
   ```bash
   docker-compose up -d
   ```

4. **Проверьте логи:**
   ```bash
   docker-compose logs -f
   ```

## Полезные команды

- **Остановить бота:**
  ```bash
  docker-compose down
  ```

- **Перезапустить бота:**
  ```bash
  docker-compose restart
  ```

- **Посмотреть статус:**
  ```bash
  docker-compose ps
  ```

- **Пересобрать образ:**
  ```bash
  docker-compose build --no-cache
  docker-compose up -d
  ```

## Структура данных

База данных SQLite будет храниться в директории `./data/summarybot.db` на хосте.

## PostgreSQL (опционально)

Для использования PostgreSQL:
1. Раскомментируйте секцию `postgres` в `docker-compose.yml`
2. Обновите `DATABASE_URL` в `.env`:
   ```
   DATABASE_URL=postgresql+psycopg2://botuser:botpass@postgres:5432/botdb
   ```
3. Перезапустите:
   ```bash
   docker-compose down
   docker-compose up -d
   ```


"""
Скрипт миграции базы данных для обновления структуры таблицы messages.
Используйте этот скрипт, если у вас уже есть база данных со старой структурой.
"""
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///summarybot.db")

# Извлекаем путь к SQLite файлу
if DATABASE_URL.startswith("sqlite:///"):
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if db_path.startswith("/"):
        # Абсолютный путь
        pass
    else:
        # Относительный путь
        db_path = os.path.join(os.getcwd(), db_path)
    
    print(f"Миграция базы данных: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Проверяем, существует ли новая колонка
        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'message_id' not in columns:
            print("Выполняю миграцию...")
            
            # Создаем временную таблицу с новой структурой
            cursor.execute("""
                CREATE TABLE messages_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id BIGINT NOT NULL,
                    message_id BIGINT NOT NULL,
                    user_id BIGINT,
                    username VARCHAR(255),
                    text TEXT,
                    created_at DATETIME,
                    UNIQUE(chat_id, message_id)
                )
            """)
            
            # Создаем индексы
            cursor.execute("CREATE INDEX idx_messages_chat_id ON messages_new(chat_id)")
            cursor.execute("CREATE INDEX idx_messages_message_id ON messages_new(message_id)")
            cursor.execute("CREATE INDEX idx_messages_user_id ON messages_new(user_id)")
            cursor.execute("CREATE INDEX idx_messages_created_at ON messages_new(created_at)")
            
            # Копируем данные (старый ID игнорируем, новый будет автоинкремент)
            # Извлекаем chat_id и message_id из старого ID
            cursor.execute("""
                INSERT INTO messages_new (chat_id, message_id, user_id, username, text, created_at)
                SELECT 
                    chat_id,
                    CASE 
                        WHEN id > 10000000000 THEN (id % 10000000000)
                        ELSE id
                    END as message_id,
                    user_id,
                    username,
                    text,
                    created_at
                FROM messages
            """)
            
            # Удаляем старую таблицу
            cursor.execute("DROP TABLE messages")
            
            # Переименовываем новую таблицу
            cursor.execute("ALTER TABLE messages_new RENAME TO messages")
            
            conn.commit()
            print("Миграция успешно завершена!")
        else:
            print("База данных уже мигрирована (колонка message_id существует).")
            
    except Exception as e:
        conn.rollback()
        print(f"Ошибка при миграции: {e}")
        raise
    finally:
        conn.close()
else:
    print("Этот скрипт работает только с SQLite. Для PostgreSQL используйте ALTER TABLE вручную.")


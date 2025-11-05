"""
Telegram Summary-Bot
=====================
A Telegram bot you can add to group chats. It collects messages and, on a daily schedule, posts a human-readable summary of the last 24 hours using an LLM (OpenAI by default, pluggable provider).

Key features
------------
- Works in groups; supports multiple group chats.
- Daily scheduled summary per chat (default 23:00 chat local time).
- Commands for admins:
  /summary_now — generate and post a summary for last 24h immediately
  /setsummarytime HH:MM — set local summary time (24h format)
  /settimezone <IANA tz> — set chat timezone (e.g. Asia/Bangkok)
  /setmodel <name> — set LLM model name (e.g. gpt-4o-mini)
  /setmaxlen <n> — cap source text length (characters) to control cost
  /settings — show current settings
- Token-safe summarization: chunks long transcripts and merges partial summaries.
- Opt-out keyword list to ignore specific messages (e.g. /nosummary, bots, stickers) — customizable.
- SQLite by default; Postgres supported via DATABASE_URL.

Deployment quickstart
---------------------
1) Create a bot with BotFather, obtain BOT_TOKEN, and DISABLE PRIVACY MODE so the bot can read group messages (/setprivacy → Disable).
2) Create .env (see template below).
3) `pip install -r requirements.txt`
4) Run `python bot.py` (this file).
5) Add the bot to your group, make it admin (optional but recommended), and run /settings.

.env template
-------------
BOT_TOKEN=123456:ABC...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
# For Postgres use: DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname
DATABASE_URL=sqlite:///summarybot.db
DEFAULT_TZ=Asia/Bangkok
DEFAULT_SUMMARY_TIME=23:00
MAX_SOURCE_CHARS=20000
SYSTEM_PROMPT=You are a helpful assistant that writes concise, neutral daily chat summaries in Russian. Use bullet points, include decisions, action items, deadlines, and unresolved questions. Group by topics. Keep it readable and non-judgmental.

Notes
-----
- Make sure your hosting supports long-running processes (e.g. a VM, Railway, Render, Fly.io, etc.). APScheduler runs timers in-process.
- If your chats are very active, consider Postgres and raising MAX_SOURCE_CHARS, but costs may increase.
- The bot ignores non-text by default; you can expand to captions, polls, etc.
"""

import asyncio
import logging
import os
import re
import textwrap
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pytz
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    Boolean,
    create_engine,
    select,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("summarybot")

# -------------------- Config --------------------
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///summarybot.db")
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Bangkok")
DEFAULT_SUMMARY_TIME = os.getenv("DEFAULT_SUMMARY_TIME", "23:00")  # HH:MM 24h
MAX_SOURCE_CHARS = int(os.getenv("MAX_SOURCE_CHARS", "20000"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | openrouter | anthropic (extensible)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You summarize group chats.")

# -------------------- DB Models --------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class Chat(Base):
    __tablename__ = "chats"
    id = Column(BigInteger, primary_key=True)
    title = Column(String(255), default="")
    timezone = Column(String(64), default=DEFAULT_TZ)
    summary_hour = Column(Integer, default=int(DEFAULT_SUMMARY_TIME.split(":")[0]))
    summary_minute = Column(Integer, default=int(DEFAULT_SUMMARY_TIME.split(":")[1]))
    llm_model = Column(String(64), default=LLM_MODEL)
    max_source_chars = Column(Integer, default=MAX_SOURCE_CHARS)

class Msg(Base):
    __tablename__ = "messages"
    id = Column(BigInteger, primary_key=True)
    chat_id = Column(BigInteger, index=True)
    user_id = Column(BigInteger, index=True)
    username = Column(String(255), default="")
    text = Column(Text)
    created_at = Column(DateTime, index=True, default=datetime.utcnow)

class RunLog(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, index=True)
    ran_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    details = Column(Text, default="")

Base.metadata.create_all(engine)

# -------------------- LLM Provider --------------------
async def call_llm(prompt: str, model: Optional[str] = None) -> str:
    model = model or LLM_MODEL
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    else:
        # Minimal stub to extend for other providers (OpenRouter, Anthropic)
        raise NotImplementedError(f"LLM provider {LLM_PROVIDER} not implemented in this sample")

# -------------------- Helpers --------------------
IGNORE_PATTERNS = [
    r"^/",  # commands
    r"^!",  # other bots
]

def should_ignore(text: str) -> bool:
    for pat in IGNORE_PATTERNS:
        if re.search(pat, text):
            return True
    return False

def ensure_chat(db, chat_id: int, title: str) -> Chat:
    row = db.get(Chat, chat_id)
    if row is None:
        row = Chat(id=chat_id, title=title)
        db.add(row)
        db.commit()
    return row

def chunk_text(s: str, max_len: int) -> List[str]:
    s = s.strip()
    if len(s) <= max_len:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(start + max_len, len(s))
        # try to cut at paragraph boundary
        cut = s.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + int(0.5 * max_len):
            cut = end
        chunks.append(s[start:cut])
        start = cut
    return [c.strip() for c in chunks if c.strip()]

async def summarize_messages(db, chat: Chat, now_utc: datetime) -> Optional[str]:
    tz = pytz.timezone(chat.timezone)
    now_local = now_utc.astimezone(tz)
    since_local = now_local - timedelta(days=1)
    # Convert window back to UTC for DB (we store UTC)
    since_utc = since_local.astimezone(pytz.utc)

    q = (
        select(Msg)
        .where(Msg.chat_id == chat.id)
        .where(Msg.created_at >= since_utc.replace(tzinfo=None))
        .order_by(Msg.created_at.asc())
    )
    rows = db.execute(q).scalars().all()
    if not rows:
        return None

    # Build source text
    lines = []
    for m in rows:
        ts = m.created_at.strftime("%Y-%m-%d %H:%M")
        user = m.username or str(m.user_id)
        line = f"[{ts}] {user}: {m.text}"
        if not should_ignore(m.text):
            lines.append(line)

    if not lines:
        return None

    src = "\n".join(lines)
    chunks = chunk_text(src, chat.max_source_chars)

    # Two-stage summarization for long chats
    partials: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        prompt = textwrap.dedent(f"""
        Summarize the following part ({idx}/{len(chunks)}) of a group chat from the last 24 hours.
        Focus on:
        - key topics and decisions,
        - action items with owners and dates if present,
        - blockers/unresolved questions,
        - notable links/documents.
        Write in Russian. Use concise bullet points.

        === CHAT PART START ===
        {chunk}
        === CHAT PART END ===
        """)
        partial = await call_llm(prompt, model=chat.llm_model)
        partials.append(partial)

    if len(partials) == 1:
        return partials[0]

    merged_prompt = textwrap.dedent(f"""
    You will receive several partial summaries from different segments of the same group chat conversation covering the last 24 hours. Merge them into one cohesive daily summary in Russian with:
    - Topic groups with short headings
    - Decisions (bold the word "Решение:")
    - Action items (prefix with «➡» and assign owners if names appear)
    - Dates or deadlines if they appear
    - Open questions
    Keep it within ~300-500 words.

    === PARTIAL SUMMARIES ===
    {"\n\n".join(partials)}
    """)
    final = await call_llm(merged_prompt, model=chat.llm_model)
    return final

# -------------------- Bot Setup --------------------
bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher()
scheduler = AsyncIOScheduler(timezone=pytz.utc)

async def schedule_job_for_chat(chat: Chat):
    # Remove existing job if any
    job_id = f"chat-{chat.id}"
    job = scheduler.get_job(job_id)
    if job:
        job.remove()

    tz = pytz.timezone(chat.timezone)
    trigger = CronTrigger(hour=chat.summary_hour, minute=chat.summary_minute, timezone=tz)

    scheduler.add_job(
        func=post_daily_summary,
        trigger=trigger,
        id=job_id,
        args=[chat.id],
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    logger.info(f"Scheduled summary for chat {chat.id} at {chat.summary_hour:02d}:{chat.summary_minute:02d} ({chat.timezone})")

async def post_daily_summary(chat_id: int):
    db = SessionLocal()
    try:
        chat = db.get(Chat, chat_id)
        if chat is None:
            return
        summary = await summarize_messages(db, chat, datetime.utcnow().replace(tzinfo=pytz.utc))
        if not summary:
            logger.info(f"No content to summarize for chat {chat_id}")
            return
        text = f"<b>Дневное саммари за последние 24 часа</b>\n\n{summary}"
        await bot.send_message(chat_id=chat_id, text=text)
        db.add(RunLog(chat_id=chat_id, success=True))
        db.commit()
    except Exception as e:
        logger.exception("Summary job failed")
        db.add(RunLog(chat_id=chat_id, success=False, details=str(e)))
        db.commit()
    finally:
        db.close()

# -------------------- Handlers --------------------
@dp.message(Command("start"))
async def cmd_start(message: Message):
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        await schedule_job_for_chat(ch)
        await message.reply(
            "Привет! Я буду собирать сообщения и присылать ежедневное саммари.\n"
            "Добавьте меня в группу, отключите privacy mode у меня в BotFather,\n"
            "а затем настройте время и таймзону: /setsummarytime 23:00, /settimezone Asia/Bangkok.\n"
            "Проверить настройки: /settings"
        )
    finally:
        db.close()

@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        text = (
            f"<b>Настройки чата</b>\n"
            f"Таймзона: {ch.timezone}\n"
            f"Время саммари: {ch.summary_hour:02d}:{ch.summary_minute:02d}\n"
            f"Модель: {ch.llm_model}\n"
            f"MAX_SOURCE_CHARS: {ch.max_source_chars}"
        )
        await message.reply(text)
    finally:
        db.close()

@dp.message(Command("setsummarytime"))
async def cmd_setsummarytime(message: Message):
    m = re.search(r"(\d{1,2}):(\d{2})", message.text)
    if not m:
        await message.reply("Формат: /setsummarytime HH:MM (24ч)")
        return
    hour = int(m.group(1))
    minute = int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        await message.reply("Неверное время")
        return
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        ch.summary_hour = hour
        ch.summary_minute = minute
        db.commit()
        await schedule_job_for_chat(ch)
        await message.reply(f"Время саммари установлено: {hour:02d}:{minute:02d}")
    finally:
        db.close()

@dp.message(Command("settimezone"))
async def cmd_settimezone(message: Message):
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply("Формат: /settimezone <IANA tz>, напр. Asia/Bangkok")
        return
    tzname = parts[1].strip()
    if tzname not in pytz.all_timezones:
        await message.reply("Неизвестная таймзона. Пример: Asia/Bangkok")
        return
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        ch.timezone = tzname
        db.commit()
        await schedule_job_for_chat(ch)
        await message.reply(f"Таймзона установлена: {tzname}")
    finally:
        db.close()

@dp.message(Command("setmodel"))
async def cmd_setmodel(message: Message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.reply("Формат: /setmodel <model_name>")
        return
    model = parts[1].strip()
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        ch.llm_model = model
        db.commit()
        await message.reply(f"Модель обновлена: {model}")
    finally:
        db.close()

@dp.message(Command("setmaxlen"))
async def cmd_setmaxlen(message: Message):
    parts = message.text.split()
    if len(parts) != 2 or not parts[1].isdigit():
        await message.reply("Формат: /setmaxlen <число_символов>")
        return
    val = int(parts[1])
    if val < 2000:
        await message.reply("Минимум 2000 символов")
        return
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        ch.max_source_chars = val
        db.commit()
        await message.reply(f"MAX_SOURCE_CHARS = {val}")
    finally:
        db.close()

@dp.message(Command("summary_now"))
async def cmd_summary_now(message: Message):
    db = SessionLocal()
    try:
        ch = ensure_chat(db, message.chat.id, message.chat.title or "")
        async with ChatActionSender(bot=bot, chat_id=message.chat.id, action="typing"):
            result = await summarize_messages(db, ch, datetime.utcnow().replace(tzinfo=pytz.utc))
        if not result:
            await message.reply("За последние 24 часа нечего суммировать.")
            return
        await message.reply(f"<b>Дневное саммари (ручной запуск)</b>\n\n{result}")
    finally:
        db.close()

# Store messages (only in groups/supergroups, text only for now)
@dp.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}) & F.text)
async def on_group_text(message: Message):
    db = SessionLocal()
    try:
        ensure_chat(db, message.chat.id, message.chat.title or "")
        rec = Msg(
            id=message.message_id + int(message.chat.id) * 10_000_000_000,  # make globally unique
            chat_id=message.chat.id,
            user_id=message.from_user.id if message.from_user else 0,
            username=("@" + message.from_user.username) if (message.from_user and message.from_user.username) else (message.from_user.full_name if message.from_user else ""),
            text=message.text or "",
            created_at=datetime.utcnow(),
        )
        db.add(rec)
        db.commit()
    finally:
        db.close()

# -------------------- Main --------------------
async def on_startup():
    scheduler.start()
    # (Re)load schedules for all chats on boot
    db = SessionLocal()
    try:
        for ch in db.execute(select(Chat)).scalars():
            await schedule_job_for_chat(ch)
    finally:
        db.close()

async def main():
    await on_startup()
    await dp.start_polling(bot, allowed_updates=["message", "chat_member", "my_chat_member"])  # polling mode

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")

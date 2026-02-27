# CosmoDiet — AeroSpace Project

## Инструкция по запуску

---

## ТРЕБОВАНИЯ

- Python 3.10+
- Установленные библиотеки (см. ниже)
- **Batch Runner** — расширение для запуска `.bat` файлов (например, [Batch Runner](https://marketplace.visualstudio.com/items?itemName=NilsSodworker.batch-runner) для VS Code)
- YOLO модели в соответствующих папках:
  - `YOLO\runs\detect\train\weights\best.pt` — CanDefect Detector
  - `MOLDYOLO\yolov8n.pt` — Mold Detector

---

## УСТАНОВКА ЗАВИСИМОСТЕЙ

Откройте терминал в папке `AEROSPACE SITE` и выполните:

```bash
pip install -r requirements.txt
```

**Список зависимостей (requirements.txt):**
- `ultralytics >= 8.0.0` — YOLO детекция
- `opencv-python >= 4.8.0` — обработка изображений
- `numpy >= 1.24.0` — работа с массивами

---

## НАСТРОЙКА API КЛЮЧЕЙ

API ключи хранятся в файле `.env` в папке `AEROSPACE SITE`. Этот файл **не загружается на GitHub** (добавлен в `.gitignore`) для безопасности.

**Если вы клонировали проект, создайте файл `.env`:**

1. Перейдите в папку `AEROSPACE SITE`
2. Создайте файл с именем `.env`
3. Запишите в него ваши ключи:

```
TELEGRAM_BOT_TOKEN=ваш_токен_телеграм_бота
OPENROUTER_API_KEY=ваш_ключ_openrouter
```

**Где получить ключи:**
- **OpenRouter API Key** — зарегистрируйтесь на [openrouter.ai](https://openrouter.ai), перейдите в Keys и создайте ключ (бесплатно)
- **Telegram Bot Token** — создайте бота через [@BotFather](https://t.me/BotFather) в Telegram

> ⚠️ **Без файла `.env` AI-чат и Telegram-бот работать не будут!**

> ✅ **Менять что-то в коде (frontend/backend) НЕ нужно** — просто создайте `.env` и запускайте.

---

## ЗАПУСК СЕРВЕРА

1. Убедитесь, что файл `.env` создан (см. выше)
2. Откройте папку `AEROSPACE SITE`
3. Дважды кликните на файл `run_server.bat`
4. Дождитесь загрузки YOLO моделей (обычно 15–30 секунд)
5. В консоли должно появиться:

```
✓ OPENROUTER_API_KEY загружен: sk-or-v1-...
✓ CanDefect модель загружена успешно!
✓ Mold Detector модель загружена успешно!
Serving on http://localhost:5000
```

6. Откройте в браузере: [http://localhost:5000](http://localhost:5000)

---

## ВАЖНО

- `run_server.bat` автоматически убивает старые процессы на порту 5000 перед запуском, поэтому можно перезапускать без проблем.
- Не закрывайте окно консоли — сервер работает пока оно открыто.
- Файл `.env` загружается автоматически при старте сервера (`server.py` сам читает его).

---

## СТРУКТУРА ПРОЕКТА

```
AeroSpace/
├── AEROSPACE SITE/
│   ├── index.html          — главная страница CosmoDiet
│   ├── diet-builder.html   — конструктор рациона
│   ├── server.py           — Python-сервер (HTTP + API + Telegram бот)
│   ├── run_server.bat      — скрипт запуска сервера
│   ├── .env                — API ключи (⚠️ создать вручную, не загружается на GitHub)
│   ├── requirements.txt    — зависимости Python
│   ├── data.json           — база данных пользователей
│   └── test_openrouter.py  — тест OpenRouter API
├── YOLO/                   — модель CanDefect (дефекты банок)
└── MOLDYOLO/               — модель Mold Detector (плесень)
```

---

## ФУНКЦИИ СЕРВЕРА

| Функция | Эндпоинт |
|---|---|
| Веб-сайт CosmoDiet (статика) | `http://localhost:5000` |
| Регистрация / авторизация | `/api/register`, `/api/login` |
| Биометрия и рационы | `/api/save_bio`, `/api/save_diet` |
| AI-чат (OpenRouter) | `/api/chat` |
| YOLO-детекция дефектов/плесени | `/api/detect` |
| Telegram-бот | автоматический polling |

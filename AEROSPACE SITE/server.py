import json
import os
import threading
import time
import uuid
import hashlib
import urllib.request
import urllib.error
import base64
import random
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Загрузка .env файла
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ[_key.strip()] = _val.strip()

# OpenRouter API (поддерживает различные модели)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "stepfun/step-3.5-flash:free"  # актуальная бесплатная модель
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Debug: показать, какой ключ загружен
if OPENROUTER_API_KEY:
    print(f"✓ OPENROUTER_API_KEY загружен: {OPENROUTER_API_KEY[:20]}...")
else:
    print("✗ OPENROUTER_API_KEY не найден!")

DATA_FILE = Path(__file__).resolve().parent / "data.json"
DATA_LOCK = threading.Lock()

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# YOLO модели для детекции
YOLO_MODELS = {}

print("\n" + "="*60)
print("🤖 ИНИЦИАЛИЗАЦИЯ YOLO ДЕТЕКЦИИ")
print("="*60)

if YOLO_AVAILABLE:
    print(f"✓ YOLO библиотека доступна")
    
    # Первая модель - дефектология (CanDefect)
    YOLO_MODEL_PATH = Path(__file__).resolve().parent.parent / "YOLO" / "runs" / "detect" / "train" / "weights" / "best.pt"
    print(f"\n📦 Попытка загрузки CanDefect модели...")
    print(f"   Путь: {YOLO_MODEL_PATH}")
    print(f"   Существует: {YOLO_MODEL_PATH.exists()}")
    
    if YOLO_MODEL_PATH.exists():
        try:
            model_obj = YOLO(str(YOLO_MODEL_PATH))
            YOLO_MODELS["can_defect"] = {
                "model": model_obj,
                "name": "CanDefect Detector",
                "description": "Детектор дефектов в консервных банках (трещины, вмятины)"
            }
            print(f"✓ CanDefect модель загружена успешно!")
            print(f"  Классы: {model_obj.names}")
        except Exception as e:
            print(f"✗ Ошибка загрузки CanDefect: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ CanDefect модель не найдена в основном пути")
        alt_path = Path(__file__).resolve().parent / "best.pt"
        print(f"   Проверяю альтернативный путь: {alt_path}")
        print(f"   Существует: {alt_path.exists()}")
        if alt_path.exists():
            try:
                model_obj = YOLO(str(alt_path))
                YOLO_MODELS["can_defect"] = {
                    "model": model_obj,
                    "name": "CanDefect Detector",
                    "description": "Детектор дефектов в консервных банках"
                }
                print(f"✓ CanDefect модель загружена из альтернативного пути!")
                print(f"  Классы: {model_obj.names}")
            except Exception as e:
                print(f"✗ Ошибка загрузки CanDefect: {e}")
    
    # Вторая модель - Mold/плесень детектор (MOLDYOLO)
    print(f"\n📦 Попытка загрузки Mold Detector модели...")
    MOLD_MODEL_PATH = Path(__file__).resolve().parent.parent / "MOLDYOLO" / "yolov8n.pt"
    print(f"   Путь: {MOLD_MODEL_PATH}")
    print(f"   Существует: {MOLD_MODEL_PATH.exists()}")
    
    if MOLD_MODEL_PATH.exists():
        try:
            model_obj = YOLO(str(MOLD_MODEL_PATH))
            YOLO_MODELS["mold_detector"] = {
                "model": model_obj,
                "name": "Mold Detector",
                "description": "Детектор плесени и микробиологических загрязнений на продуктах"
            }
            print(f"✓ Mold Detector модель загружена успешно!")
            print(f"  Классы: {model_obj.names}")
        except Exception as e:
            print(f"✗ Ошибка загрузки Mold Detector: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ MOLDYOLO модель не найдена: {MOLD_MODEL_PATH}")
else:
    print(f"✗ Ошибка: YOLO библиотека не установлена!")
    print(f"     Команда установки: pip install ultralytics")

print(f"\n📊 Итого загружено моделей: {len(YOLO_MODELS)}")
if YOLO_MODELS:
    for model_name in YOLO_MODELS:
        print(f"   ✓ {model_name}")
print("="*60 + "\n")

PENDING_LINK = {}

SYSTEM_PROMPT = (
    "Ты - ассистент сайта CosmoDiet. Отвечай на вопросы о сайте, его разделах, функциях и работе. "
    "Можешь отвечать на любые вопросы, но предпочитай ответы, связанные с сайтом. "
    "Отвечай кратко и по делу. Старайся быть дружелюбным и помогающим."
)


def read_data():
    if not DATA_FILE.exists():
        return {"users": []}
    with DATA_LOCK:
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"users": []}


def write_data(data):
    with DATA_LOCK:
        DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def find_user_by_email(data, email):
    return next((u for u in data["users"] if u["email"].lower() == email.lower()), None)


def find_user_by_token(data, token):
    for u in data["users"]:
        if token in u.get("tokens", []):
            return u
    return None


def issue_token(user):
    token = uuid.uuid4().hex
    user.setdefault("tokens", []).append(token)
    return token


def telegram_request(method, payload):
    if not TELEGRAM_BOT_TOKEN:
        return None
    url = TELEGRAM_API.format(token=TELEGRAM_BOT_TOKEN, method=method)
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def send_telegram_message(chat_id, text, keyboard=None):
    payload = {"chat_id": chat_id, "text": text}
    if keyboard:
        payload["reply_markup"] = keyboard
    return telegram_request("sendMessage", payload)


FOOD_RECOMMENDATIONS = [
    "Сублимированная курица", "Лиофилизированные овощи", "Обезвоженные фрукты",
    "Энергетические батончики", "Протеиновые коктейли", "Омега-3 капсулы",
    "Витаминные комплексы", "Минерализованная вода", "Сублимированный творог",
    "Ореховые пасты",
]

GRAVITY_MAP = {
    "1": ("Микрогравитация (МКС)", 0.85),
    "2": ("Лунная гравитация", 0.92),
    "3": ("Марсианская гравитация", 0.95),
}

ACTIVITY_MAP = {
    "1": ("Низкая", 1.2),
    "2": ("Средняя", 1.55),
    "3": ("Высокая", 1.9),
}


def calculate_diet(height, weight, age, activity_factor, gravity_factor):
    """Та же формула что на сайте Diet Builder."""
    bmr = (10 * weight + 6.25 * height - 5 * age + 5) * gravity_factor
    calories = round(bmr * activity_factor)
    protein = round(calories * 0.3 / 4)
    fat = round(calories * 0.25 / 9)
    carbs = round(calories * 0.45 / 4)
    return calories, protein, fat, carbs


def telegram_keyboard():
    return {
        "keyboard": [
            [{"text": "🍽 Рассчитать рацион"}],
            [{"text": "📊 Моя биометрия"}, {"text": "🥗 Мой рацион"}],
            [{"text": "📅 История рационов"}],
            [{"text": "🔗 Привязать аккаунт"}],
        ],
        "resize_keyboard": True,
    }


def handle_telegram_message(message):
    chat_id = message["chat"]["id"]
    text = (message.get("text") or "").strip()

    if text == "/start":
        send_telegram_message(
            chat_id,
            "🚀 Добро пожаловать в CosmoDiet Bot!\n\n"
            "Я помогу рассчитать космический рацион питания.\n"
            "Выберите действие ниже 👇",
            telegram_keyboard(),
        )
        return

    state = PENDING_LINK.get(chat_id, {})
    step = state.get("step", "")

    # ─── Привязка аккаунта ───
    if text in ("🔗 Привязать аккаунт", "Привязать аккаунт"):
        PENDING_LINK[chat_id] = {"step": "email"}
        send_telegram_message(chat_id, "📧 Введите email от аккаунта:")
        return

    if step == "email":
        PENDING_LINK[chat_id] = {"step": "password", "email": text}
        send_telegram_message(chat_id, "🔑 Введите пароль:")
        return

    if step == "password":
        email = state.get("email")
        data = read_data()
        user = find_user_by_email(data, email)
        if not user or user.get("password_hash") != hash_password(text):
            send_telegram_message(chat_id, "❌ Неверный email или пароль.")
            PENDING_LINK.pop(chat_id, None)
            return
        user["telegram_id"] = chat_id
        user["password"] = text
        write_data(data)
        send_telegram_message(chat_id, "✅ Аккаунт успешно привязан!", telegram_keyboard())
        PENDING_LINK.pop(chat_id, None)
        return

    # ─── Расчёт рациона (пошаговый) ───
    if text in ("🍽 Рассчитать рацион", "Рассчитать рацион", "/calc"):
        PENDING_LINK[chat_id] = {"step": "calc_height"}
        send_telegram_message(chat_id, "📏 Шаг 1/5 — Введите ваш рост (см):\n\nНапример: 175")
        return

    if step == "calc_height":
        try:
            h = float(text)
            if h < 50 or h > 250:
                raise ValueError
        except ValueError:
            send_telegram_message(chat_id, "❌ Введите корректный рост (50-250 см):")
            return
        state["height"] = h
        state["step"] = "calc_weight"
        PENDING_LINK[chat_id] = state
        send_telegram_message(chat_id, "⚖️ Шаг 2/5 — Введите ваш вес (кг):\n\nНапример: 70")
        return

    if step == "calc_weight":
        try:
            w = float(text)
            if w < 20 or w > 300:
                raise ValueError
        except ValueError:
            send_telegram_message(chat_id, "❌ Введите корректный вес (20-300 кг):")
            return
        state["weight"] = w
        state["step"] = "calc_age"
        PENDING_LINK[chat_id] = state
        send_telegram_message(chat_id, "🎂 Шаг 3/5 — Введите ваш возраст:\n\nНапример: 30")
        return

    if step == "calc_age":
        try:
            a = int(text)
            if a < 10 or a > 120:
                raise ValueError
        except ValueError:
            send_telegram_message(chat_id, "❌ Введите корректный возраст (10-120):")
            return
        state["age"] = a
        state["step"] = "calc_activity"
        PENDING_LINK[chat_id] = state
        send_telegram_message(
            chat_id,
            "🏃 Шаг 4/5 — Выберите уровень активности:\n\n"
            "1️⃣ — Низкая (сидячая работа)\n"
            "2️⃣ — Средняя (лёгкие тренировки)\n"
            "3️⃣ — Высокая (интенсивные тренировки)\n\n"
            "Введите цифру (1, 2 или 3):",
        )
        return

    if step == "calc_activity":
        if text not in ACTIVITY_MAP:
            send_telegram_message(chat_id, "❌ Введите 1, 2 или 3:")
            return
        state["activity_name"], state["activity_factor"] = ACTIVITY_MAP[text]
        state["step"] = "calc_gravity"
        PENDING_LINK[chat_id] = state
        send_telegram_message(
            chat_id,
            "🌍 Шаг 5/5 — Выберите условия гравитации:\n\n"
            "1️⃣ — Микрогравитация (МКС)\n"
            "2️⃣ — Луна (0.16g)\n"
            "3️⃣ — Марс (0.38g)\n\n"
            "Введите цифру (1, 2 или 3):",
        )
        return

    if step == "calc_gravity":
        if text not in GRAVITY_MAP:
            send_telegram_message(chat_id, "❌ Введите 1, 2 или 3:")
            return
        gravity_name, gravity_factor = GRAVITY_MAP[text]
        h = state["height"]
        w = state["weight"]
        a = state["age"]
        af = state["activity_factor"]

        calories, protein, fat, carbs = calculate_diet(h, w, a, af, gravity_factor)

        foods = random.sample(FOOD_RECOMMENDATIONS, min(6, len(FOOD_RECOMMENDATIONS)))
        foods_str = "\n".join([f"  • {f}" for f in foods])

        date_str = time.strftime("%d.%m.%Y, %H:%M")

        msg = (
            f"✅ Ваш космический рацион рассчитан!\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 Дата: {date_str}\n"
            f"📏 Рост: {h} см | ⚖️ Вес: {w} кг | 🎂 Возраст: {a}\n"
            f"🏃 Активность: {state['activity_name']}\n"
            f"🌍 Гравитация: {gravity_name}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 Калории: {calories} ккал/сутки\n"
            f"🥩 Белки: {protein} г\n"
            f"🧈 Жиры: {fat} г\n"
            f"🍞 Углеводы: {carbs} г\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🍽 Рекомендуемые продукты:\n{foods_str}"
        )
        send_telegram_message(chat_id, msg, telegram_keyboard())

        # Сохраняем в историю привязанного пользователя
        data = read_data()
        user = next((u for u in data["users"] if u.get("telegram_id") == chat_id), None)
        if user:
            diet_entry = {
                "date": date_str,
                "height": h,
                "weight": w,
                "age": a,
                "activity": state["activity_name"],
                "gravity": gravity_name,
                "calories": calories,
                "protein": protein,
                "fat": fat,
                "carbs": carbs,
                "recommendedFoods": foods,
            }
            user.setdefault("diet_history", []).append(diet_entry)
            write_data(data)

        PENDING_LINK.pop(chat_id, None)
        return

    # ─── Биометрия ───
    if text in ("/bio", "📊 Моя биометрия", "Моя биометрия"):
        data = read_data()
        user = next((u for u in data["users"] if u.get("telegram_id") == chat_id), None)
        if not user:
            send_telegram_message(chat_id, "⚠️ Аккаунт не привязан. Нажмите «🔗 Привязать аккаунт».")
            return
        bio = user.get("bio_history", [])
        if not bio:
            send_telegram_message(chat_id, "📭 Нет сохраненной биометрии.")
            return
        last = bio[-1]
        msg = (
            f"📊 Биометрия ({last.get('date')}):\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📏 Рост: {last.get('height')} см\n"
            f"⚖️ Вес: {last.get('weight')} кг\n"
            f"🎂 Возраст: {last.get('age')}\n"
            f"💓 Пульс: {last.get('pulse')}\n"
            f"🏃 Активность: {last.get('activity')}\n"
            f"😰 Стресс: {last.get('stressLevel')}\n"
            f"🕐 Длительность: {last.get('missionDuration')} дн.\n"
            f"🌍 Гравитация: {last.get('gravity')}"
        )
        send_telegram_message(chat_id, msg)
        return

    # ─── Последний рацион ───
    if text in ("/diet", "🥗 Мой рацион", "Мой рацион"):
        data = read_data()
        user = next((u for u in data["users"] if u.get("telegram_id") == chat_id), None)
        if not user:
            send_telegram_message(chat_id, "⚠️ Аккаунт не привязан. Нажмите «🔗 Привязать аккаунт».")
            return
        diets = user.get("diet_history", [])
        if not diets:
            send_telegram_message(chat_id, "📭 Нет сохраненных рационов.\nНажмите «🍽 Рассчитать рацион» чтобы создать первый!")
            return
        last = diets[-1]
        foods = last.get("recommendedFoods", [])
        foods_str = "\n".join([f"  • {f}" for f in foods]) if foods else "Нет рекомендаций"
        msg = (
            f"🥗 Последний рацион ({last.get('date')}):\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 Калории: {last.get('calories')} ккал\n"
            f"🥩 Белки: {last.get('protein')} г\n"
            f"🧈 Жиры: {last.get('fat')} г\n"
            f"🍞 Углеводы: {last.get('carbs')} г\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🍽 Рекомендуемые продукты:\n{foods_str}"
        )
        send_telegram_message(chat_id, msg)
        return

    # ─── История рационов ───
    if text in ("📅 История рационов", "История рационов", "/history"):
        data = read_data()
        user = next((u for u in data["users"] if u.get("telegram_id") == chat_id), None)
        if not user:
            send_telegram_message(chat_id, "⚠️ Аккаунт не привязан. Нажмите «🔗 Привязать аккаунт».")
            return
        diets = user.get("diet_history", [])
        if not diets:
            send_telegram_message(chat_id, "📭 Нет сохраненных рационов.\nНажмите «🍽 Рассчитать рацион» чтобы создать первый!")
            return

        msg = f"📅 История ваших рационов ({len(diets)} шт.):\n━━━━━━━━━━━━━━━━━━━━━\n"
        for i, d in enumerate(diets, 1):
            msg += f"{i}. 📋 {d.get('date')} — {d.get('calories')} ккал\n"
        msg += f"\n━━━━━━━━━━━━━━━━━━━━━\nОтправьте номер рациона (1-{len(diets)}) чтобы посмотреть подробности:"

        PENDING_LINK[chat_id] = {"step": "pick_diet"}
        send_telegram_message(chat_id, msg)
        return

    if step == "pick_diet":
        data = read_data()
        user = next((u for u in data["users"] if u.get("telegram_id") == chat_id), None)
        diets = user.get("diet_history", []) if user else []
        try:
            idx = int(text) - 1
            if idx < 0 or idx >= len(diets):
                raise ValueError
        except ValueError:
            send_telegram_message(chat_id, f"❌ Введите число от 1 до {len(diets)}:")
            return

        d = diets[idx]
        foods = d.get("recommendedFoods", [])
        foods_str = "\n".join([f"  • {f}" for f in foods]) if foods else "Нет рекомендаций"
        msg = (
            f"📋 Рацион #{idx + 1} ({d.get('date')}):\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📏 Рост: {d.get('height', '—')} см | ⚖️ Вес: {d.get('weight', '—')} кг\n"
            f"🎂 Возраст: {d.get('age', '—')}\n"
            f"🏃 Активность: {d.get('activity', '—')}\n"
            f"🌍 Гравитация: {d.get('gravity', '—')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔥 Калории: {d.get('calories')} ккал/сутки\n"
            f"🥩 Белки: {d.get('protein')} г\n"
            f"🧈 Жиры: {d.get('fat')} г\n"
            f"🍞 Углеводы: {d.get('carbs')} г\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🍽 Рекомендуемые продукты:\n{foods_str}"
        )
        send_telegram_message(chat_id, msg, telegram_keyboard())
        PENDING_LINK.pop(chat_id, None)
        return


def telegram_polling():
    if not TELEGRAM_BOT_TOKEN:
        return
    offset = 0
    while True:
        try:
            resp = telegram_request("getUpdates", {"timeout": 20, "offset": offset})
            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                if "message" in update:
                    handle_telegram_message(update["message"])
        except Exception:
            time.sleep(2)


class Handler(SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self.handle_api()
            return
        self.send_error(404)

    def handle_api(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if self.path == "/api/register":
            name = payload.get("name", "").strip()
            email = payload.get("email", "").strip()
            password = payload.get("password", "")
            if not name or not email or not password:
                return self.send_json({"error": "Invalid data"}, 400)
            data = read_data()
            if find_user_by_email(data, email):
                return self.send_json({"error": "Email already exists"}, 400)
            user = {
                "id": uuid.uuid4().hex,
                "name": name,
                "email": email,
                "password": password,
                "password_hash": hash_password(password),
                "regDate": time.strftime("%d.%m.%Y, %H:%M:%S"),
                "tokens": [],
                "bio_history": [],
                "diet_history": [],
            }
            token = issue_token(user)
            data["users"].append(user)
            write_data(data)
            return self.send_json({"user": {k: user[k] for k in ["id", "name", "email", "regDate"]}, "token": token})

        if self.path == "/api/login":
            email = payload.get("email", "").strip()
            password = payload.get("password", "")
            data = read_data()
            user = find_user_by_email(data, email)
            if not user or user.get("password_hash") != hash_password(password):
                return self.send_json({"error": "Invalid credentials"}, 401)
            token = issue_token(user)
            write_data(data)
            return self.send_json({"user": {k: user[k] for k in ["id", "name", "email", "regDate"]}, "token": token})

        if self.path == "/api/me":
            token = payload.get("token")
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            return self.send_json({"user": {k: user[k] for k in ["id", "name", "email", "regDate"]}})

        if self.path == "/api/save_bio":
            token = payload.get("token")
            bio = payload.get("bio", {})
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            user.setdefault("bio_history", []).append(bio)
            write_data(data)
            return self.send_json({"ok": True})

        if self.path == "/api/save_diet":
            token = payload.get("token")
            diet = payload.get("diet", {})
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            user.setdefault("diet_history", []).append(diet)
            write_data(data)
            return self.send_json({"ok": True})

        if self.path == "/api/get_bio":
            token = payload.get("token")
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            return self.send_json({"bio": user.get("bio_history", [])})

        if self.path == "/api/get_diets":
            token = payload.get("token")
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            return self.send_json({"diets": user.get("diet_history", [])})

        if self.path == "/api/telegram/status":
            token = payload.get("token")
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            return self.send_json({"linked": bool(user.get("telegram_id"))})

        if self.path == "/api/telegram/test":
            token = payload.get("token")
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            chat_id = user.get("telegram_id")
            if not chat_id:
                return self.send_json({"error": "Not linked"}, 400)
            send_telegram_message(chat_id, "Тестовое уведомление от CosmoDiet ✅")
            return self.send_json({"ok": True})

        if self.path == "/api/chat":
            print(f"\n✓ /api/chat вызван!")
            messages = payload.get("messages", [])
            print(f"  messages count: {len(messages)}")

            if not OPENROUTER_API_KEY:
                print("✗ OPENROUTER_API_KEY не задан!")
                return self.send_json({"reply": "Ошибка: API ключ OpenRouter не задан. Запускайте сервер через run_server.bat."})

            # Формируем список сообщений с системным промптом
            full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

            try:
                body = json.dumps({
                    "model": OPENROUTER_MODEL,
                    "messages": full_messages,
                    "max_tokens": 512,
                    "temperature": 0.7,
                }).encode("utf-8")

                req = urllib.request.Request(
                    OPENROUTER_API_URL,
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "http://localhost:5000",
                        "X-Title": "CosmoDiet AI",
                    },
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                reply = result["choices"][0]["message"]["content"].strip()
                print(f"✓ OpenRouter ответ получен, длина: {len(reply)}")
                return self.send_json({"reply": reply})

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                print(f"✗ OpenRouter HTTP ошибка {e.code}: {error_body}")
                return self.send_json({"reply": f"Ошибка OpenRouter ({e.code}): {error_body}"}, 502)
            except Exception as e:
                print(f"✗ Ошибка запроса к OpenRouter: {e}")
                import traceback
                traceback.print_exc()
                return self.send_json({"reply": f"Не удалось получить ответ от ИИ: {str(e)}"}, 502)

        if self.path == "/api/detect":
            if not YOLO_MODELS:
                return self.send_json({"error": "YOLO модели не загружены"}, 500)
            
            token = payload.get("token")
            image_data = payload.get("image", "")
            model_type = payload.get("model", "can_defect")  # Выбор модели
            
            if model_type not in YOLO_MODELS:
                return self.send_json({"error": f"Модель '{model_type}' не найдена. Доступные: {list(YOLO_MODELS.keys())}"}, 400)
            
            # Токен опционален (позволяем diet-builder работать без авторизации)
            data = read_data()
            if token:
                user = find_user_by_token(data, token)
                if not user:
                    print(f"⚠ Invalid token for /api/detect")
            
            try:
                # Декодируем base64 изображение
                print(f"📥 Получено изображение, начальный размер: {len(image_data)} символов")
                
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                    print(f"📋 После удаления заголовка: {len(image_data)} символов")
                
                if not image_data:
                    print(f"❌ Пустые данные изображения")
                    return self.send_json({"error": "No image data provided"}, 400)
                
                image_bytes = base64.b64decode(image_data)
                print(f"✓ Декодировано {len(image_bytes)} байт из base64")
                
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print(f"❌ Ошибка декодирования изображения")
                    return self.send_json({"error": "Failed to decode image"}, 400)
                
                print(f"✓ Изображение декодировано, размер: {frame.shape}")
                
                # Получаем выбранную модель
                model_info = YOLO_MODELS[model_type]
                yolo_model = model_info["model"]
                
                print(f"🎯 Выполняю детекцию с моделью: {model_type} ({model_info['name']})")
                
                # Выполняем детекцию
                results = yolo_model(frame, verbose=False, conf=0.3)
                print(f"✓ YOLO обработка завершена, результат объектов: {len(results) if results else 0}")
                
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    names = result.names
                    
                    print(f"✓ Обнаружено боксов: {len(boxes)}")
                    
                    for img_idx, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = names.get(cls_id, f"Class {cls_id}")
                        
                        detections.append({
                            "class": class_name,
                            "confidence": round(conf, 3)
                        })
                        print(f"  └─ #{img_idx+1}: {class_name} ({conf*100:.1f}%)")
                        
                    print(f"✓ Обнаружено объектов: {len(detections)}")
                else:
                    print(f"✓ Объекты не обнаружены (пусто)")
                
                # Отрисовываем аннотации на изображении (даже если детекций нет)
                try:
                    if results and len(results) > 0:
                        annotated_frame = results[0].plot()
                        print(f"✓ Аннотированное изображение созданo")
                    else:
                        # Если детекций нет, просто копируем оригинальное изображение
                        annotated_frame = frame.copy()
                        print(f"✓ Используется оригинальное изображение (детекций нет)")
                    
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    annotated_b64 = base64.b64encode(buffer).decode()
                    print(f"✓ Изображение закодировано в base64, размер: {len(annotated_b64)} символов")
                except Exception as plot_err:
                    print(f"⚠ Ошибка при отрисовке аннотаций: {plot_err}")
                    # Если ошибка при отрисовке, используем оригинальное изображение
                    _, buffer = cv2.imencode('.jpg', frame)
                    annotated_b64 = base64.b64encode(buffer).decode()
                
                response = {
                    "model": model_type,
                    "model_name": model_info["name"],
                    "detections": detections,
                    "annotated_image": f"data:image/jpeg;base64,{annotated_b64}"
                }
                print(f"✓ Возвращаю ответ с {len(detections)} детекциями")
                return self.send_json(response)
            except Exception as e:
                print(f"❌ Detection error: {e}")
                import traceback
                traceback.print_exc()
                return self.send_json({"error": str(e)}, 500)

        if self.path == "/api/save_detection_session":
            token = payload.get("token")
            session = payload.get("session", {})
            
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            
            user.setdefault("detection_sessions", []).append(session)
            write_data(data)
            return self.send_json({"ok": True})

        if self.path == "/api/get_detection_sessions":
            token = payload.get("token")
            
            data = read_data()
            user = find_user_by_token(data, token)
            if not user:
                return self.send_json({"error": "Unauthorized"}, 401)
            
            sessions = user.get("detection_sessions", [])
            return self.send_json({"sessions": sessions})

        return self.send_json({"error": "Not found"}, 404)

    def send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

def main():
    root = Path(__file__).resolve().parent
    handler = lambda *args, **kwargs: Handler(*args, directory=str(root), **kwargs)
    server = ThreadingHTTPServer(("0.0.0.0", 5000), handler)
    print("Serving on http://localhost:5000")
    if TELEGRAM_BOT_TOKEN:
        threading.Thread(target=telegram_polling, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()

from ultralytics import YOLO

# Путь к твоему обученному весу
model = YOLO(r"runs/detect/train/weights/best.pt")

# Запуск камеры (0 — основная вебка)
model.predict(
    source=0,
    show=True,
    conf=0.4
)

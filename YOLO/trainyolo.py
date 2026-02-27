from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="CanDefect/data.yaml",
    epochs=55,
    imgsz=640,
    batch=8
)

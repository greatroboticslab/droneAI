from ultralytics import YOLO

model = YOLO("./yolo_models/yolov8n-cls.pt")
results = model.train(data="./classification", epochs=100, imgsz=640, project='./results')

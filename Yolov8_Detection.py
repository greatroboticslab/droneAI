from ultralytics import YOLO

model = YOLO("./yolo_models/yolov8n.pt")
results = model.train(data="./detection/data.yaml", epochs=100, imgsz=640, project='./results')
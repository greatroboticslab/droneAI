# purpose of this file is to get the labels for each model version

from ultralytics import YOLO

def load_models_and_get_labels(model_paths):
    model_labels = {}

    for path in model_paths:
        model = YOLO(path)
        labels = model.names
        model_labels[path] = labels

    return model_labels

model_paths = [
    "./results/landing_class_v1/weights/best.pt",
    "./results/landing_class_v2/weights/best.pt",
    "./results/landing_class_v3/weights/best.pt"
]

labels_dict = load_models_and_get_labels(model_paths)
for model_path, labels in labels_dict.items():
    print(f"Labels for {model_path}: {labels}")

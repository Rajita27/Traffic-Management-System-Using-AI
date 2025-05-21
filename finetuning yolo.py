from ultralytics import YOLO

# Load the model
model = YOLO("yolov5m.pt")

# Train the model on your dataset
model.train(
    data="/Users/rajitamaharjan/PycharmProjects/TrafficManagementSystemUsingAI/combined dataset/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    project="runs/train",
    name="vehicle_finetuning",
    exist_ok=True,
)

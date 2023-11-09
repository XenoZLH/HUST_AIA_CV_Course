from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='my_dataset.yaml', epochs=100, imgsz=640)

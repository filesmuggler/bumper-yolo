import torch
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

results = model.train(data="/home/yerba/Datasets/bumper/data.yaml", epochs=10)  # train the model
results = model.val()  # evaluate model performance on the validation set
results = model("/home/yerba/Projects/bumper_project/data/porsche.jpeg")  # predict on an image
success = model.export(format="onnx")  # export the model to ONNX format
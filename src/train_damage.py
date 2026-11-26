import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/damage/data.yaml")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()
    YOLO("yolov8n.pt").train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, project="runs", name="damage")

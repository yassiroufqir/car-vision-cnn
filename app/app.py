import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

CLS_PATH = "models/classifier.pt"
DMG_PATH = "runs/damage/weights/best.pt"

ckpt = torch.load(CLS_PATH, map_location="cpu")
classes = ckpt["classes"]
clf = models.efficientnet_b0()
clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, len(classes))
clf.load_state_dict(ckpt["model"])
clf.eval()
det = YOLO(DMG_PATH)

tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def predict(img):
    with torch.no_grad():
        probs = torch.softmax(clf(tf(img).unsqueeze(0)), 1)[0]
    top = probs.topk(3)
    labels = {classes[i]: float(p) for p, i in zip(top.values, top.indices)}
    annotated = det(img)[0].plot()[..., ::-1]
    return labels, annotated

gr.Interface(predict, gr.Image(type="pil"),
             [gr.Label(num_top_classes=3, label="Make / model"), gr.Image(label="Damage")],
             title="Car Vision CNN").launch()

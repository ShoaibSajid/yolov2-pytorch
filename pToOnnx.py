from torch.autograd import Variable

import torch.onnx
import torchvision
import torch
from yolov2_tiny_2 import Yolov2

dummy_input = Variable(torch.randn(1, 3, 256, 256))
state_dict = torch.load('data/pretrained/yolov2_best_map.pth')
model = Yolov2(classes=["Vehicle", "Rider", "Pedestrian"])
model.load_state_dict(state_dict['model'])
torch.onnx.export(model, dummy_input, "best.onnx")
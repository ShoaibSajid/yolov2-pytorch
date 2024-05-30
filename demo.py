import os
import argparse
import time
import torch
from torch.autograd import Variable
from PIL import Image
from test import prepare_im_data
# from yolov2 import Yolov2
from yolov2_tiny_2 import Yolov2
from yolo_eval import yolo_eval
from util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from util.network import WeightLoader
import cv2
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160', type=str)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--data', type=str,
                        default=None,
                        help='Path to txt file containing images list')

    args = parser.parse_args()
    return args

def drawBox(label:np.array, img:np.ndarray, classes, rel=False):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    if label.size == 6:
        box = [label[0], label[1], label[2], label[3]]
        cls =  classes[int(label[-1])]
        conf = round(label[-2], 2)
    elif label.size == 5:
        box = [label[0], label[1], label[2], label[3]]
        conf = label[-1]    
    elif label.size == 4:    
        box = [label[0], label[1], label[2], label[3]]
    else:
        raise ValueError("Invalid size array only accept arrays of size 4 or 5")    
    
    # color = list(np.random.random(size=3) * 256)
    color = [(0,0,255), (0,255,0), (255,0,0)]
    if not cls:
        cls = ''
    if not conf:
        conf = ''    
    text = f"{cls} {conf}:"
    if rel:
        img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), 
                            (int(box[2]*w), int(box[3]*h)), 
                            color[int(label[-1])], 2)
        
        cv2.putText(img, text, (int((box[0]*w)+2), int((box[1]*h) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[int(label[-1])], 1)
    else:
        img = cv2.rectangle(img,(int(box[0]), int(box[1])), 
                            (int(box[2]), int(box[3])), 
                            color[int(label[-1])], 2)
        
        cv2.putText(img, text, (int((box[0])+2), int((box[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    color[int(label[-1])], 1)
    return img

def showImg(img, labels, meta=False, cls='', relative=False):
    # Convert the tensor to a numpy array
    _img = img
    # image_np = _image.numpy().transpose((1, 2, 0))
    # image_np = std * image_np + mean
    # image_np = np.clip(image_np, 0, 1)*255
    # _img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    _img = np.array(_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    if meta:
        _img = cv2.resize(_img, (int(meta['width'].item()),  int(meta['height'].item())), interpolation= cv2.INTER_LINEAR)
    for i in range(labels.shape[0]):
        label = labels[i]
        conf = label[-2]
        if conf >= 0.25:
            if relative:
                _img = drawBox(label, _img, True)
            else:    
                _img = drawBox(label, _img, cls)
    cv2.imshow('', _img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def demo():
    args = parse_args()
    print('call with args: {}'.format(args))

    if args.data==None:

        # input images
        images_dir = 'images'
        images_names = ['image1.jpg', 'image2.jpg']
    else:
        with open(args.data, 'r') as f:
            images_names = f.readlines()
    
    # classes = ('aeroplane', 'bicycle', 'bird', 'boat',
    #                         'bottle', 'bus', 'car', 'cat', 'chair',
    #                         'cow', 'diningtable', 'dog', 'horse',
    #                         'motorbike', 'person', 'pottedplant',
    #                         'sheep', 'sofa', 'train', 'tvmonitor')
    classes = ('Vehicle', 'Rider', 'Person')        
    
    model = Yolov2(classes=classes)
    # weight_loader = WeightLoader()
    # weight_loader.load(model, 'yolo-voc.weights')
    # print('loaded')

    # model_path = os.path.join(args.output_dir, args.model_name + '.pth')
    model_path = args.model_name
    print('loading model from {}'.format(model_path))
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location=('cuda:0'))
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if args.use_cuda:
        model.cuda()

    model.eval()
    print('model loaded')

    for image_name in images_names:
        if args.data==None:
            image_path = os.path.join(images_dir, image_name)
            img = Image.open(image_path)
        else:
            image_path = image_name.split('\n')[0]
            img = Image.open(image_path)   
        im_data, im_info = prepare_im_data(img)

        if args.use_cuda:
            im_data_variable = Variable(im_data).cuda()
        else:
            im_data_variable = Variable(im_data)

        tic = time.time()

        yolo_output = model(im_data_variable)
        yolo_output = [item[0].data for item in yolo_output]
        detections = yolo_eval(yolo_output, im_info, conf_threshold=0.05, nms_threshold=0.45)

        toc = time.time()
        cost_time = toc - tic
        print('im detect, cost time {:4f}, FPS: {}'.format(
            toc-tic, int(1 / cost_time)))

        
        if len(detections)>0:
            det_boxes = detections[:, :5].cpu().numpy()
            det_classes = detections[:, -1].long().cpu().numpy()
            pred = np.zeros((det_boxes.shape[0],6))
            pred[:, :5] = det_boxes
            pred[:,-1] = det_classes
            showImg(img, pred, cls=classes)
            # im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=classes)
        # else:
        #     im2show = img
        # plt.figure()
        # plt.imshow(im2show)
        # plt.show()

if __name__ == '__main__':
    demo()

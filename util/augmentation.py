import cv2
import numpy as np
from PIL import Image
from config import config as cfg
import pdb
import random

def random_scale_translation(img, boxes, jitter=0.2):
    """

    Arguments:
    img -- PIL.Image
    boxes -- numpy array of shape (N, 4) N is number of boxes
    factor -- max scale size
    im_info -- dictionary {width:, height:}

    Returns:
    im_data -- numpy.ndarray
    boxes -- numpy array of shape (N, 4)
    """

    w, h = img.size

    dw = int(w*jitter)
    dh = int(h*jitter)

    pl = np.random.randint(-dw, dw)
    pr = np.random.randint(-dw, dw)
    pt = np.random.randint(-dh, dh)
    pb = np.random.randint(-dh, dh)

    # scaled width, scaled height
    sw = w - pl - pr
    sh = h - pt - pb

    cropped = img.crop((pl, pt, pl + sw - 1, pt + sh - 1))

    # update boxes accordingly
    # print(boxes.shape)
    boxes[:, 0::2] -= pl
    boxes[:, 1::2] -= pt

    # clamp boxes
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, sw-1)
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, sh-1)

    # if flip
    if np.random.randint(2):
        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        boxes[:, 0::2] = (sw-1) - boxes[:, 2::-2]

    return cropped, boxes


def convert_color(img, source, dest):
    """
    Convert color

    Arguments:
    img -- numpy.ndarray
    source -- str, original color space
    dest -- str, target color space.

    Returns:
    img -- numpy.ndarray
    """

    if source == 'RGB' and dest == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif source == 'HSV' and dest == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def rand_scale(s):
    scale = np.random.uniform(1, s)
    if np.random.randint(1, 10000) % 2:
        return scale
    return 1./scale


def random_distort(img, hue=.1, sat=1.5, val=1.5):

    hue = np.random.uniform(-hue, hue)
    sat = rand_scale(sat)
    val = rand_scale(val)

    img = img.convert('HSV')
    cs = list(img.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    img = Image.merge(img.mode, tuple(cs))

    img = img.convert('RGB')
    return img


def random_hue(img, rate=.1):
    """
    adjust hue
    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue
    Returns:
    img -- numpy.ndarray
    """

    delta = rate * 360.0 / 2

    if np.random.randint(2):
        img[:, :, 0] += np.random.uniform(-delta, delta)
        img[:, :, 0] = np.clip(img[:, :, 0], a_min=0.0, a_max=360.0)

    return img


def random_saturation(img, rate=1.5):
    """
    adjust saturation

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 1] *= np.random.uniform(lower, upper)
        img[:, :, 1] = np.clip(img[:, :, 1], a_min=0.0, a_max=1.0)

    return img


def random_exposure(img, rate=1.5):
    """
    adjust exposure (In fact, this function change V (HSV))

    Arguments:
    img -- numpy.ndarray
    rate -- float, factor used to adjust hue

    Returns:
    img -- numpy.ndarray
    """

    lower = 0.5  # hard code
    upper = rate

    if np.random.randint(2):
        img[:, :, 2] *= np.random.uniform(lower, upper)
        img[:, :, 2] = np.clip(img[:, :, 2], a_min=0.0, a_max=255.0)

    return img

def UpSampling(image, boxes, sf=2):
    """
    Upsampling images & boxes by scale_factor
    """
    width, height = image.size
    image = image.resize(size=(int(round(number=width*sf, ndigits=0)), int(round(number=height*sf, ndigits=0))))
    width, height = image.size
    boxes[:,0::2] *= width  # box[0], box[2] *= sf X width(former)
    boxes[:,1::2] *= height # box[1], box[3] *= sf * height(former)

    return image, boxes, width, height

def OptimumArea(boxes, bestLeft, bestTop, bestRight, bestBottom, most_width, most_height):
    """
    Find optimum Left, Top Right, Bottom to crop the image"""
    for box in boxes:
        box_width, box_height = box[2] - box[0], box[3] - box[1]

        if box_width * box_height > (0.25 * most_width) * (0.25 * most_height):
            if bestLeft > box[0]:
                bestLeft = box[0]
            
            if bestTop > box[1]:
                bestTop = box[1]
            
            if bestRight < box[2]:
                bestRight = box[2]
            
            if bestBottom < box[3]:
                bestBottom = box[3]

    return bestLeft, bestTop, bestRight, bestBottom

def Padding(image, bestLeft, bestTop, bestRight, bestBottom, alpha=0.05):
    width, height = image.size

    if 0 < bestLeft < alpha*width:
        bestLeft = 0
    else:
        bestLeft -= alpha * width

    if 0 < bestTop < alpha*height:
        bestTop = 0
    else:
        bestTop -= alpha * height

    if (1-alpha)*width < bestRight < width:
        bestRight = width
    else:
        bestRight += alpha * width

    if (1-alpha)*height < bestBottom < height:
        bestBottom = height
    else:
        bestBottom += alpha * height

    return bestLeft, bestTop, bestRight, bestBottom

def AdjustBBOX(boxes, x1, y1, image):
    """
    move (x1,y1) to (0, 0) and this will be the basis\n
    And then move all boxes based on basis
    """
    width, height = image.size
    boxes[:,0::2] -= x1
    boxes[:,1::2] -= y1
    boxes[:,0::2] = np.clip(boxes[:,0::2], 0, width-1)
    boxes[:,1::2] = np.clip(boxes[:,1::2], 0, height-1)

    return boxes

def Normalization(boxes, image):
    width, height = image.size
    boxes[:,0::2] /= width  # box[0], box[2] /= width
    boxes[:,1::2] /= height # box[1], box[3] /= height

    return boxes

def scaleAndcrop(image, boxes, classes):
    boxes = Normalization(boxes, image)
    mostLeft, mostTop, mostRight, mostBottom = 1, 1, 0, 0
    for box in boxes:
        mostLeft, mostTop, mostRight, mostBottom = min(box[0], mostLeft), min(box[1], mostTop), max(box[2], mostRight), max(box[3], mostBottom)

    # Upsampling image, boxes
    image, boxes, width, height = UpSampling(image=image, boxes=boxes, sf=2)
    # Upsampling mostLeft, mostRight, mostTop, mostBottom
    mostLeft, mostRight, mostTop, mostBottom = int(round(number=mostLeft*width, ndigits=0)), int(round(number=mostRight*width, ndigits=0)), int(round(number=mostTop*height, ndigits=0)), int(round(number=mostBottom*height, ndigits=0))

    # Find the optimum cropping area
    bestLeft, bestTop, bestRight, bestBottom = np.average(a=boxes, axis=0)
    most_width, most_height = mostRight - mostLeft, mostBottom - mostTop
    bestLeft, bestTop, bestRight, bestBottom = OptimumArea(boxes=boxes, bestLeft=bestLeft, bestTop=bestTop, bestRight=bestRight, bestBottom=bestBottom, most_width=most_width, most_height=most_height)
    # Padding
    bestLeft, bestTop, bestRight, bestBottom = Padding(image=image, bestLeft=bestLeft, bestTop=bestTop, bestRight=bestRight, bestBottom=bestBottom)

    # cropping by best Left, Top, Right, Bottom
    image = image.crop(box=(bestLeft, bestTop, bestRight, bestBottom))
    width, height = image.size
    # Adjust the bounding box coordinate
    boxes = AdjustBBOX(boxes=boxes, x1=bestLeft, y1=bestTop, image=image)

    # keep the box that is not width == 0 | height == 0
    keep     = (boxes[:, 0] != boxes[:, 2]) & (boxes[:, 1] != boxes[:, 3])
    boxes   = boxes[keep, :]
    classes = classes[keep]

    # keep the box that width > 7 & height > 7
    keep =  ((boxes[:, 2] - boxes[:, 0]) > 7) & ((boxes[:, 3] - boxes[:, 1])>7)
    boxes   = boxes[keep, :]
    classes = classes[keep]

########################################################################################################################
    # image__ = np.array(image)
    # image__ = cv2.cvtColor(image__, cv2.COLOR_RGB2BGR)
    # for i in range(boxes.shape[0]):
    #     label = boxes[i]
    #     x1, y1, x2, y2 =  label[0], label[1], label[2], label[3]
    #     cv2.rectangle(img=image__, pt1=(int(x1),int(y1)), pt2=(int(x2),int(y2)), color=(0, 255, 0), thickness=2)
    # cv2.imshow(winname='image & boxes', mat=image__)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()

    return image, boxes, classes

def augment_img(img, boxes, gt_classes, scaleCrop=False):
    """
    Apply data augmentation.
    1. convert color to HSV
    2. adjust hue(.1), saturation(1.5), exposure(1.5)
    3. convert color to RGB
    4. random scale (up to 20%)
    5. translation (up to 20%)
    6. resize to given input size.

    Arguments:
    img -- PIL.Image object
    boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    im_info -- dictionary {width:, height:}

    Returns:
    au_img -- numpy array of shape (H, W, 3)
    au_boxes -- numpy array of shape (N, 4) N is number of boxes, (x1, y1, x2, y2)
    au_gt_classes -- numpy array of shape (N). ground truth class index 0 ~ (N-1)
    """
    # img = np.array(img).astype(np.float32)
    # _img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # _box = boxes[0]
    # cv2.rectangle(_img, (int(_box[0]), int(_box[1]), (int(_box[2]), int(_box[3]))), (0,0,255), 1)
    # cv2.imshow('', _img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    boxes = np.copy(boxes).astype(np.float32)
    if random.choice([True,False]) & scaleCrop:
        img_t, boxes_t, gt_classes_t = scaleAndcrop(img, boxes, gt_classes)
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            gt_classes = gt_classes_t

    for i in range(5):
        img_t, boxes_t = random_scale_translation(img.copy(), boxes.copy(), jitter=cfg.jitter)
        # print('Trace here----//')
        # print(boxes_t)
        # pdb.set_trace()
        keep = (boxes_t[:, 0] != boxes_t[:, 2]) & (boxes_t[:, 1] != boxes_t[:, 3])
        boxes_t = boxes_t[keep, :]
        if boxes_t.shape[0] > 0:
            img = img_t
            boxes = boxes_t
            gt_classes = gt_classes[keep]
            break

    img = random_distort(img, cfg.hue, cfg.saturation, cfg.exposure)
    
    # img_ = np.array(img)
    # img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
    # for i in range(boxes.shape[0]):
    #     label = boxes[i]
    #     x =  label[0]
    #     y =  label[1]
    #     p =  label[2]
    #     q =  label[3]
    #     cv2.rectangle(img_, (int(x),int(y)), (int(p),int(q)), (0,0,255), 2)
    # cv2.imshow('', img_)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return img, boxes, gt_classes
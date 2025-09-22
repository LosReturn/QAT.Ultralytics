import argparse
import colorsys
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_CATEGORIES = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "airplane": 5,
    "bus": 6,
    "train": 7,
    "truck": 8,
    "boat": 9,
    "traffic light": 10,
    "fire hydrant": 11,
    "stop sign": 13,
    "parking meter": 14,
    "bench": 15,
    "bird": 16,
    "cat": 17,
    "dog": 18,
    "horse": 19,
    "sheep": 20,
    "cow": 21,
    "elephant": 22,
    "bear": 23,
    "zebra": 24,
    "giraffe": 25,
    "backpack": 27,
    "umbrella": 28,
    "handbag": 31,
    "tie": 32,
    "suitcase": 33,
    "frisbee": 34,
    "skis": 35,
    "snowboard": 36,
    "sports ball": 37,
    "kite": 38,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard": 41,
    "surfboard": 42,
    "tennis racket": 43,
    "bottle": 44,
    "wine glass": 46,
    "cup": 47,
    "fork": 48,
    "knife": 49,
    "spoon": 50,
    "bowl": 51,
    "banana": 52,
    "apple": 53,
    "sandwich": 54,
    "orange": 55,
    "broccoli": 56,
    "carrot": 57,
    "hot dog": 58,
    "pizza": 59,
    "donut": 60,
    "cake": 61,
    "chair": 62,
    "couch": 63,
    "potted plant": 64,
    "bed": 65,
    "dining table": 67,
    "toilet": 70,
    "tv": 72,
    "laptop": 73,
    "mouse": 74,
    "remote": 75,
    "keyboard": 76,
    "cell phone": 77,
    "microwave": 78,
    "oven": 79,
    "toaster": 80,
    "sink": 81,
    "refrigerator": 82,
    "book": 84,
    "clock": 85,
    "vase": 86,
    "scissors": 87,
    "teddy bear": 88,
    "hair drier": 89,
    "toothbrush": 90,
}


def correct_json(input_file, output_file=None):
    with open(input_file, "r") as f:
        contents = json.load(f)

    for content in contents:
        # print(content)
        content["category_id"] = COCO_CATEGORIES[CLASS_NAMES[content["category_id"]]]
    if output_file == None:
        output_file = input_file.split(".json")[0] + "_correct.json"

    with open(output_file, "w") as f:
        json.dump(contents, f)
    return output_file


def COCOMAP(gt_file, result_file, correct_result=True):
    if correct_result:
        result_file = correct_json(result_file)

    cocoGt = COCO(gt_file)
    cocoDt = cocoGt.loadRes(result_file)
    imgIds = sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

ANCHORS = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
STRIDES = [8, 16, 32]
YUV_INPUTS = False

# CONF_THRESH = 0.01
# IOU_THRESH = 0.45
# STRIDES = [8, 16, 32]
# ANCHORS = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
# NUM_OUTPUTS = 85
# INPUT_SHAPE = [640, 640]


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
):
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=prediction.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # print(multi_label)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def letterbox_yolov5(
    im: np.ndarray,
    new_shape: tuple[int] = (640, 640),
    color: tuple[int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def pre_processing(image_raw: np.ndarray, img_shape: Tuple[int], do_transpose: bool, do_dequant: bool):

    img = letterbox_yolov5(image_raw, img_shape, stride=32, auto=False)[0]
    img = img[:, :, ::-1]  # BGR to RGB

    if do_transpose:
        img = img.transpose((2, 0, 1))  # (h, w, c) to (c, h, w)

    if do_dequant:
        img = img / 255
        img = img.astype(np.float32)

    origin_shape = image_raw.shape[:2]
    input_shape = img.shape[-2:] if do_transpose else img.shape[:2]

    return img, origin_shape, input_shape




def make_grid(nx=20, ny=20):
    """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""

    y, x = torch.arange(ny, dtype=torch.float), torch.arange(nx, dtype=torch.float)
    yv, xv = torch.meshgrid(y, x)  # torch>=0.7 compatibility
    grid = torch.stack((xv, yv), 2)  # add grid offset, i.e. y = 2.0 * x - 0.5
    return grid


def gen_proposals(outputs: list[torch.Tensor], do_transpose: bool = True, nc=80) -> torch.Tensor:

    device = "cpu"
    anchors = torch.Tensor(ANCHORS)
    anchor_grid = anchors.clone().view(3, 1, -1, 1, 1, 2)
    anchor_grid = anchor_grid.to(device)

    grid = [torch.zeros(1).to(device)] * 3
    z = []
    no = nc + 5
    for i, pred_map in enumerate(outputs):
        if isinstance(pred_map, np.ndarray):
            pred_map=torch.from_numpy(pred_map).to(device)
        pred_map = torch.sigmoid(pred_map)  # maybe not need
        if do_transpose:
            (
                bs,
                _,
                ny,
                nx,
            ) = pred_map.shape
            y = pred_map.view(bs, 3, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        else:
            bs, _, ny, nx, _ = pred_map.shape
            y = pred_map.contiguous()
        grid = make_grid(nx, ny).to(device)
        y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid) * STRIDES[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, no))
    out = torch.cat(z, 1)

    return out


def draw_bbox(image, bboxes, classes=None, show_label=True, threshold=0.4):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = 80
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / 5, 1.0, 1.0) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        if score < threshold:
            continue
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        print(f"bbox:{coor}, scale:{score}, label: {class_ind} class: {class_ind}")
        if show_label:
            bbox_mess = "%s: %.2f" % (class_ind, score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)

            cv2.putText(
                image,
                bbox_mess,
                (c1[0], c1[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                (0, 0, 0),
                bbox_thick // 2,
                lineType=cv2.LINE_AA,
            )

    return image


def post_processing(
    outputs: list[torch.Tensor],
    origin_shape: list[int],
    input_shape: list[int],
    conf_thresh=0.001,
    iou_thresh=0.6,
    do_transpose=True,
    nc=80,
):

    proposals = gen_proposals(outputs, do_transpose, nc=nc)
    pred = non_max_suppression(proposals, conf_thresh, iou_thresh, labels=[], multi_label=True, agnostic=False)

    for det in pred:
        det[:, :4] = scale_coords(input_shape, det[:, :4], origin_shape)

    return det

class SampleDataset(Dataset):
    """_summary_
    FIXME: hardcode for transform; need replace
    """

    def __init__(self, image_folder: str, input_shape=[704, 1280]):

        image_folder = Path(image_folder)
        self.image_paths = [
            item for item in image_folder.rglob("*") if item.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]

        self.input_shape = input_shape  # assume input shape has batch
        self.len = len(self.image_paths)

        # self.mean = [float(m) / 255 for m in acc_config.pre_config.mean]
        # self.std = [float(s) / 255 for s in acc_config.pre_config.std]
        self.mean = [0, 0, 0]
        self.std = [255, 255, 255]

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        name = self.image_paths[index].name
        image_data = cv2.imread(str(self.image_paths[index]))

        image_data, origin_shape, input_shape = pre_processing(image_data, self.input_shape, True, True)

        image_data = image_data[np.newaxis,...]
        return image_data, name, origin_shape



def run(
    model_path: str,
    dataset_path: str,
    save_results_path: str = None,
    conf=0.001,
    iou=0.6,
    nc=80,
):
    if model_path.endswith(".axmodel"):
        import axengine as ort
    session = ort.InferenceSession(model_path)
    input_shape = None
    input_name = None
    for inp_meta in session.get_inputs():
        input_shape = inp_meta.shape[2:]
        input_name = inp_meta.name
    dataset_path = Path(dataset_path)
    val_path = dataset_path / "images" / "val2017"
    gt_path = dataset_path / "annotations" / "instances_val2017.json"
    assert val_path.exists()
    assert gt_path.exists()
    dataset = SampleDataset(val_path, input_shape)
    print(f"sample contains {dataset.len} data")
    res_dict = []
    for data, name, origin_shape in tqdm(dataset):
        results = session.run(None, {input_name:data})
        det = post_processing(
            results, origin_shape, input_shape, conf_thresh=conf, iou_thresh=iou, do_transpose=False, nc=nc
        )
        for box in det:
            item = {
                "image_id": int(name.split(".")[0]),
                "category_id": int(box[-1]),
                "bbox": [box[0].item(), box[1].item(), (box[2] - box[0]).item(), (box[3] - box[1]).item()],
                "score": float(box[4]),
            }
            res_dict.append(item)
    with open(save_results_path, "w") as f:
        json.dump(res_dict, f)

    COCOMAP(gt_path, save_results_path, correct_result=True)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="benchmark exsample")
    parser.add_argument("--model", type=str, help="onnx or compiled.axmodel path")
    parser.add_argument("--dataset", type=str, help="dataset path")
    parser.add_argument("--save_results", type=str, default="cache/results.json", help="save results path")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="iou threshold")
    parser.add_argument("--nc", type=float, default=80, help="number of classes")
    args = parser.parse_args()
    return args


# python val_on_board.py --model compiled.axmodel --dataset ../dataset/coco/ 
if __name__ == "__main__":

    args = parse_args()

    print(f"Command: {' '.join(sys.argv)}")
    print("Parameters:")
    print(f"  --model: {args.model}")
    print(f"  --dataset_path: {args.dataset}")
    print(f"  --save_results_path: {args.save_results}")
    print(f"  --conf: {args.conf}")
    print(f"  --iou: {args.iou}")
    print(f"  --num_class: {args.nc}")
    print()

    run(args.model, args.dataset, args.save_results, args.conf, args.iou, args.nc)


仓库用于做yolo系列的QAT训练；

|model|map@50-95|map@50|
|--|--|--|
|yolov5s.pt|0.374|0.572|
|yolov5s_8w8f_qdq.onnx|0.367|0.567|

## 环境安装

```
pip install -r requirements.txt
```
**要求 pytorch==2.6;**

## 数据集路径修改

修改 coco.yaml 中的数据集路径;

## 训练
```python qat_base_ptq.py --data coco.yaml --epochs 50 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 96 --device 3 --lsq --hyp hyp.no-augmentation.yaml --save-period 10```

## 测试
qdq 模型测试: ```python val.py --weights yolov5s_8w8f_qdq.onnx --data coco.yaml --training-onnx```


```bash
Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 [02:00<00:00, 41.47it/s]
  all       5000      36335       0.67      0.518      0.562      0.365

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.396
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.371
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.712
```

浮点模型测试：```python val.py --weights yolov5s.pt --data coco.yaml```

```bash
Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 [00:35<00:00,  4.43it/s]
  all       5000      36335      0.672      0.519      0.566      0.371

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.572
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722
```
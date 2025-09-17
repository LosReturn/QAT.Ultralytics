
命令

## 最佳实践
python qat_base_ptq.py --data coco.yaml --epochs 50 --weights yolov5s.pt --cfg yolov5s.yaml  --batch-size 96 --device 3 --lsq --hyp hyp.no-augmentation.yaml

## 测试
python val.py --weights yolov5s_qat_slim.onnx --data coco.yaml --training-onnx

改动:
- ax_quantizer.py 中 PTQ里加了FakeQuantize; 可以控制observer开启与否;
- val.py / common.py 适配trainging export模型



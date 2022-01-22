## YOLOV4-Tiny：You Only Look Once-Tiny目标检测模型在Keras当中的实现


## 1.训练步骤
具体训练步骤可以参考<https://github.com/bubbliiiing/yolov4-tiny-keras>
nets/CSPdarknet53_tiny.py中的resblock_body有一个切片操作Vitis-AI compiler不支持，因此用1x1卷积代替
![image](https://user-images.githubusercontent.com/71107056/150624788-ed027f3b-4b67-45c5-9ade-a6db1dcad58f.png)

## 2.将训练好的h5冻结为pb格式
修改keras2pb.py里面的配置参数

a.将input_model设置为训练好的h5权重文件路径

b.将output_model设置为转换后的pb文件路径

c.在keras2pb.py里面第59行设置你自己数据集的num_class

修改好后运行就会生成pb文件
## Vitis-AI量化步骤
## Vitis-AI编译步骤

## Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch

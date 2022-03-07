## YOLOV4-Tiny：You Only Look Once-Tiny目标检测模型在Keras当中的实现


## 1.训练步骤
具体训练步骤可以参考<https://github.com/bubbliiiing/yolov4-tiny-keras>

nets/CSPdarknet53_tiny.py中的resblock_body有一个切片操作Vitis-AI compiler不支持，因此用1x1卷积代替

![image](https://user-images.githubusercontent.com/71107056/150624788-ed027f3b-4b67-45c5-9ade-a6db1dcad58f.png)

## 2.将训练好的h5冻结为pb格式
修改keras2pb.py里面的配置参数


修改好后运行就会生成pb文件
## Vitis-AI量化步骤

进入Vitis-AI docker环境，激活vitis-ai-tensorflow

vai_q_tensorflow quantize \
     --input_frozen_graph ~.pb \
     --input_nodes input_1 \
     --input_shapes ?,416,416,3 \
     --output_dir ./quantize14 \
     --output_nodes conv2d_21/BiasAdd,conv2d_24/BiasAdd \
     --input_fn input_fn.calib_input \
     --calib_iter 25
     
将input_fn.py放进量化的环境中

## Vitis-AI编译步骤

vai_c_tensorflow \
    --f ./quantize14/quantize_eval_model.pb \
    --a   kv260arch_B3136.json \
    --output_dir compile14 \
    --n   mask_detection \
    --options '{"input_shape": "1,224,224,3"}'


## Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/bubbliiiing/yolov4-tiny-keras

# image_processing

## 空白照片识别

### 数据获取

获取对应的标注数据，把图片中出现人和动物的照片作为1，将图片中没出现人和动物的定义0
运行如下代码能够获取正负例数据文件
``` shell
python ./image_classifier/get_classifier_data.py -m /hpc_input_export/人为活动.json -a /hpc_input_export/物种图片.json -n /hpc_input_export/空白图片.json
```

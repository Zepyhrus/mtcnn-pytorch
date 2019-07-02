# mtcnn-pytorch

![](./img/res1.jpg)

![](./img/res2.jpg)

# 介绍

mtcnn用pytorch实现代码（从入门到工程化）

mtcnn实现了由粗到精的人脸检测框架，具有承上启下的意义。

mtcnn分为三个网络，网络模型都很小。原版论文里面的多任务有人脸检测、人脸目标框回归及人脸关键点回归。

这里做了简化，只做了人脸检测和人脸目标框回归。

在实现过程中，参考了：[MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)和[MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)




# 安装

1. wider face数据集，下载后放置到： `~/dataset/WIDER_FACE`目录下，该目录下的目录为：
```
wider_face_split
WIDER_train
WIDER_val
```
2. python3(anaconda)
3. pytorchv1.0.0, lmdb, opencv, numpy, pylab
4. (c++) cmake, opencv 

# 测试
python模型文件在 `${REPO}/scripts/models`目录下

c++ 模型文件在 `${REPO}/cpp`目录下

1， python预测

修改MTCNN.py 中的图片路径，即可进行测试。（SHOW_FIGURE=True，将会显示预测结果）

2, C++预测

修改 main.cpp中的图片路径及 pt模型文件的路径，即可进行测试




# 训练
cd到 scripts 目录下：
1. 制作pnet 样本：
`python data_gen/GeneratePNetData.py `

2. 训练pnet：
`~/anaconda3/bin/python train_pnet.py`

3. 制作 rnet 样本：
编辑GenerateRONetData.py中的net_type及 Pnet的网络权重路径，将rnet和onet置为None。

`~/anaconda3/bin/python data_gen/GenerateRONetData.py `

4. 训练 rnet 

`~/anaconda3/bin/python train_rnet.py`

5, 制作 onet 样本：

编辑GenerateRONetData.py中的net_type及 Pnet和rnet的网络权重路径，将onet置为None。

`~/anaconda3/bin/python data_gen/GenerateRONetData.py `

6， 训练 onet 

`~/anaconda3/bin/python train_onet.py`

# Validation
1. model: `20190624`, min_face_size:20, threshold: [0.6, 0.7, 0.7], 71 seconds:
Detect  Missing False   All
2488    78      153     2412
Precision: 0.9385
Recall: 0.9677
F1 score: 0.9529

2. model: `20190624`, min_face_size: 20, threshold: [0.6, 0.8, 0.9], 55 seconds:
Detect  Missing False   All
2373    124     84      2412
Precision: 0.9646
Recall: 0.9486
F1 score: 0.9565

2. model: `20181218`, mfs: 20, threshold： [0.6, 0.8, 0.9], 58s:
Detect  Missing False   All
2406    129     123     2412
Precision: 0.9489
Recall: 0.9465
F1 score: 0.9477


3. model: `20190624`, min_face_size: 40, threshold: [0.6, 0.8, 0.9], 25 seconds:
Detect  Missing False   All
2052    420     60      2412
Precision: 0.9708
Recall: 0.8259
F1 score: 0.8925

3. model: `20181218`, mfs: 40, threshold： [0.6, 0.8, 0.9], 58s:
Detect  Missing False   All
2057    435     80      2412
Precision: 0.9611
Recall: 0.8197
F1 score: 0.8848

4. model: `20190624`, min_face_size: 40, threshold: [0.6, 0.95, 0.9], 25 seconds:
Detect  Missing False   All
1968    475     31      2412
Precision: 0.9842
Recall: 0.8031
F1 score: 0.8845






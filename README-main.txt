该项目可支持分类，检测，分割三任务

环境配置请参照mmdetection3.0官方配置
使用前请熟悉mmdetection3.0官方文档

使用说明：
1.数据集格式：
目前只支持coco-json格式数据标注格式，相对原版coco.json标注只有annotaitons部分新增一些标注参数
以以下数据集为例
	person,dog,car(bus,suv),eat(food,bowl)
在该数据集中，person,dog,car,eat为四大类数据，其中属于car类的图片需要进行目标检测(bbox类别为bus,suv)，属于eat类图片需要进行实例分割（分割类别为food,bowl）
因此我们为纯分类数据虚构bbox与mask(均为全图)，目标检测类数据虚构mask(同bbox)，此外我们对每个annotaiton新增class_id与shape_type字段，
其中class_id以大类为索引，即person:1，dog:2，bus,suv:3，food,bowl:4，shape_type用来区分分类，检测，分割
对于原有的category_id字段，纯分类数据与class_id相同，检测与分割数据则只排下分小类即 bus，suv的category_id为3，4，food,bowl为5，6
以上面数据集为例
person的格式应该是：
{
      "iscrowd": 0,
      "image_id": 1,
      "shape_type": "null",
      "bbox": [
        1,
        1,
        638,
        425
      ],
      "segmentation": [
        [
          1,
          1,
          1,
          426,
          639,
          426,
          639,
          1
        ]
      ],
      "class_id": 1,
      "category_id": 1,
      "id": 1,
      "area": 271150
    }
dog的格式应该是：
{
      "iscrowd": 0,
      "image_id": 2,
      "shape_type": "null",
      "bbox": [
        1,
        1,
        638,
        425
      ],
      "segmentation": [
        [
          1,
          1,
          1,
          426,
          639,
          426,
          639,
          1
        ]
      ],
      "class_id": 2,
      "category_id": 2,
      "id": 2,
      "area": 271150
    }
bus的格式应该是：
{
      "iscrowd": 0,
      "image_id": 3,
      "shape_type": "rectangle",
      "bbox": [
        464,
        220,
        87,
        75
      ],
      "segmentation": [
        [
          464.811320754717,
          295.01886792452825,
          464.811320754717,
          220.49056603773585,
          551.132075471698,
          220.49056603773585,
          551.132075471698,
          295.01886792452825
        ]
      ],
      "class_id": 3,
      "category_id": 3,
      "id": 3,
      "area": 6525
    }
suv的格式应该是：
{
      "iscrowd": 0,
      "image_id": 3,
      "shape_type": "rectangle",
      "bbox": [
        464,
        220,
        87,
        75
      ],
      "segmentation": [
        [
          464.811320754717,
          295.01886792452825,
          464.811320754717,
          220.49056603773585,
          551.132075471698,
          220.49056603773585,
          551.132075471698,
          295.01886792452825
        ]
      ],
      "class_id": 3,
      "category_id": 4,
      "id": 4,
      "area": 6525
    }
food的格式应该是：
{
      "iscrowd": 0,
      "image_id": 4,
      "shape_type": "polygon",
      "bbox": [
       273,
        0,
        234,
        96
      ],
      "segmentation": [
        [
          .......
        ]
      ],
      "class_id": 4,
      "category_id": 5,
      "id": 5,
      "area": ....
    }
bowl的格式应该是：
{
      "iscrowd": 0,
      "image_id": 4,
      "shape_type": "polygon",
      "bbox": [
       273,
        0,
        234,
        96
      ],
      "segmentation": [
        [
          .......
        ]
      ],
      "class_id": 4,
      "category_id": 6,
      "id": 5,
      "area": ....
    }
那么coco.json中的categories为person,dog,bus,suv,food,bowl
2.相关配置文件
数据集配置：

mmdet/dataset/coco.py
classes元组与categories相同
base_classes : person,dog,car,eat
detection_classes ：car   #因为是元组请末尾加上逗号
segmentation_classes：eat

config/base/dataset/coco_instance.py
需修改位置与官方相同

模型配置文件
configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py
需修改类别数量，分类头类别数量与base_classes数量相同，检测和分割头与classes数量相同
目前仅支持mask_rcnn架构

3预测
使用根目录下的infer-maser.py进行预测，参数在代码中已做解释

4其他任务

对于单任务
单分类：需仅包含分类数据，coco_instace.py的metric改为空
单检测、分割：configs\swin\faster-rcnn_swin-t-p4-w7_fpn_1x_coco.py +-分割头

双任务：根据是否有分割+-分割头，其余与三任务同理



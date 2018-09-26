### Deeplab V3训练自己的数据集 
1. 在deeplab/datasets目录下递归创建***pascal_voc_seg /VOCdevkit/VOC2012***目录，该目录应该包含如下内容：

      - **JPEGImages** 目录，
      - **SegmentationClassRaw** 目录，
      - **ImageSets/Segmentation **目录，

2. 修改datasets/download_and_convert_voc2012.sh文件

3. 修改datasets/segmentation_dataset.py文件，_PASCAL_VOC_SEG_INFORMATION 中对应的值。

4. 修改trian.py文件，修改内容如下：

      flags.DEFINE_boolean('initialize_last_layer', False,

      ​                     'Initialize the last layer.')

      flags.DEFINE_multi_integer('train_crop_size', [336, 336],

      ​                           'Image crop size [height, width] during training.')

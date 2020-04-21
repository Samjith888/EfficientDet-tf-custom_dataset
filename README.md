# EfficientDet-tf-custom_dataset
EfficientDet tensorflow object detection implementation with custom dataset

This is based on the official implentation of EfficientDet by [google](https://github.com/google/automl/tree/master/efficientdet). Have a look at their [paper](https://arxiv.org/abs/1911.09070) for more theoritical knowledge. 
Thank you for the great work.

## Table of Contents
1. [Prerequisites](#Prerequisites)
2. [Inference](#Inference)
3. [Results](#Results)
4. [Training](#Training)
    1. [Preprocess](#Preprocess)
    2. [Training](#Training) 


## Prerequisites
1) Tensorflow 

recommedned : create a conda environment with tensorflow 2 by using following command 

```bash
conda create -n tf2 tensorflow-gpu
conda activate tf2
```
Note : Make sure that you are in the above conda environment while processing every operations in this. 

## Inference 

Following command can be used for inference.

```bash
  python model_inspect.py --runmode=infer --model_name=$MODEL \
  --input_image_size=1920x1280 --max_boxes_to_draw=100   --min_score_thresh=0.2 \
  --ckpt_path='trained weight with path' --input_image=img.png --num_classes  1 --output_image_dir='path to output directory'
```

## Results
Will update on this soon


## Training

#### Preprocess

The custom dataset have to be converted into tfrecords which will be acceptable by the tensorflow training pipeline.

- Kitti dataset to tfrecord 

1) Kitti dataset can be converted into tfrecord directly by using the following command( script is from [tensorflow repo](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py))

```bash
 python dataset_convert/create_kitti2tfrecord.py \
        --data_dir='path of kitti_dataset folder'
        --output_path='path to store tfrecords'
        --label_map_path = 'path to label_map.pbtxt'
 ```   
 Note : install tf object detection api by  `pip install tensorflow-object-detection-api`
 Below is the folder structre of kitti format used by `create_kitti2tfrecord.py`
 ``` 
 kitti_dataset/ 
        ├─images/
        |       ├─ img1.jpg
        |       ├─ img2.jpg
        ├─labels/ 
                ├─ img1.txt
                ├─ img2.txt 
 ```
 label_map.pbtxt file format for one class 
 
 ```
 item {
  id: 1
  name: 'class_name'
}
```

Note : There are also several scripts available to convert from kitti to tfrecords. (eg: [`cvdata_convert` command from cvdata pip package](https://github.com/monocongo/cvdata#annotation-format-conversion))

2) Kitti cab be converted into Pascal voc by using the following script 
```bash
 python dataset_convert/kitti2voc.py 
 ```   

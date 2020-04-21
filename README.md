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

Tfrecords have to be generated from custom dataset, which will be acceptable by the tensorflow training pipeline.
Lets make a folder `tfrecord` 

```bash
cd EfficientDet-tf-custom_dataset
mkdir tfrecord
```
**a) Kitti dataset to tfrecord** 

1) Kitti dataset can be converted into tfrecord directly by using the following command( script is from [tensorflow repo](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_kitti_tf_record.py))

```bash
 python kiti_convert/create_kitti2tfrecord.py \
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
 python kitti_convert/kitti2voc.py -ipi 'path to images' -ipl 'path to kitti labels'
 ```   
 Convert kitti to PascalVOC datastructure for train and validation seperately if needed. Hence the `dataset/create_pascal_tfrecord.py` script to generate tfrecords from PascalVOC whihc is mentioned below.
  
**b) PascalVOC to tfrecord** 

Open the create_tfrecord.py file in the dataset folder and modify the line 56 with custom class. Modify: pascal_label_map_dict = {'back_ground': 0, 'classname': 1}. 

create `train.txt` file by using following command 
```bash
cd VOCdevkit/VOC2012 && ls -1 JPEGImages | cut -d. -f1 > train.txt && cd -
$ head VOCdevkit/VOC2012/ImageSets/Main/train.txt
 ```
The above command will generate a `train.txt` file under `VOCdevkit/VOC2012/ImageSets/Main` directory.

```bash
PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
    --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal
 ```
 Note : The tf record can be alos genreated for train and validation if needed.

#### Training

* Download backbone using follwoing command 

```bash
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz
tar xf efficientnet-b0.tar.gz 
```
* Start the training using following command 

```bash
python main.py --mode=train_and_eval \
    --training_file_pattern=tfrecord/pascal*.tfrecord \
    --validation_file_pattern=tfrecord/pascal*.tfrecord \
    --val_json_file=tfrecord/json_pascal.json \
    --model_name=efficientdet-d0 \
    --model_dir=/tmp/efficientdet-d0-scratch  \
    --backbone_ckpt=efficientnet-b0  \
    --train_batch_size=8 \
    --eval_batch_size=8 --eval_samples=512 \
    --num_examples_per_epoch=5717 --num_epochs=1  \
    --hparams="use_bfloat16=false,num_classes=20,moving_average_decay=0" \
    --use_tpu=False
```




![1](https://user-images.githubusercontent.com/39676803/79858451-115cbe80-83ed-11ea-8155-3396e2283f43.png)
![2](https://user-images.githubusercontent.com/39676803/79864874-d1e79f80-83f7-11ea-941c-81c6a8ee69f1.png)


Note : For more features and information , please visit the official [EfficientDet-tf github repo](https://github.com/google/automl/tree/master/efficientdet)

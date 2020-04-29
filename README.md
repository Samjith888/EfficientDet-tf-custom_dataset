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
5. [Common_issues](#Issues)

## Prerequisites
- lxml
- Cython
- matplotlib
- pycocotools (pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI)
- Tensorflow 

- recommedned : create a conda environment with tensorflow 2 by using following command 

  ```bash
  conda create -n tf2 tensorflow-gpu
  conda activate tf2
  ```
  Note : Make sure that you are in the above conda environment while processing every operations in this. 

## Inference 
- Open the inference.py file and modify the [line](https://github.com/Samjith888/EfficientDet-tf-custom_dataset/blob/25bba2341d977ba18f7de70daabc771413de7413/inference.py#L39) with your custom class and mapping id same as mentioned while generating tfrecords.
- Following command is used for inference.

  ```bash
    python model_inspect.py --runmode=infer --model_name=$MODEL \
    --input_image_size=1920x1280 --max_boxes_to_draw=100   --min_score_thresh=0.2 \
    --ckpt_path='trained weight with path' --input_image=img.png --num_classes  1 --output_image_dir='path to output directory'
  ```

## Results
Will update on this soon


## Training

#### Preprocess

- Tfrecords have to be generated from custom dataset, which will be acceptable by the tensorflow training pipeline.
- Lets make a folder `tfrecord` 

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
- Below is the folder structre of kitti format used by `create_kitti2tfrecord.py`
    ``` 
    kitti_dataset/ 
          ├─images/
          |       ├─ img1.jpg
          |       ├─ img2.jpg
          ├─labels/ 
                  ├─ img1.txt
                  ├─ img2.txt 
    ```
- label_map.pbtxt file format for one class 
 
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
    The `dataset/create_pascal_tfrecord.py` script to generate tfrecords from PascalVOC whihc is mentioned below.
  
**b) PascalVOC to tfrecord** 

- Open the create_tfrecord.py file in the dataset folder and modify [this line](https://github.com/Samjith888/EfficientDet-tf-custom_dataset/blob/25bba2341d977ba18f7de70daabc771413de7413/dataset/create_pascal_tfrecord.py#L56) with custom classes.

- create `train.txt` file by using following command 

    ```bash
    cd VOCdevkit/VOC2012 && ls -1 JPEGImages | cut -d. -f1 > train.txt && cd -
    head VOCdevkit/VOC2012/train.txt
    ```
    The above command will generate a `train.txt` file under `VOCdevkit/VOC2012` , move it into `ImageSets/Main` directory.

- Run the following command and generate the tfrecords from PascalVOC.

    ```bash
    PYTHONPATH=".:$PYTHONPATH"  python dataset/create_pascal_tfrecord.py  \
        --data_dir=VOCdevkit --year=VOC2012  --output_path=tfrecord/pascal
    ```
    Note : The tf record can be alos genreated for train and validation if needed.

#### Training

* Download backbone using follwoing command into the `EfficientDet-tf-custom_dataset` folder.

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



## Common issues

* TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.  
solution : `pip install numpy==1.17.0` (downgrade numpy from 1.18 to 1.17)

* tensorflow.python.framework.errors_impl.InvalidArgumentError: assertion failed
  [[{{node parser/Assert/Assert}}]]
  [[IteratorGetNext]]
  
  solution : change [MAX_NUM_INSTANCES](https://github.com/Samjith888/EfficientDet-tf-custom_dataset/blob/master/dataloader.py#L27) with maximum number of instances in image of the custom dataset
  

Note : For more features and information , please visit the official [EfficientDet-tf github repo](https://github.com/google/automl/tree/master/efficientdet)

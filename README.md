# Coral TPU training scripts for 73B2 Kitchen dataset

## Preparation

### Installation

```bash
# install gdown
sudo pip install gdown

# install edgetpu_compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```

### Build Docker image

```bash
git clone https://github.com/knorth55/73b2_kitchen_edgetpu_object_detection.git
cd 73b2_kitchen_edgetpu_object_detection
mkdir learn/
docker build docker/ --tag 73b2-kitchen-object-detection
```

### Download and prepare 73b2 kitchen dataset

```bash
cd learn/
gdown https://drive.google.com/uc?id=1iBSxX7I0nFDJfYNpFEb1caSQ0nl4EVUa
tar zxvf kitchen_dataset.tgz
cd ..
```

### Download and prepare your own dataset

First you need VOC format annotation.
For the annotation, please read [here](https://jsk-docs.readthedocs.io/projects/jsk_recognition/en/latest/deep_learning_with_image_dataset/annotate_images_with_labelme.html).

Download and rename, setup your dataset directory like below.

```bash
learn/kitchen_dataset
|-- train  # train dataset
|   |-- JPEGImages
|   |-- SegmentationClass
|   |-- SegmentationClassPNG
|   |-- SegmentationClassVisualization
|   |-- SegmentationObject
|   |-- SegmentationObjectPNG
|   |-- SegmentationObjectVisualization
|   `-- class_names.txt
`-- test   # test dataset
    |-- JPEGImages
    |-- SegmentationClass
    |-- SegmentationClassPNG
    |-- SegmentationClassVisualization
    |-- SegmentationObject
    |-- SegmentationObjectPNG
    |-- SegmentationObjectVisualization
    `-- class_names.txt
```

## Training

### Fine tuning in docker container

```bash
bash run.bash
```

Inside docker

```bash
cd 73b2_kitchen_scripts/
# prepare dataset
./prepare_checkpoint_and_dataset.sh --train_whole_model false --network_type mobilenet_v2_ssd
# retraining on GPU 0
CUDA_VISIBLE_DEVICES=0 ./retrain_detection_model.sh --num_training_steps 500 --num_eval_steps 100 

# change to edgetpu model
./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 500
```

### Whole retraining in docker container

```bash
bash run.bash
```

Inside docker

```
cd 73b2_kitchen_scripts/
# prepare dataset
./prepare_checkpoint_and_dataset.sh --train_whole_model true --network_type mobilenet_v2_ssd
# retraining on GPU 0
CUDA_VISIBLE_DEVICES=0 ./retrain_detection_model.sh --num_training_steps 50000 --num_eval_steps 2000 

# change to edgetpu model
./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 50000
```

### Run Tensorboard for visualization (Optional)

```
docker exec -it 73b2-kitchen-edgetpu /bin/bash
```

Inside docker

```
tensorboard --logdir=./73b2_kitchen_learn/train
```

You can see Tensorboard in localhost:6006.

## Compile

### Compile the model to Edge TPU

Compile and get `output_tflite_graph_edgetpu.tflite` model file!

```
cd 73b2_kitchen_edgetpu_object_detection/learn
sudo chmod 777 models
cd models
sudo chmod 755 *
edgetpu_compiler output_tflite_graph.tflite
```

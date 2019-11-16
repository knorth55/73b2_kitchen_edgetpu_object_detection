# Coral TPU training scripts for 73B2 Kitchen dataset

## Preparation

```
git clone https://github.com/knorth55/73b2_kitchen_edgetpu_object_detection.git
cd 73b2_kitchen_edgetpu_object_detection
mkdir learn/
docker build docker/ --tag 73b2-kitchen-object-detection
```

## Training in docker

```
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

## Run Tensorboard

```
docker exec -it 73b2-kitchen-edgetpu /bin/bash
```

Inside docker

```
tensorboard --logdir=./73b2_kitchen_learn/train
```

You can see Tensorboard in localhost:6006.

## Compile the model to Edge TPU

```
cd 73b2_kitchen_edgetpu_object_detection/learn
sudo chown 777 models
cd models
sudo chmod 755 *
edgetpu_compiler output_tflite_graph.tflite
```

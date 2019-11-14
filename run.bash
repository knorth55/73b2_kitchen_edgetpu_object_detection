THIS_DIR="$PWD"

docker run --name 73b2-kitchen-edgetpu \
--rm -it --privileged -p 6006:6006 \
--mount type=bind,src=${THIS_DIR}/learn,dst=/tensorflow/models/research/73b2_kitchen_learn \
--mount type=bind,src=${THIS_DIR}/scripts,dst=/tensorflow/models/research/73b2_kitchen_scripts \
73b2-kitchen-object-detection

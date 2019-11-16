#!/bin/bash

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Downloads checkpoint and dataset needed for the tutorial.

  --network_type      Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd],
                      mobilenet_v1_ssd by default.
  --train_whole_model Whether or not to train all layers of the model. false
                      by default, in which only the last few layers are trained.
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v1_ssd"
train_whole_model="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --train_whole_model)
      train_whole_model=$2
      shift 2;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

echo "PREPARING checkpoint..."
mkdir -p "${LEARN_DIR}"

ckpt_link="${ckpt_link_map[${network_type}]}"
ckpt_name="${ckpt_name_map[${network_type}]}"
cd "${LEARN_DIR}"
wget -O "${ckpt_name}.tar.gz" "$ckpt_link"
tar zxvf "${ckpt_name}.tar.gz"
rm "${CKPT_DIR}" -rf
mv "${ckpt_name}" "${CKPT_DIR}"

echo "PREPARING dataset..."
cd "${LEARN_DIR}"
gdown ${DATASET_URL}
tar zxvf ${DATASET_TGZ}

echo "CHOSING config file..."
config_filename="${config_filename_map[${network_type}-${train_whole_model}]}"
cd "${OBJ_DET_DIR}"
cp "${OBJ_DET_DIR}/configs/${config_filename}" "${CKPT_DIR}/pipeline.config"

echo "REPLACING variables in config file..."
sed -i "s%CKPT_DIR_TO_CONFIGURE%${CKPT_DIR}%g" "${CKPT_DIR}/pipeline.config"
sed -i "s%DATASET_DIR_TO_CONFIGURE%${DATASET_DIR}%g" "${CKPT_DIR}/pipeline.config"

echo "CONVERTING dataset to TF Record..."
cd "${OBJ_DET_DIR}"
python create_73b2_kitchen_tf_record.py \
    --data_dir="${DATASET_DIR}" \
    --output_dir="${DATASET_DIR}"

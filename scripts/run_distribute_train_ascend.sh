#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# mpirun --allow-run-as-root -n $DEVICE_NUM \
#   python train.py \
#   --device_num=$DEVICE_NUM \
#   --device_target="Ascend" \
#   --TRAIN_BATCH_SIZE=12 \
#   --config_path='./configs/CTW1500_config.yaml' \
#   --TRAIN_MODEL_SAVE_PATH="./ctw_ceshi" \
#   --run_distribute=True > log/ctw_ceshi.log 2>&1 &


# mpirun --allow-run-as-root -n $DEVICE_NUM \
#   python train.py \
#   --device_num=$DEVICE_NUM \
#   --device_target="Ascend" \
#   --TRAIN_BATCH_SIZE=16 \
#   --config_path='./configs/ICDAR2015_config.yaml' \
#   --TRAIN_MODEL_SAVE_PATH="./icdar_ceshi" \
#   --run_distribute=True > log/icdar_ceshi.log 2>&1 &
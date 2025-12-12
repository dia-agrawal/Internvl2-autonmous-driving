#!/usr/bin/env bash
set -x

# Prevent DeepSpeed from attempting to JIT-compile fused ops at runtime.
# This avoids nvcc / compute-arch related build failures on systems
# where compiling extensions is undesirable or unsupported.
export DS_BUILD_OPS=${DS_BUILD_OPS:-0}
# If you want DeepSpeed to build ops for your GPU, set `TORCH_CUDA_ARCH_LIST`,
# e.g. `export TORCH_CUDA_ARCH_LIST="8.6"` (set to your GPU compute capability).

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v3/internvl3_1b_dynamic_res_2nd_finetune_full'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: ${GPUS}
# batch size per gpu: ${PER_DEVICE_BATCH_SIZE}
# gradient accumulation steps: ${GRADIENT_ACC}
# total batch size: ${BATCH_SIZE}
# epoch: 1

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path OpenGVLab/InternVL3-1B \
  --conv_style internvl2_5 \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "internvl_chat/shell/data/drivelm_finetune.json" \
  --overwrite_output_dir True \
  --force_image_size 320 \
  --max_dynamic_patch 4 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version=v2 \
  #--deepspeed="${PWD}/internvl_chat/zero_stage1_config.json" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

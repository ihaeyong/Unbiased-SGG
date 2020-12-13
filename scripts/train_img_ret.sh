#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4,$5,$6

if [ $2 == "imgret" ]; then
    python -m torch.distributed.launch \
           --master_port 10042 \
           --nproc_per_node=$1 \
           tools/image_retrieval_main.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 408900 \
           SOLVER.BASE_LR 0.12 \
           SOLVER.PRINT_GRAD_FREQ 5 \
           SOLVER.VAL_PERIOD 1 \
           SOLVER.CHECKPOINT_PERIOD 1 \
           SOLVER.GRAD_NORM_CLIP 5.0 \
           SOLVER.SCHEDULE.TYPE "WarmupReduceLROnPlateau" \
           SOLVER.SCHEDULE.MAX_DECAY_STEP 5 \
           SOLVER.SCHEDULE.PATIENCE 5 \
           SOLVER.SCHEDULE.FACTOR 0.1 \
           GLOVE_DIR ./datasets/glove \
           OUTPUT_DIR ./checkpoints/motifs-causal_l2_5e-7_warmup_embed512_lr12e-2_b12_patience5
fi

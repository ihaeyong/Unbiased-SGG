#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$2,$3

if [ $1 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10026 \
           --nproc_per_node=2 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH 2 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 50000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-exmp

elif [ $1 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10025 \
           --nproc_per_node=2 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH 2 \
           DTYPE "float16" SOLVER.MAX_ITER 50000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/motif-precls-exmp
fi

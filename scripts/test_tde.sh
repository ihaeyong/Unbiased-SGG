#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$2

if [ $1 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10028 \
           --nproc_per_node=1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  \
           TEST.IMS_PER_BATCH 1 \
           DTYPE "float16" GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/causal-motifs-sgcls-exmp \
           OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-baseline

elif [ $1 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10027 \
           --nproc_per_node=1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
           TEST.IMS_PER_BATCH 1 DTYPE "float16" \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/motif-precls-exmp \
           OUTPUT_DIR ./checkpoints/motif-precls-baseline
fi

#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4,$5,$6

if [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10028 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs  \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/causal-motifs-sgcls-exmp \
           OUTPUT_DIR ./checkpoints/causal-motifs-sgcls-baseline

elif [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10023 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR SGraphPredictor \
           MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 512 \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False \
           MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER 0 \
           MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER 1 \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum_v7 \
           TEST.IMS_PER_BATCH $1 DTYPE "float16" \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/obj_spectrum_gcn_sum_v7_0.7-predcls \
           OUTPUT_DIR ./checkpoints/obj_spectrum_gcn_sum_v7_0.7-predcls
fi

#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4

if [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
	   MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
	   MODEL.ROI_RELATION_HEAD.LOSS.USE_NBDT_LOSS False \
           MODEL.ROI_RELATION_HEAD.LOSS.USE_CLASS_BALANCED_LOSS False \
	   MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS True \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/vctree_embed_v3_skew0.7_ent0.01-sgcls \
           OUTPUT_DIR ./checkpoints/vctree_embed_v3_skew0.7_ent0.01-sgcls

elif [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
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
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum_v3 \
           MODEL.ROI_RELATION_HEAD.RIB_SCALE 2 \
           MODEL.ROI_RELATION_HEAD.RIB_GEOMETRIC True \
           MODEL.ROI_RELATION_HEAD.RIB_EMBEDDING True \
           MODEL.ROI_RELATION_HEAD.RIB_OBJ_CONTEXT False \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/iba0.02_s2_inv_prop0.03_power0.5_sum_v3-predcls \
           OUTPUT_DIR ./checkpoints/iba0.02_s2_inv_prop0.03_power0.5_sum_v3-predcls\
           RIB_FMAP_SAVE True

elif [ $2 == "sgdet" ]; then
    python -m torch.distributed.launch \
           --master_port 10022 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR SGraphPredictor \
           MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 512 \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False \
           MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER 0 \
           MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER 1 \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum_v3 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/obj_spectrum_gcn_sum_v3_0.7-sgdet \
           OUTPUT_DIR ./checkpoints/obj_spectrum_gcn_sum_v3_0.7-sgdet
fi

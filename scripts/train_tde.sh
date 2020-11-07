#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4,$5,$6

if [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10032 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR SGraphPredictor \
           MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 512 \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False \
           MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER 1 \
           MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER 0 \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/obj_spectrum_ctx1_label-decoder-sgcls

elif [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port 10039 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR SGraphPredictor \
           MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 512 \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS False \
           MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER 1 \
           MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER 0 \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/obj_spectrum_ctx1_iid_minus_bias_label-predcls
fi

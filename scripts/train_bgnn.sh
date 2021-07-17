#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4

if [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relBGNN_vg.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE False \
           MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON False \
           MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS True \
           SOLVER.IMS_PER_BATCH 12 \
           SOLVER.BASE_LR 0.01 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 1000 \
           SOLVER.CHECKPOINT_PERIOD 1000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/vg_faster_det.pth \
           OUTPUT_DIR ./checkpoints/bgnn-reweight-updated-obj-v1-skew0.9-ent0.05-predcls

elif [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           SOLVER.IMS_PER_BATCH 12 \
           SOLVER.BASE_LR 0.01 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 1000 \
           SOLVER.CHECKPOINT_PERIOD 1000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/bgnn-sample-reweight-v1-skew0.9-ent0.05-sgcls


elif [ $2 == "sgdet" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           SOLVER.IMS_PER_BATCH 8 \
           SOLVER.BASE_LR 0.01 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/vctree-avg-cls-obj2.2-rel0.9-sgdet

fi
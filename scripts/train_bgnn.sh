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
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT False \
           MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON False \
           MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 90000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/bgnn_embed_lr_target-skew0.9_ent0.19-0.02-predcls

elif [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relBGNN_vg.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE False \
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT False \
           MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON False \
           MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 90000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/bgnn_embed_lr_sample-skew0.9_2.0_0.7_ent0.19-0.06-sgcls

elif [ $2 == "sgdet" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relBGNN_vg.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE False \
           MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT False \
           MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON False \
           MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 90000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/bgnn_embed_lr_sample-skew0.9_2.0_ent0.19-0.06-sgdet
fi

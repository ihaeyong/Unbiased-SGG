#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4

if [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/causal-motifs-sgcls
    
elif [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           SOLVER.IMS_PER_BATCH 12 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/causal-motifs-predcls

elif [ $2 == "sgdet" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_train_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
           MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE none \
           MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum \
           MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           SOLVER.IMS_PER_BATCH 8 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" SOLVER.MAX_ITER 70000 \
           SOLVER.VAL_PERIOD 2000 \
           SOLVER.CHECKPOINT_PERIOD 2000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
           OUTPUT_DIR ./checkpoints/causal-motifs-sgdet

fi
#!/usr/bin/env bash

# Train Motifnet using different orderings
export PYTHONPATH=$HOME/workspaces/unbiased-sg
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=$3,$4

if [ $2 == "predcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
           --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
           MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           SOLVER.IMS_PER_BATCH 12 \
           SOLVER.BASE_LR 0.001 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 40000 \
           SOLVER.VAL_PERIOD 5000 \
           SOLVER.CHECKPOINT_PERIOD 5000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-predcls \
           OUTPUT_DIR ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-predcls

elif [ $2 == "sgcls" ]; then
    python -m torch.distributed.launch \
           --master_port $5 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
	   --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           SOLVER.IMS_PER_BATCH 12 \
           SOLVER.BASE_LR 0.001 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 40000 \
           SOLVER.VAL_PERIOD 5000 \
           SOLVER.CHECKPOINT_PERIOD 5000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-sgcls \
           OUTPUT_DIR ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-sgcls
 

elif [ $2 == "sgdet" ]; then
    python -m torch.distributed.launch \
           --master_port 10022 \
           --nproc_per_node=$1 \
           tools/relation_test_net.py \
	   --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
           MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
           MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
           MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
           MODEL.ROI_RELATION_HEAD.RECT_BOX_EMB True \
           MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
           SOLVER.IMS_PER_BATCH 8 \
           SOLVER.BASE_LR 0.001 \
           TEST.IMS_PER_BATCH $1 \
           DTYPE "float16" \
           SOLVER.MAX_ITER 40000 \
           SOLVER.VAL_PERIOD 5000 \
           SOLVER.CHECKPOINT_PERIOD 5000 \
           GLOVE_DIR ./datasets/glove \
           MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-sgdet \
           OUTPUT_DIR ./checkpoints/sg-transform-embed_v3_lr1e-3_target-skew0.7_0.7_ent0.06-sgdet
fi
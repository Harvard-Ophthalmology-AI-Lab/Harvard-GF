#!/bin/bash
TASK=cls
MODEL_TYPE='efficientnet'
LOSS_TYPE='bce' 
LR=1e-5
NUM_EPOCH=30
BATCH_SIZE=2
MODALITY_TYPE='oct_bscans'
ATTRIBUTE_TYPE='race'
NORMALIZATION_TYPE=fin
PERF_FILE=${MODEL_TYPE}_${NORMALIZATION_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python train_glaucoma_fair_3D_fin.py \
	--data_dir /path/to/EyeFair/dataset \
	--result_dir ./results/3D_crosssectional_${MODALITY_TYPE}_${NORMALIZATION_TYPE}_${ATTRIBUTE_TYPE}/fullysup_${MODEL_TYPE}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE} \
	--model_type ${MODEL_TYPE} \
	--image_size 200 \
	--loss_type ${LOSS_TYPE} \
	--lr ${LR} --weight-decay 0. --momentum 0.1 \
	--batch-size ${BATCH_SIZE} \
	--task ${TASK} \
	--epochs ${NUM_EPOCH} \
	--modality_types ${MODALITY_TYPE} \
	--perf_file ${PERF_FILE} \
	--normalization_type ${NORMALIZATION_TYPE} \
	--attribute_type ${ATTRIBUTE_TYPE} 
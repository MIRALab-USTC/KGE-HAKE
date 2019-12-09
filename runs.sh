#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATASET=$3
GPU_DEVICE=$4
SAVE_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}
MODULUS_WEIGHT=${14}
PHASE_WEIGHT=${15}

if [ $MODE == "train" ]
then

echo "Start Training......"

    if [ $MODEL == "HAKE" ]
    then
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_train \
            --do_valid \
            --do_test \
            --data_path $FULL_DATA_PATH \
            --model $MODEL \
            -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
            -g $GAMMA -a $ALPHA \
            -lr $LEARNING_RATE --max_steps $MAX_STEPS \
            -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
            -mw $MODULUS_WEIGHT -pw $PHASE_WEIGHT
     elif [ $MODEL == "ModE" ]
     then
         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_train \
            --do_valid \
            --do_test \
            --data_path $FULL_DATA_PATH \
            --model $MODEL \
            -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
            -g $GAMMA -a $ALPHA \
            -lr $LEARNING_RATE --max_steps $MAX_STEPS \
            -save $SAVE --test_batch_size $TEST_BATCH_SIZE
     fi

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_valid -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/runs.py --do_test -init $SAVE

else
   echo "Unknown MODE" $MODE
fi
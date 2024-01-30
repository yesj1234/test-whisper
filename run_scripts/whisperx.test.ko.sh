#! /usr/bin/env python3


# get the huggingface token from huggingface.co
# export HF_TOKEN='' 
export WHISPERX_TEST=/home/ubuntu/test_whisper/whisperx.test.py 
export MODEL=large-v2
export LOAD_SCRIPT_BASE=/home/ubuntu/test_whisper/load_scripts/korean
export LANG=ko
export LANGUAGE=korean
export METRIC=cer

#libry 
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_libri.py \
# --dataset_name libri \
# --metric $METRIC \
# --split test.clean

#cv5
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_cv5.py \
# --dataset_name cv5 \
# --metric $METRIC 

#cv9
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_cv9.py \
# --dataset_name cv9 \
# --metric $METRIC 

#ihm
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_ihm.py \
# --dataset_name ihm \
# --metric $METRIC 

#sdm
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_sdm.py \
# --dataset_name sdm \
# --metric $METRIC 

# #ted
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_ted.py \
# --dataset_name ted \
# --metric $METRIC 

#fleurs 
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang ko \
# --language $LANGUAGE \
# --load_script google/fleurs \
# --dataset_name fleurs \
# --metric $METRIC \
# --split test 

# vox
# python3 $WHISPERX_TEST \
# --model $MODEL \
# --lang $LANG \
# --language $LANGUAGE \
# --load_script $LOAD_SCRIPT_BASE/my_vox.py \
# --dataset_name vox \
# --metric $METRIC \
# --split test 

# covost2. 
python3 $WHISPERX_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_covost2.py \
--dataset_name covost2 \
--metric $METRIC \
--split test \
--data_dir /home/ubuntu/covost_ko/ko/ko/

#! /usr/bin/env python3


# get the huggingface token from huggingface.co
# export HF_TOKEN='' 
export WHISPERX_TEST=/home/ubuntu/test_whisper/whisperx.test.py 
export MODEL=large-v2
export LOAD_SCRIPT_BASE=/home/ubuntu/test_whisper/load_scripts/chinese
export LANG=zh
export LANGUAGE=chinese
export METRIC=cer


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

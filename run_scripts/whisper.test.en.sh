#! /usr/bin/env python3


# get the huggingface token from huggingface.co
# export HF_TOKEN='' 
export WHISPER_TEST=/home/ubuntu/test_whisper/whisper.test.py 
export MODEL=openai/whisper-large-v2
export LOAD_SCRIPT_BASE=/home/ubuntu/test_whisper/load_scripts/english
export LANG=en
export LANGUAGE=english
export METRIC=wer
export MAX_SIZE=10000
#libry 
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_libri.py \
--dataset_name libri \
--metric $METRIC \
--split test.clean \
--max_size $MAX_SIZE

# #cv5
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_cv5.py \
--dataset_name cv5 \
--metric $METRIC \
--max_size $MAX_SIZE

# #cv9
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_cv9.py \
--dataset_name cv9 \
--metric $METRIC \
--max_size $MAX_SIZE

#ihm
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_ihm.py \
--dataset_name ihm \
--metric $METRIC \
--max_size $MAX_SIZE

#sdm
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_sdm.py \
--dataset_name sdm \
--metric $METRIC \
--max_size $MAX_SIZE

# #ted
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_ted.py \
--dataset_name ted \
--metric $METRIC \
--max_size $MAX_SIZE

#fleurs 
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang en_us \
--language $LANGUAGE \
--load_script google/fleurs \
--dataset_name fleurs \
--metric $METRIC \
--split test \
--max_size $MAX_SIZE

# vox
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_vox.py \
--dataset_name vox \
--metric $METRIC \
--split test \
--max_size $MAX_SIZE

# covost2. 
python3 -W ignore $WHISPER_TEST \
--model $MODEL \
--lang $LANG \
--language $LANGUAGE \
--load_script $LOAD_SCRIPT_BASE/my_covost2.py \
--dataset_name covost2 \
--metric $METRIC \
--split test \
--data_dir /home/ubuntu/covost_en/en/en/ \
--max_size $MAX_SIZE

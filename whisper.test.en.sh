#! /usr/bin/env python3


# get the huggingface token from huggingface.co
# export HF_TOKEN='' 

#libry 
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_libri.py \
# --dataset_name libri \
# --metric wer \
# --split test.clean

#cv5
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_cv5.py \
# --dataset_name cv5 \
# --metric wer 

#cv9
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_cv9.py \
# --dataset_name cv9 \
# --metric wer 

#ihm
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_ihm.py \
# --dataset_name ihm \
# --metric wer 

#sdm
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_sdm.py \
# --dataset_name sdm \
# --metric wer 

# #ted
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_ted.py \
# --dataset_name ted \
# --metric wer 

#fleurs 
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en_us \
# --language english \
# --load_script google/fleurs \
# --dataset_name fleurs \
# --metric wer \
# --split test 

# vox
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script /home/ubuntu/test_whisper/load_scripts/english/my_vox.py \
# --dataset_name vox \
# --metric wer \
# --split test 

# covost2. # lang config name for covost2 is like source_target for example, en_ja or en_zh-CN. for audio transcription they are all the same. 
python3 whisper.test.py \
--model openai/whisper-large-v2 \
--lang en_ja \
--language english \
--load_script /home/ubuntu/test_whisper/load_scripts/english/my_covost2.py \
--dataset_name covost2 \
--metric wer \
--split test 

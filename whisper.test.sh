#! /usr/bin/env python3


# get the huggingface token from huggingface.co
# export HF_TOKEN='' 

#libry 
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script ./load_scripts/my_libri.py \
# --dataset_name libri \
# --metric wer \
# --split test.clean

#cv5
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script ./load_scripts/my_cv5.py \
# --dataset_name cv5 \
# --metric wer 

#cv9
# python3 whisper.test.py \
# --model openai/whisper-large-v2 \
# --lang en \
# --language english \
# --load_script ./load_scripts/my_cv9.py \
# --dataset_name cv9 \
# --metric wer 

#ihm
python3 whisper.test.py \
--model openai/whisper-large-v2 \
--lang en \
--language english \
--load_script ./load_scripts/my_ihm.py \
--dataset_name ihm \
--metric wer 


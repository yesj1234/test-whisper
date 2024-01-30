#! /usr/bin/env python3 

# covost2. # lang config name for covost2 is like source_target for example, en_ja or en_zh-CN. for audio transcription they are all the same. 
python3 whisper.test.py \
--model openai/whisper-large-v2 \
--lang ko \
--language english \
--load_script /home/ubuntu/test_whisper/load_scripts/korean/my_covost2.py \
--dataset_name covost2 \
--metric cer \
--split test 

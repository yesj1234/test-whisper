#! /usr/bin/bash 

export BASE=/home/ubuntu/test-whisper

python3 score_by_cat.py \
--files ${BASE}/game_predictions.txt,${BASE}/food_predictions.txt,${BASE}/travel_predictions.txt,${BASE}/fashion_predictions.txt,${BASE}/communication_predictions.txt
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from datasets import load_dataset 
import torch
import evaluate
from .dataset_reformer import MyReformer
import re
from tqdm import tqdm
import logging
import sys
import os
logger = logging.getLogger("WhisperLogger")
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s|%(levelname)s|%(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream= sys.stdout
)
# LANGUAGES: en(english), zh(chinese), ko(korean), ja(japanese) 


class MyWhisper:
    def __init__(self, processor, model, *args):
        # init model and processor for conditional generation 
        self.processor = processor 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
    def predict_transcription(self, input_features):
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription
            
        
    def main(self, **kwargs): 
        audio_arraies = kwargs.pop("audio_arraies", [])
        if not audio_arraies:
            raise ValueError("audio arraies are empty")
        predictions = []
        for array in tqdm(audio_arraies, desc="Running Prediction"):
            input_features = self.processor(array, sampling_rate = 16_000, return_tensors="pt").input_features.to(self.device)
            prediction = self.predict_transcription(input_features)
            predictions.append(prediction)
        return predictions

    
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="repo/model_name")
    parser.add_argument("--language", help="english, korean, chinese, japanese")
    parser.add_argument("--load_script", help="repo/dataset_name")
    parser.add_argument("--dataset_name", help="one of [cv5, cv9, ihm, sdm, libri, ted, fleur, vox]")
    parser.add_argument("--lang", help="en ko ja zh-CN")
    parser.add_argument("--metric", help="wer for english / cer for korean, japanese, chinese")
    args = parser.parse_args()
    
    # 1. get the model and processor and initialize MyWhisper 
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    whisperer = MyWhisper(processor=processor, model=model)
    
    # 2. load the dataset
    # multi language support: fleurs, vox, cv9, covost2(later work)
    if args.dataset_name == "cv5": 
        ds = load_dataset(args.load_script, args.lang, token=os.environ["HF_TOKEN"], trust_remote_code=True) 
    if args.dataset_name == "cv9": 
        ds = load_dataset(args.load_script, args.lang, token=os.environ["HF_TOKEN"], trust_remote_code=True) 
    if args.dataset_name == "ihm": 
        ds = load_dataset(args.load_script, 'ihm', trust_remote_code=True) 
    if args.dataset_name == "sdm": 
        ds = load_dataset(args.load_script, 'sdm', trust_remote_code=True) 
    if args.dataset_name == "libri":
        ds = load_dataset(args.load_script, trust_remote_code=True) 
    if args.dataset_name == "ted":
        ds = load_dataset(args.load_script, trust_remote_code=True) 
    if args.dataset_name == "fleur":
        ds = load_dataset(args.load_script, trust_remote_code=True) 
    if args.dataset_name == "vox":
        ds = load_dataset(args.load_script, trust_remote_code=True) 

    logger.info(ds.info.description)
    dataset_reformer = MyReformer()
    ds = dataset_reformer(ds)
    logger.info(ds)
    
    transcriptions=[]
    audio_arraies=[]    

    for sample in ds:
        transcriptions.append(sample["transcription"])
        audio_arraies.append(sample["audio"]["array"])

    # 3. generate predictions and make some post processing. 
    predictions = whisperer.main(transcriptions=transcriptions, audio_arraies=audio_arraies)
    def post_processing(x):
        x = re.sub("[.,?!']", "", x)
        x = x.lower()
        x = x.strip()
        return x 
    
    
    predictions = list(map(lambda x: x[0], predictions))
    predictions = list(map(post_processing, predictions))
    transcriptions = list(map(post_processing, transcriptions))
    # 4. load the metric to compute 
    metric = evaluate.load(args.metric)
    score_file_name = f"./{args.dataset_name.replace('/', '_')}_{args.model.replace('/','_')}_{args.language}scores.txt"
    with open(score_file_name, mode="w", encoding="utf-8") as f:
        for ref, pred in zip(transcriptions, predictions):
            try:
                score = metric.compute(predictions=[pred], references=[ref])
                f.write(f"{pred} :: {ref} :: {round(score, 6)}\n")
            except Exception as e:
                print(e)
                continue

    with open(score_file_name, mode="r", encoding="utf-8") as g, open("model_dataset_scores.txt", mode="a+", encoding="utf-8") as h:
        lines = g.readlines()
        total = 0
        for line in lines:
            ref, pred, score = line.split(" :: ")
            score = float(score[:-2])
            total += score
        logger.info(f"average score: {total / int(len(lines))}")
        h.write(f"{args.model} | {args.dataset_name} | {total/int(len(lines))}\n")

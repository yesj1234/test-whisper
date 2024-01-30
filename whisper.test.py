from transformers import WhisperProcessor, WhisperForConditionalGeneration 
from datasets import load_dataset 
import torch
import evaluate
from utils.dataset_reformer import MyReformer
import re
from tqdm import tqdm
import logging
import sys
import os
from utils.loading import DataLoader
from normalizers.english import EnglishTextNormalizer
from normalizers.basic import BasicTextNormalizer
logger = logging.getLogger("WhisperLogger")
logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s|[%(levelname)s]|[%(name)s]|%(message)s",
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
    parser.add_argument("--model", help="repo/model_name, [openai/whisper-large-v2, openai/whisper-large-v3]", default="openai/whisper-large-v2")
    parser.add_argument("--language", help="english, korean, chinese, japanese")
    parser.add_argument("--load_script", help="repo/dataset_name")
    parser.add_argument("--dataset_name", help="one of [cv5, cv9, ihm, sdm, libri, ted, fleur, vox, covost2]")
    parser.add_argument("--lang", help="usually pick one of the following [en ko ja zh-CN]. \n For fleur pick one of [cmn, en_us, ja_jp, ko_kr, yue]")
    parser.add_argument("--metric", help="wer for english / cer for korean, japanese, chinese")
    parser.add_argument("--split", help="Usually one of [test, validation, train]. Libri[test.other, test.clean]")
    parser.add_argument("--data_dir", help="required for using covost2 since it requires the manual download of the data")
    args = parser.parse_args()
    
    # 1. get the model and processor and initialize MyWhisper 
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    whisperer = MyWhisper(processor=processor, model=model)
    
    # 2. load the dataset
    # multi language support: fleurs, vox, cv9, covost2(later work)
    dataLoader = DataLoader()
    ds = dataLoader.load(dataset_name=args.dataset_name, load_script=args.load_script, lang=args.lang, split=args.split, data_dir=args.data_dir)
    dataset_reformer = MyReformer()
    ds = dataset_reformer(ds, name=args.dataset_name)
    logger.info(ds.info.description)
    logger.info(ds)
    
    transcriptions=[]
    audio_arraies=[]    

    for sample in ds:
        transcriptions.append(sample["transcription"])
        audio_arraies.append(sample["audio"]["array"])

    # 3. generate predictions and make some post processing. 
    predictions = whisperer.main(transcriptions=transcriptions, audio_arraies=audio_arraies)
    if args.language == "english":
        normalizer = EnglishTextNormalizer()
    if args.language in ["japanese", "chinese"]:
        normalizer = BasicTextNormalizer(split_letters=True)
    if args.language == "korean":
        normalizer = BasicTextNormalizer()
    
    def post_processing(x):
        x = normalizer(x)
        return x 
    
    
    predictions = list(map(lambda x: x[0], predictions))
    predictions = list(map(post_processing, predictions))
    transcriptions = list(map(post_processing, transcriptions))
    
    # 4. load the metric to compute and write loggings.
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

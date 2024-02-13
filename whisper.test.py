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
import traceback
import numpy as np
import pandas as pd 
from pprint import pformat
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


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="repo/model_name, [openai/whisper-large-v2, openai/whisper-large-v3]", default="openai/whisper-large-v2")
    parser.add_argument("--language", help="english, korean, chinese, japanese")
    parser.add_argument("--load_script", help="repo/dataset_name")
    parser.add_argument("--dataset_name", help="one of [cv5, cv9, ihm, sdm, libri, ted, fleur, vox, covost2]")
    parser.add_argument("--lang", help="usually pick one of the following [en ko ja zh-CN]. \n For fleur pick one of [cmn, en_us, ja_jp, ko_kr, yue]")
    parser.add_argument("--metric", help="wer for english / cer for korean, japanese, chinese")
    parser.add_argument("--split", help="Usually one of [test, validation, train]. Libri[test.other, test.clean]", default="test")
    parser.add_argument("--data_dir", help="required for using covost2 since it requires the manual download of the data")
    parser.add_argument("--max_size", help="max size of the input datset. Set this for large datasets to reduce the runtime.", type=int, default=100000)
    args = parser.parse_args()
    
    logger.info(f"""
                ***** Simple Summary of the args used *****
{pformat(vars(args))}
                """)
    
    # 1. get the model and processor and metric. 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    metric = evaluate.load(args.metric)
    
    if args.language == "english":
        normalizer = EnglishTextNormalizer()
    if args.language in ["japanese", "chinese"]:
        normalizer = BasicTextNormalizer(split_letters=True)
    if args.language == "korean":
        normalizer = BasicTextNormalizer()
        
    # 2. load the dataset
    # multi language support: fleurs, vox, cv9, covost2(later work)
    dataLoader = DataLoader()
    ds = dataLoader.load(dataset_name=args.dataset_name, load_script=args.load_script, lang=args.lang, split=args.split, data_dir=args.data_dir)
    dataset_reformer = MyReformer()
    ds = dataset_reformer(ds, name=args.dataset_name)
    logger.info(ds.info.description)
    logger.info(ds)
    TOTAL = len(ds) if len(ds) <= args.max_size else args.max_size
    if 'path' not in ds.column_names:    
        DF_KEYS = list(filter(lambda x: x != "audio", ds.column_names)) + ['path', 'model_prediction', 'score']
    else: 
        DF_KEYS = list(filter(lambda x: x != "audio", ds.column_names)) + ['model_prediction', 'score']
        
    df = pd.DataFrame(columns=DF_KEYS)
    
    #3. generate predictions
    count = 0
    for sample in tqdm(ds, total=TOTAL, desc="generating prediction", ascii=" =", leave=True,position=0):
        count += 1
        if count > TOTAL:
            break 
        try:
            input_features = processor(sample['audio']['array'], sampling_rate = 16_000, return_tensors="pt").input_features.to(device)
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            transcription = normalizer(transcription[0])
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            
        if not sample.get('path'):
            sample['path'] = sample['audio']['path']
        del sample['audio']
        sample['model_prediction'] = transcription
        try:
            score = metric.compute(predictions=[transcription], references=[normalizer(sample['transcription'])])
            sample['score'] = round(score, 6)
        except Exception as e:
            sample['score'] = None
            traceback.print_tb(e.__traceback__)
        temp = pd.DataFrame([sample])
        df = pd.concat([df, temp])    
    
    #4 save the dataframe.
    DF_FILENAME = f'{args.model.replace("/", "_")}_{args.dataset_name.replace("/", "_")}.csv'
    df.reset_index(drop=True)
    df.to_csv(DF_FILENAME)

import whisperx
import numpy as np 
import pandas as pd
from datasets import load_dataset, Dataset 
from utils.dataset_reformer import MyReformer
import torch
import evaluate
import re
from tqdm import tqdm
import logging
import sys
import os
import traceback
from pprint import pformat 
from utils.loading import DataLoader
from normalizers.english import EnglishTextNormalizer
from normalizers.basic import BasicTextNormalizer
logger = logging.getLogger("WhisperXLogger")
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
    parser.add_argument("--model", help="large-v2 / large-v3", default="large-v2")
    parser.add_argument("--language", help="english, korean, chinese, japanese")
    parser.add_argument("--load_script", help="repo/dataset_name")
    parser.add_argument("--dataset_name", help="one of [cv5, cv9, ihm, sdm, libri, ted, fleur, vox, covost2]")
    parser.add_argument("--lang", help="usually pick one of the following [en ko ja zh-CN]. \n For fleur pick one of [cmn, en_us, ja_jp, ko_kr, yue]")
    parser.add_argument("--metric", help="wer for english / cer for korean, japanese, chinese")
    parser.add_argument("--split", help="Usually one of [test, validation, train]. Libri[test.other, test.clean]")
    parser.add_argument("--data_dir", help="required for using covost2 since it requires the manual download of the data")
    parser.add_argument("--max_size", help="max size of the dataset to predict", default=20000, type=int)
    args = parser.parse_args()
    
    logger.info(f"""
                ***** Simple Summary of the args used *****
{pformat(vars(args))}
                """)
    # 1. get the model and processor and initialize MyWhisper 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    COMPUTE_TYPE = "float16" if DEVICE=="cuda" else "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
    LANGUAGE = args.lang
    model = whisperx.load_model(args.model, DEVICE, language=LANGUAGE,compute_type=COMPUTE_TYPE)
    metric = evaluate.load(args.metric)

    if args.language == "english":
        normalizer = EnglishTextNormalizer()
    if args.language in ["japanese", "chinese"]:
        normalizer = BasicTextNormalizer(split_letters=True)
    if args.language == "korean":
        normalizer = BasicTextNormalizer()
        
    # 2. load the dataset
    # multi language support: covost2(later work)
    dataLoader = DataLoader()
    ds = dataLoader.load(dataset_name=args.dataset_name, load_script=args.load_script, lang=args.lang, split=args.split, data_dir=args.data_dir)
    dataset_reformer = MyReformer()
    ds = dataset_reformer(ds, name=args.dataset_name)
    assert isinstance(ds, Dataset) == True, "ds is not a instance of class datasets.Dataset"
    logger.info(ds.info.description)
    logger.info(ds)
    TOTAL = len(ds) if len(ds) <= args.max_size else args.max_size
    if 'path' not in ds.column_names:    
        DF_KEYS = list(filter(lambda x: x != "audio", ds.column_names)) + ['path', 'model_prediction', 'score']
    else: 
        DF_KEYS = list(filter(lambda x: x != "audio", ds.column_names)) + ['model_prediction', 'score']
        
    df = pd.DataFrame(columns=DF_KEYS)
    count = 0
    for data in tqdm(ds, total=TOTAL, ascii=" =", leave=True, position=0, desc="Running Prediction"):
        count += 1
        if count > TOTAL:
            break 
        if not data.get('path'):
            data['path'] = data['audio']['path']
        model_prediction = ""
        try:
            arr = data['audio']['array'].astype(np.float32) # convert from float64 to float32 since torch.from_numpy function is used internally in whisperx
            result = model.transcribe(arr, batch_size=BATCH_SIZE)
            model_prediction = normalizer(result['segments'][0]['text'])
            data['model_prediction'] = model_prediction
        except Exception as e:
            data['model_prediction'] = None
            traceback.print_tb(e.__traceback__)
            pass 
        
        try: 
            score = metric.compute(predictions=[model_prediction], references=[normalizer(data['transcription'])])
            data['score'] = round(score, 6)
        except Exception as e:
            data['score'] = None
            traceback.print_tb(e.__traceback__)
            pass
        del data['audio']
        temp = pd.DataFrame([data])
        df = df.reset_index(drop=True)
        df = pd.concat([df, temp], ignore_index=True)
    
    DF_FILENAME = f'{args.model.replace("/", "_")}_{args.dataset_name.replace("/", "_")}.csv'
    df.reset_index(drop=True)
    df.to_csv(DF_FILENAME)

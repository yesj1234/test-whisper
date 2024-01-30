import whisperx 
from datasets import load_dataset, Dataset 
from utils.dataset_reformer import MyReformer
import torch
import evaluate
import re
from tqdm import tqdm
import logging
import sys
import os
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
    args = parser.parse_args()
    
    # 1. get the model and processor and initialize MyWhisper 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    COMPUTE_TYPE = "float16" if DEVICE=="cuda" else "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
    LANGUAGE = args.lang
    model = whisperx.load_model(args.model, DEVICE, language=LANGUAGE,compute_type=COMPUTE_TYPE)

    
    # 2. load the dataset
    # multi language support: covost2(later work)
    dataLoader = DataLoader()
    ds = dataLoader.load(dataset_name=args.dataset_name, load_script=args.load_script, lang=args.lang, split=args.split, data_dir=args.data_dir)
    dataset_reformer = MyReformer()
    ds = dataset_reformer(ds, name=args.dataset_name)
    assert isinstance(ds, Dataset) == True, "ds is not a instance of class datasets.Dataset"
    logger.info(ds.info.description)
    logger.info(ds)
    
    transcriptions=[] # references 
    paths=[] # audio file path will be here  

    for sample in ds:
        transcriptions.append(sample["transcription"])
        paths.append(sample["audio"]["path"])

    # 3. generate predictions and make some post processing. 
    predictions = []
    for path, transcription in tqdm(zip(paths, transcriptions), total=len(paths),desc="Running Prediction", ncols=100, ascii=' =', leave=True):
        audio = whisperx.load_audio(path)
        result = model.transcribe(audio, batch_size=BATCH_SIZE)
        predictions.append(result['segments'][0]["text"])
        
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
from datasets import load_dataset
import os 
class DataLoader:
    def __init__(self):
        pass 
    
    def load(self, dataset_name, **kwargs):
        load_script = kwargs.pop("load_script")
        token=os.environ["HF_TOKEN"]
        lang = kwargs.pop("lang")
        split = kwargs.pop("split", "")
        if dataset_name == "cv5":
            return load_dataset(load_script, lang, token=token, trust_remote_code=True)
        if dataset_name == "cv9":
            return load_dataset(load_script, lang, token=token, trust_remote_code=True)
        if dataset_name == "ihm":
            return load_dataset(load_script, 'ihm', trust_remote_code=True)
        if dataset_name == "sdm":
            return load_dataset(load_script, 'sdm', trust_remote_code=True)
        if dataset_name == "libri":
            return load_dataset(load_script, split=split, trust_remote_code=True)
        if dataset_name == "ted":
            return load_dataset(load_script, 'release1', trust_remote_code=True)
        if dataset_name == "fleurs":
            return load_dataset(load_script, lang, split=split, trust_remote_code=True)
        if dataset_name == "vox":
            return load_dataset(load_script, lang, trust_remote_code = True)
        if dataset_name == "covost2":
            return load_dataset(load_script, "en_ja", split=split, data_dir="/home/ubuntu/covost", trust_remote_code=True)
        
    
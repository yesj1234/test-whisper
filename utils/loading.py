from datasets import load_dataset
import os 
class DataLoader:
    def __init__(self):
        pass 
    
    def load(self, dataset_name, **kwargs):
        load_script = kwargs.pop("load_script")
        lang = kwargs.pop("lang")
        split = kwargs.pop("split", "")
        if dataset_name == "cv5":
            token=os.environ["HF_TOKEN"]
            return load_dataset(load_script, lang, token=token, trust_remote_code=True)
        if dataset_name == "cv9":
            token=os.environ["HF_TOKEN"]
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
            data_dir = kwargs.pop("data_dir", "")
            return load_dataset(load_script, lang, split=split, data_dir=data_dir, trust_remote_code=True)
        
    
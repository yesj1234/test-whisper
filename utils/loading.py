from datasets import load_dataset
import os 
class DataLoader:
    def __init__(self):
        pass 
    
    def load_cv5(self, script, lang, token): 
        ds = load_dataset(script, lang, token=token, trust_remote_code=True)
        return ds 
    
    
    def load_cv9(self, script, lang, token): 
        ds = load_dataset(script, lang, token=token, trust_remote_code=True)
        return ds
    
    
    def load_ihm(self, script):
        ds = load_dataset(script, 'ihm', trust_remote_code=True)
        return ds
    
    
    def load_sdm(self, script):
        ds = load_dataset(script, 'sdm', trust_remote_code=True)
        return ds        
    
    
    def load_libri(self, script, split):
        ds = load_dataset(script, split=split, trust_remote_code=True)
        return ds
    
    def load_ted(self, script):
        ds = load_dataset(script, 'release1', trust_remote_code=True)
        return ds 
    
    def load_fleurs(self, script, lang, split):
        ds = load_dataset(script, lang, split=split, trust_remote_code=True)
        return ds  
    
    def load_vox(self, script, lang):
        ds = load_dataset(script, lang, trust_remote_code = True)
        return ds
    
    def load(self, dataset_name, **kwargs):
        load_script = kwargs.pop("load_script")
        token=os.environ["HF_TOKEN"]
        lang = kwargs.pop("lang")
        split = kwargs.pop("split", "")
        if dataset_name == "cv5":
            return self.load_cv5(load_script,lang,token)
        if dataset_name == "cv9":
            return self.load_cv9(load_script,lang,token)
        if dataset_name == "ihm":
            return self.load_ihm(load_script)
        if dataset_name == "sdm":
            return self.load_sdm(load_script)
        if dataset_name == "libri":
            return self.load_libri(load_script, split)
        if dataset_name == "ted":
            return self.load_ted(load_script)
        if dataset_name == "fleurs":
            return self.load_fleurs(load_script, lang, split)
        if dataset_name == "vox":
            return self.load_vox(load_script, lang)
    
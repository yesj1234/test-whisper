# Post Processing datasets from multiple sources registered in huggingface hub. 
from datasets import Dataset, DatasetDict
import logging 

class MyReformer:
    def __init__(self): 
        self.logger = self.setup_logger()
        self.dataset_cols = {
            "covost2": {},
            "common_voice5.1":{
                "remove_cols":  ['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'],
                "remain_cols": ['sentence','audio']},
            "common_voice9": {
                "remove_cols":  ['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'],
                "remain_cols": ['sentence', 'audio']},
            "fleur": {
                "remove_cols": ['id', 'num_samples', 'path', 'gender', 'lang_id', 'language', 'lang_group_id'],
                "remain_cols": ['transcription', 'audio']},
            "ihm": {
                "remove_cols": ['meeting_id', 'id', 'text', 'audio', 'begin_time', 'end_time', 'microphone_id', 'speaker_id', 'whisper_transcript'],
                "remain_cols": ['text', 'audio']}
            "sdm": {
                "remove_cols": ['meeting_id', 'id', 'text', 'audio', 'begin_time', 'end_time', 'microphone_id', 'speaker_id', 'whisper_transcript'],
                "remain_cols": ['text', 'audio']},
            "ted": {
                "remove_cols": ['speaker_id', 'gender', 'id', 'file'],
                "remain_cols": ['text', 'audio']},
            "vox": {
                "remove_cols": ["language", "raw_text", "gender", "speaker_id", "is_gold_transcript", "accent", "audio_id"],
                "remain_cols": ["normalized_text", 'audio']
            }
        }
        
    @classmethod
    def setup_logger(cls, self):
        logger = logging.getLogger("MyPostProcessor")
        logger.setLevel(logging.INFO)
        logger.setFormatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt = "%Y-%b-%d %H:%M%S")
        logger.addHandler(logging.StreamHandler())
        return logger  

    def process_dataset(self, dataset, name):
        #1. if self.dataset is of class DatasetDict get the test split only
        if isinstance(dataset, DatasetDict):
            first_split = list(dataset.keys())[0] # should be test split 
            dataset = dataset[first_split]
        
        #2. filter out unnecessary columns 
        if not name or name == "":
            raise ValueError(f"dataset name not given. pass the name to the class. {', '.join([*self.remove_cols.keys()])}")
        if name not in [*self.dataset_cols.keys()]:
            raise ValueError(f"{name} not registered in the processor class.")
        
        cols_to_remove = self.dataset_cols[name]["remove_cols"]
        remain_cols = self.dataset_cols[name]["remain_cols"]
        dataset.remove_columns(cols_to_remove)
        dataset.rename_column(remain_cols[0], "transcription")
                
            
        return dataset 
    
    
            
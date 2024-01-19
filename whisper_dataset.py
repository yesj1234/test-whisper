import os
import datasets 
import re
import librosa 

from datasets.download.download_manager import DownloadManager

_DESCRIPTION = "Data for testing Whisper model"

class SampleSpeech(datasets.GeneratorBasedBuilder): 
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "target_text": datasets.Value("string"),
                    "duration": datasets.Value("float"),
                    "audio": datasets.Audio(sampling_rate=16_000)
                }
            )
        )
    
    """Returns SplitGenerators."""
    VERSION = datasets.Version("0.0.1")
    def _split_generators(self, dl_manager: DownloadManager):
        self.data_dir = os.path.join("/home/ubuntu/split_for_snappy")
        self.audio_dir = os.path.join("/home/ubuntu/3.보완조치완료/2.Validation/")
        return [
            datasets.SplitGenerator(
                name="game",
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "game_refined.tsv"),
                    "split": "game"
                }
            ),
            datasets.SplitGenerator(
                name="travel",
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "travel_refined.tsv"),
                    "split": "travel"
                }
            ),
            datasets.SplitGenerator(
                name="food",
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "food_refined.tsv"),
                    "split": "food"
                }
            ),
            datasets.SplitGenerator(
                name="communication",
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "communication_refined.tsv"),
                    "split": "communication"
                }
            ),
            datasets.SplitGenerator(
                name="fashion",
                gen_kwargs={
                    "filepath": os.path.join(self.data_dir, "fashion_refined.tsv"),
                    "split": "fashion"
                }
            )
        ]    
    def _generate_examples(self, filepath, split): 
        """Yields examples as (key, example) tuples."""
        with open(filepath, encoding='utf-8') as f:
            data = f.read().strip()
            for id_, row in enumerate(data.split("\n")):
                path, sentence = tuple(row.split(" :: "))
                if os.path.exists(os.path.join(self.audio_dir, path)):
                    with open(os.path.join(self.audio_dir, path), 'rb') as audio_file:
                        audio_data = audio_file.read()
                    audio = {
                        "path": os.path.join(self.audio_dir, path),
                        "bytes": audio_data,
                        "sampling_rate": 16_000
                    }
                    duration = librosa.get_duration(path=os.path.join(self.audio_dir, path))
                    yield id_, {
                        "file": os.path.join(self.audio_dir, path),
                        "audio": audio,
                        "target_text": sentence,
                        "duration": duration
                    }
                

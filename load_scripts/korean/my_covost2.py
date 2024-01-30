# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import csv
import os

import pandas as pd

import datasets


_VERSION = "1.0.0"

_CITATION = """
@misc{wang2020covost,
    title={CoVoST 2: A Massively Multilingual Speech-to-Text Translation Corpus},
    author={Changhan Wang and Anne Wu and Juan Pino},
    year={2020},
    eprint={2007.10310},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
"""

_DESCRIPTION = """
CoVoST 2, a large-scale multilingual speech translation corpus covering translations from 21 languages into English \
and from English into 15 languages. The dataset is created using Mozilla’s open source Common Voice database of \
crowdsourced voice recordings.
"""

_HOMEPAGE = "https://github.com/facebookresearch/covost"


class Covost2(datasets.GeneratorBasedBuilder):
    """CoVOST2 Dataset."""

    VERSION = datasets.Version(_VERSION)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                client_id=datasets.Value("string"),
                file=datasets.Value("string"),
                audio=datasets.Audio(sampling_rate=16_000),
                sentence=datasets.Value("string"),
                id=datasets.Value("string"),
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_root = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        cv_tsv_path = os.path.join(data_root, "validated.tsv")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "source_path": data_root,
                    "cv_tsv_path": cv_tsv_path,
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, source_path, cv_tsv_path, split):
        df = self._load_df_from_tsv(cv_tsv_path)
        
        
        for i, row in df.iterrows():
            yield i, {
                "id": row["path"].replace(".mp3", ""),
                "client_id": row["client_id"],
                "sentence": row["sentence"],
                "file": os.path.join(source_path, "clips", row["path"]),
                "audio": os.path.join(source_path, "clips", row["path"]),
            }

    def _load_df_from_tsv(self, path):
        return pd.read_csv(
            path,
            sep="\t",
            header=0,
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            na_filter=False,
        )
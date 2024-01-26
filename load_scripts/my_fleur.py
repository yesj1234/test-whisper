# coding=utf-8
# Copyright 2022 The Google and HuggingFace Datasets Authors and the current dataset script contributor.
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

import os
from collections import OrderedDict

import datasets

logger = datasets.logging.get_logger(__name__)


""" FLEURS Dataset"""

_FLEURS_LANG_TO_ID = OrderedDict([("Mandarin Chinese", "cmn_hans"), ("Cantonese Chinese", "yue_hant"), ("English", "en"), ("Japanese", "ja"),("Korean", "ko")])
_FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}


_FLEURS_LANG = sorted([ "en_us", "ja_jp", "ko_kr", "cmn_hans", "yue_hant"])
_FLEURS_LONG_TO_LANG = {_FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k for k in _FLEURS_LANG}
_FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}

_FLEURS_GROUP_TO_LONG = OrderedDict({
    "western_european_we": ["Asturian", "Bosnian", "Catalan", "Croatian", "Danish", "Dutch", "English", "Finnish", "French", "Galician", "German", "Greek", "Hungarian", "Icelandic", "Irish", "Italian", "Kabuverdianu", "Luxembourgish", "Maltese", "Norwegian", "Occitan", "Portuguese", "Spanish", "Swedish", "Welsh"],
    "eastern_european_ee": ["Armenian", "Belarusian", "Bulgarian", "Czech", "Estonian", "Georgian", "Latvian", "Lithuanian", "Macedonian", "Polish", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Ukrainian"],
    "central_asia_middle_north_african_cmn": ["Arabic", "Azerbaijani", "Hebrew", "Kazakh", "Kyrgyz", "Mongolian", "Pashto", "Persian", "Sorani-Kurdish", "Tajik", "Turkish", "Uzbek"],
    "sub_saharan_african_ssa": ["Afrikaans", "Amharic", "Fula", "Ganda", "Hausa", "Igbo", "Kamba", "Lingala", "Luo", "Northern-Sotho", "Nyanja", "Oromo", "Shona", "Somali", "Swahili", "Umbundu", "Wolof", "Xhosa", "Yoruba", "Zulu"],
    "south_asian_sa": ["Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Oriya", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"],
    "south_east_asian_sea": ["Burmese", "Cebuano", "Filipino", "Indonesian", "Javanese", "Khmer", "Lao", "Malay", "Maori", "Thai", "Vietnamese"],
    "chinese_japanase_korean_cjk": ["Mandarin Chinese", "Cantonese Chinese", "Japanese", "Korean"],
})
_FLEURS_LONG_TO_GROUP = {a: k for k, v in _FLEURS_GROUP_TO_LONG.items() for a in v}
_FLEURS_LANG_TO_GROUP = {_FLEURS_LONG_TO_LANG[k]: v for k, v in _FLEURS_LONG_TO_GROUP.items()}

_ALL_LANG = _FLEURS_LANG
_ALL_CONFIGS = []

for langs in _FLEURS_LANG:
    _ALL_CONFIGS.append(langs)


# TODO(FLEURS)
_DESCRIPTION = "FLEURS is the speech version of the FLORES machine translation benchmark, covering 2000 n-way parallel sentences in n=102 languages."
_CITATION = ""
_HOMEPAGE_URL = ""

_BASE_PATH = "data/{langs}/"
_DATA_URL = _BASE_PATH + "audio/{split}.tar.gz"
_META_URL = _BASE_PATH + "{split}.tsv"


class FleursConfig(datasets.BuilderConfig):
    """BuilderConfig for xtreme-s"""

    def __init__(
        self, name, description, citation, homepage
    ):
        super(FleursConfig, self).__init__(
            name=self.name,
            version=datasets.Version("2.0.0", ""),
            description=self.description,
        )
        self.name = name
        self.description = description
        self.citation = citation
        self.homepage = homepage


def _build_config(name):
    return FleursConfig(
        name=name,
        description=_DESCRIPTION,
        citation=_CITATION,
        homepage=_HOMEPAGE_URL,
    )


class Fleurs(datasets.GeneratorBasedBuilder):

    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [_build_config(name) for name in _ALL_CONFIGS]

    def _info(self):
        task_templates = None
        langs = _ALL_CONFIGS
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "num_samples": datasets.Value("int32"),
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "transcription": datasets.Value("string"),
                "raw_transcription": datasets.Value("string"),
                "gender": datasets.ClassLabel(names=["male", "female", "other"]),
                "lang_id": datasets.ClassLabel(names=langs),
                "language": datasets.Value("string"),
                "lang_group_id": datasets.ClassLabel(
                    names=list(_FLEURS_GROUP_TO_LONG.keys())
                ),
            }
        )

        return datasets.DatasetInfo(
            description=self.config.description + "\n" + _DESCRIPTION,
            features=features,
            supervised_keys=("audio", "transcription"),
            homepage=self.config.homepage,
            citation=self.config.citation + "\n" + _CITATION,
            task_templates=task_templates,
        )

    # Fleurs
    def _split_generators(self, dl_manager):
        splits = ["test"]

        # metadata_path = dl_manager.download_and_extract(_METADATA_URL)

        
        data_urls = {split: [_DATA_URL.format(langs=self.config.name, split=split)] for split in splits}
        meta_urls = {split: [_META_URL.format(langs=self.config.name, split=split)] for split in splits}

        archive_paths = dl_manager.download(data_urls)
        local_extracted_archives = dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}
        archive_iters = {split: [dl_manager.iter_archive(path) for path in paths] for split, paths in archive_paths.items()}

        meta_paths = dl_manager.download(meta_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archives": local_extracted_archives.get("test", [None] * len(meta_paths.get("test"))),
                    "archive_iters": archive_iters.get("test"),
                    "text_paths": meta_paths.get("test")
                },
            ),
        ]

    def _get_data(self, lines, lang_id):
        data = {}
        gender_to_id = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            (
                _id,
                file_name,
                raw_transcription,
                transcription,
                _,
                num_samples,
                gender,
            ) = line.strip().split("\t")

            lang_group = _FLEURS_LANG_TO_GROUP[lang_id]

            data[file_name] = {
                "id": int(_id),
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "num_samples": int(num_samples),
                "gender": gender_to_id[gender],
                "lang_id": _FLEURS_LANG.index(lang_id),
                "language": _FLEURS_LANG_TO_LONG[lang_id],
                "lang_group_id": list(_FLEURS_GROUP_TO_LONG.keys()).index(
                    lang_group
                ),
            }

        return data

    def _generate_examples(self, local_extracted_archives, archive_iters, text_paths):
        assert len(local_extracted_archives) == len(archive_iters) == len(text_paths)
        key = 0

        if self.config.name == "all":
            langs = _FLEURS_LANG
        else:
            langs = [self.config.name]

        for archive, text_path, local_extracted_path, lang_id in zip(archive_iters, text_paths, local_extracted_archives, langs):
            with open(text_path, encoding="utf-8") as f:
                lines = f.readlines()
                data = self._get_data(lines, lang_id)

            for audio_path, audio_file in archive:
                audio_filename = audio_path.split("/")[-1]
                if audio_filename not in data.keys():
                    continue

                result = data[audio_filename]
                extracted_audio_path = (
                    os.path.join(local_extracted_path, audio_filename)
                    if local_extracted_path is not None
                    else None
                )
                result["path"] = extracted_audio_path
                result["audio"] = {"path": audio_path, "bytes": audio_file.read()}
                yield key, result
                key += 1
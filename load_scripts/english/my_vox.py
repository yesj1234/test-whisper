from collections import defaultdict
import os
import json
import csv

import datasets


_DESCRIPTION = """
A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation.
"""

_CITATION = """
@inproceedings{wang-etal-2021-voxpopuli,
    title = "{V}ox{P}opuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, 
    Semi-Supervised Learning and Interpretation",
    author = "Wang, Changhan  and
      Riviere, Morgane  and
      Lee, Ann  and
      Wu, Anne  and
      Talnikar, Chaitanya  and
      Haziza, Daniel  and
      Williamson, Mary  and
      Pino, Juan  and
      Dupoux, Emmanuel",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics 
    and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.80",
    doi = "10.18653/v1/2021.acl-long.80",
    pages = "993--1003",
}
"""

_HOMEPAGE = "https://github.com/facebookresearch/voxpopuli"

_LICENSE = "CC0, also see https://www.europarl.europa.eu/legal-notice/en/"

_ASR_LANGUAGES = [
    "en"
]
_ASR_ACCENTED_LANGUAGES = [
    "en_accented"
]

_LANGUAGES = _ASR_LANGUAGES + _ASR_ACCENTED_LANGUAGES

_BASE_DATA_DIR = "data/"

_N_SHARDS_FILE = _BASE_DATA_DIR + "n_files.json"

_AUDIO_ARCHIVE_PATH = _BASE_DATA_DIR + "{lang}/{split}/{split}_part_{n_shard}.tar.gz"

_METADATA_PATH = _BASE_DATA_DIR + "{lang}/asr_{split}.tsv"


class VoxpopuliConfig(datasets.BuilderConfig):
    """BuilderConfig for VoxPopuli."""

    def __init__(self, name, languages="all", **kwargs):
        """
        Args:
          name: `string` or `List[string]`:
            name of a config: either one of the supported languages or "multilang" for many languages.
            By default, "multilang" config includes all languages, including accented ones.
            To specify a custom set of languages, pass them to the `languages` parameter
          languages: `List[string]`: if config is "multilang" can be either "all" for all available languages,
            excluding accented ones (default), or a custom list of languages.
          **kwargs: keyword arguments forwarded to super.
        """
        if name == "multilang":
            self.languages = _ASR_LANGUAGES if languages == "all" else languages
            name = "multilang" if languages == "all" else "_".join(languages)
        else:
            self.languages = [name]

        super().__init__(name=name, **kwargs)


class Voxpopuli(datasets.GeneratorBasedBuilder):
    """The VoxPopuli dataset."""

    VERSION = datasets.Version("1.3.0")  # TODO: version
    BUILDER_CONFIGS = [
        VoxpopuliConfig(
            name=name,
            version=datasets.Version("1.3.0"),
            )
        for name in _LANGUAGES + ["multilang"]
    ]
    DEFAULT_WRITER_BATCH_SIZE = 256

    def _info(self):
        features = datasets.Features(
            {
                "audio_id": datasets.Value("string"),
                "language": datasets.ClassLabel(names=_LANGUAGES),
                "audio": datasets.Audio(sampling_rate=16_000),
                "raw_text": datasets.Value("string"),
                "normalized_text": datasets.Value("string"),
                "gender": datasets.Value("string"),  # TODO: ClassVar?
                "speaker_id": datasets.Value("string"),
                "is_gold_transcript": datasets.Value("bool"),
                "accent": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        n_shards_path = dl_manager.download_and_extract(_N_SHARDS_FILE)
        with open(n_shards_path) as f:
            n_shards = json.load(f)

        splits = ["test"]

        audio_urls = defaultdict(dict)
        for split in splits:
            for lang in self.config.languages:
                audio_urls[split][lang] = [
                    _AUDIO_ARCHIVE_PATH.format(lang=lang, split=split, n_shard=i) for i in range(n_shards[lang][split])
                ]

        meta_urls = defaultdict(dict)
        for split in splits:
            for lang in self.config.languages:
                meta_urls[split][lang] = _METADATA_PATH.format(lang=lang, split=split)

        # dl_manager.download_config.num_proc = len(urls)

        meta_paths = dl_manager.download_and_extract(meta_urls)
        audio_paths = dl_manager.download(audio_urls)

        local_extracted_audio_paths = (
            dl_manager.extract(audio_paths) if not dl_manager.is_streaming else
            {
                split: {lang: [None] * len(audio_paths[split][lang]) for lang in self.config.languages} for split in splits
            }
        )
        if self.config.name == "en_accented":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "audio_archives": {
                            lang: [dl_manager.iter_archive(archive) for archive in lang_archives]
                            for lang, lang_archives in audio_paths["test"].items()
                        },
                        "local_extracted_archives_paths": local_extracted_audio_paths["test"],
                        "metadata_paths": meta_paths["test"],
                    }
                ),
            ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "audio_archives": {
                        lang: [dl_manager.iter_archive(archive) for archive in lang_archives]
                        for lang, lang_archives in audio_paths["test"].items()
                    },
                    "local_extracted_archives_paths": local_extracted_audio_paths["test"],
                    "metadata_paths": meta_paths["test"],
                }
            ),
        ]

    def _generate_examples(self, audio_archives, local_extracted_archives_paths, metadata_paths):
        assert len(metadata_paths) == len(audio_archives) == len(local_extracted_archives_paths)
        features = ["raw_text", "normalized_text", "speaker_id", "gender", "is_gold_transcript", "accent"]

        for lang in self.config.languages:
            assert len(audio_archives[lang]) == len(local_extracted_archives_paths[lang])

            meta_path = metadata_paths[lang]
            with open(meta_path) as f:
                metadata = {x["id"]: x for x in csv.DictReader(f, delimiter="\t")}

            for audio_archive, local_extracted_archive_path in zip(audio_archives[lang], local_extracted_archives_paths[lang]):
                for audio_filename, audio_file in audio_archive:
                    audio_id = audio_filename.split(os.sep)[-1].split(".wav")[0]
                    path = os.path.join(local_extracted_archive_path, audio_filename) if local_extracted_archive_path else audio_filename

                    yield audio_id, {
                        "audio_id": audio_id,
                        "language": lang,
                        **{feature: metadata[audio_id][feature] for feature in features},
                        "audio": {"path": path, "bytes": audio_file.read()},
                    }
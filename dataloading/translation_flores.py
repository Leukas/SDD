import datasets
from collections import defaultdict
from itertools import permutations

data_folder = "data/biomed/"

url = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
langids = { "en": "eng", "de": "deu", "ar": "ara", "tr": "tur", "xh": "xho", "zu": "zul"}

LANG_PAIRS = permutations(['ar', 'en', 'de', 'tr', 'xh', 'zu'], 2)

class MTConfig(datasets.BuilderConfig):
    def __init__(self, *args, lang1=None, lang2=None, **kwargs):
        super().__init__(
            *args,
            name=f"{lang1}-{lang2}",
            **kwargs,
        )
        self.lang1 = lang1
        self.lang2 = lang2


class MT(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MTConfig(
            lang1=l1,
            lang2=l2,
            description="%s <-> %s" % (l1, l2),
            version=datasets.Version("1.0.0"),
        ) for l1, l2 in LANG_PAIRS
    ]
    BUILDER_CONFIG_CLASS = MTConfig

    def _info(self):
        return datasets.DatasetInfo(
            description="Translation dataset",
            features=datasets.Features(
                {
                    "translation": datasets.features.Translation(languages=(self.config.lang1, self.config.lang2)),
                },
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(url)

        dev_path1 = "%s/flores101_dataset/dev/%s.dev" % (download_dir, langids[self.config.lang1]) 
        dev_path2 = "%s/flores101_dataset/dev/%s.dev" % (download_dir, langids[self.config.lang2]) 

        devtest_path1 = "%s/flores101_dataset/devtest/%s.devtest" % (download_dir, langids[self.config.lang1]) 
        devtest_path2 = "%s/flores101_dataset/devtest/%s.devtest" % (download_dir, langids[self.config.lang2]) 
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"l1_datapath": dev_path1, "l2_datapath": dev_path2},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"l1_datapath": dev_path1, "l2_datapath": dev_path2},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"l1_datapath": devtest_path1, "l2_datapath": devtest_path2},
            ),
        ]

    def _generate_examples(self, l1_datapath, l2_datapath):
        with open(l1_datapath, encoding="utf-8") as f1, open(l2_datapath, encoding="utf-8") as f2:
            for i, (s1, s2) in enumerate(zip(f1, f2)):
                result = (i, {"translation": {self.config.lang1: s1.strip(), self.config.lang2: s2.strip()}})
                yield result

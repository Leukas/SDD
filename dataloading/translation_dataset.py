import datasets

# def config():
lang1 = "de"
lang2 = "en"
# _TRAIN_FILE = "http://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz"
_L1_TRAIN_FILE = "http://data.statmt.org/news-crawl/de/news.2018.de.shuffled.deduped.gz"
_L2_TRAIN_FILE = "http://data.statmt.org/news-crawl/en/news.2018.en.shuffled.deduped.gz"
_DEV_FOLDER = "http://data.statmt.org/wmt21/translation-task/dev.tgz"

_L1_VALID_FILENAME = "/dev/sgm/newstest2019-ende-ref.de.sgm"
_L2_VALID_FILENAME = "/dev/sgm/newstest2019-ende-src.en.sgm"

_L1_TEST_FILENAME = "/dev/sgm/newstest2020-ende-ref.de.sgm"
_L2_TEST_FILENAME = "/dev/sgm/newstest2020-ende-src.en.sgm"

def strip_sgm(file):
    import re
    lines = []
    for line in file:
        if line.startswith("<seg id="):
            line = re.sub(r'<seg id=\"\d+\">', '', line)
            line = re.sub(r'</seg>', '', line)
            lines.append(line)
    return lines

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
            lang1=lang1,
            lang2=lang2,
            description=f"{lang1} <-> {lang2}",
            version=datasets.Version("1.0.0"),
        )
    ]
    BUILDER_CONFIG_CLASS = MTConfig

    def _info(self):
        return datasets.DatasetInfo(
            description="Translation dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "translation": datasets.Translation(languages=(self.config.lang1, self.config.lang2)),
                },
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        train_file_l1 = dl_manager.download_and_extract(_L1_TRAIN_FILE)
        train_file_l2 = dl_manager.download_and_extract(_L2_TRAIN_FILE)
        # train_l2 = dl_manager.download_and_extract(_TRAIN_L2)

        dev_file = dl_manager.download_and_extract(_DEV_FOLDER)
        print(dev_file)
        valid_l1 = dev_file + _L1_VALID_FILENAME
        valid_l2 = dev_file + _L2_VALID_FILENAME
        test_l1 = dev_file + _L1_TEST_FILENAME
        test_l2 = dev_file + _L2_TEST_FILENAME
        # valid_l2 = dl_manager.download_and_extract(_VALID_L2)

        # test_l1 = dl_manager.download_and_extract(_TEST_L1)
        # test_l2 = dl_manager.download_and_extract(_TEST_L2)


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"l1_datapath": train_file_l1, "l2_datapath": train_file_l2, "train": True},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"l1_datapath": valid_l1, "l2_datapath": valid_l2, "train": False},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"l1_datapath": test_l1, "l2_datapath": test_l2, "train": False},
            ),
        ]

    def _generate_examples(self, l1_datapath, l2_datapath, train):
        if l2_datapath is None:
            with open(l1_datapath, encoding="utf-8") as f1:
                for i, line in enumerate(f1):
                    line_split = line.split("\t")
                    s1 = line_split[0]
                    s2 = line_split[1]
                    result = (i, {"id": str(i), "translation": {lang1: s1, lang2: s2}})
                    yield result
        elif train:
            with open(l1_datapath, encoding="utf-8") as f1, open(l2_datapath, encoding="utf-8") as f2:
                for i, (s1, s2) in enumerate(zip(f1, f2)):
                    if i < 1000000:
                        result = (i, {"id": str(i), "translation": {lang1: s1, lang2: s2}})
                        yield result
        else:
            with open(l1_datapath, encoding="utf-8") as f1, open(l2_datapath, encoding="utf-8") as f2:
                lines1 = strip_sgm(f1)
                lines2 = strip_sgm(f2)
                assert len(lines1) == len(lines2), str(len(lines1)) + ", " + str(len(lines2))

                for i, (s1, s2) in enumerate(zip(lines1, lines2)):
                    result = (i, {"id": str(i), "translation": {lang1: s1.strip(), lang2: s2.strip()}})
                    yield result
        #     for sentence_counter, (x, y) in enumerate(zip(f1, f2)):
        #         x = x.strip()
        #         y = y.strip()
        #         result = (
        #             sentence_counter,
        #             {
        #                 "id": str(sentence_counter),
        #                 "translation": {"ig": x, "en": y},
        #             },
        #         )
        #         yield result

import datasets

lang1 = "xh"
lang2 = "zu"

train_folder = "/data/p284491/nmt/data/mono/xhzu/data/para"
dev_folder = "/data/p284491/nmt/data/mono/xhzu/data/dev"

train_folder = "data/xhzu/"
dev_folder = "data/xhzu/"

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
                    "translation": datasets.features.Translation(languages=(self.config.lang1, self.config.lang2)),
                },
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        train_file_l1 = train_folder + "/sent.%s" % self.config.lang1
        train_file_l2 = train_folder + "/sent.%s" % self.config.lang2

        valid_file_l1 = dev_folder + "/dev.%s" % self.config.lang1
        valid_file_l2 = dev_folder + "/dev.%s" % self.config.lang2

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"l1_datapath": train_file_l1, "l2_datapath": train_file_l2},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"l1_datapath": valid_file_l1, "l2_datapath": valid_file_l2},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"l1_datapath": valid_file_l1, "l2_datapath": valid_file_l2},
            ),
        ]

    def _generate_examples(self, l1_datapath, l2_datapath):
        with open(l1_datapath, encoding="utf-8") as f1, open(l2_datapath, encoding="utf-8") as f2:
            for i, (s1, s2) in enumerate(zip(f1, f2)):
                result = (i, {"translation": {self.config.lang1: s1.strip(), self.config.lang2: s2.strip()}})
                yield result

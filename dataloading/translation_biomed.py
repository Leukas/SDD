import datasets
from collections import defaultdict

data_folder = "data/biomed/"

def align_files(filepath1, filepath2):
    aligned_lines = defaultdict(list)
    with open(filepath1, 'r', encoding='utf8') as file1:
        for line in file1:
            doc_id, sid, sentence = line.split("\t", 2)
            aligned_lines[(doc_id, sid)].append(sentence.strip())

    with open(filepath2, 'r', encoding='utf8') as file2:
        for line in file2:
            doc_id, sid, sentence = line.split("\t", 2)
            aligned_lines[(doc_id, sid)].append(sentence.strip())

    return list(aligned_lines.values())

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
            lang1="de",
            lang2="en",
            description="de <-> en",
            version=datasets.Version("1.0.0"),
        ),
        MTConfig(
            lang1="en",
            lang2="de",
            description="en <-> de",
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
        file_l1 = data_folder + "/medline_%s2%s_%s.txt" % (self.config.lang1, self.config.lang2, self.config.lang1)
        file_l2 = data_folder + "/medline_%s2%s_%s.txt" % (self.config.lang1, self.config.lang2, self.config.lang2)

        aligned = align_files(file_l1, file_l2)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"lines": aligned},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"lines": aligned},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"lines": aligned},
            ),
        ]

    def _generate_examples(self, lines):
        for i, sent in enumerate(lines):
            if len(sent) != 2:
                continue
            
            s1 = sent[0]
            s2 = sent[1]
            result = (i, {"translation": {self.config.lang1: s1.strip(), self.config.lang2: s2.strip()}})
            yield result

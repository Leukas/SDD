# Subword-Delimited Downsampling
Here you can find our code related to the work in:
Lukas Edman, Antonio Toral, and Gertjan van Noord. 2022. Subword-Delimited Downsampling for Better Character-Level Translation. _Findings of the Association for Computational Linguistics: EMNLP 2022. ([Arxiv Link](https://arxiv.org/pdf/2212.01304.pdf))


### Requirements
- Python 3
- [PyTorch](https://pytorch.org/)
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [pynvml](https://pypi.org/project/pynvml/)
- [einops](https://github.com/arogozhnikov/einops)
- [sentencepiece](https://github.com/google/sentencepiece)

### Running
#### Training
To train the models, run ```rc.py``` for review classification, ```nli.py``` for NLI, ```nmt_sdd.py``` for NMT (SDD model), or ```nmt.py``` for NMT (all other models). For example, training a byte-level model for NMT on German into English would look like:
```python nmt.py --batch_size 32 --grad_acc 4 --lr 2e-4 --langs de,en --model_type char```

For the word-level and SDD models, it requires a trained SentencePiece model (otherwise it will use the tokenization of mT5-small). To do this, add the flag: ```--spm_model <path_to_spm>```.

#### Evaluation
Models are saved according to a SLURM ID, or a randomly generated ID if SLURM isn't used. To reload a model, add the flag: ```--reload_id <ID>``` 
To skip training, use the ```--eval_only``` flag. 


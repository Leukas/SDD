import sys
import torch
from transformers import TrainerCallback

class EmbeddingLengthSaveCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def __init__(self, model_type, logger):
        super().__init__()
        self.logger = logger
        self.model_type = model_type
        self.prev_embeddings_enc = None
        self.prev_embeddings_dec = None

    def get_embeddings(self, model):
        if self.model_type == "word" or self.model_type == "char":
            embeddings_enc = model.encoder.embeddings.word_embeddings.weight
            embeddings_dec = model.decoder.bert.embeddings.word_embeddings.weight
        elif self.model_type == "sdd":
            embeddings_enc = model.enc_lee.emb.weight
            embeddings_dec = model.dec_lee.emb.weight
        return embeddings_enc.clone(), embeddings_dec.clone()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.prev_embeddings_enc, self.prev_embeddings_dec = self.get_embeddings(model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        embeddings_enc, embeddings_dec = self.get_embeddings(model)
        lr = logs["learning_rate"]
        cossim = torch.nn.functional.cosine_similarity
        avg_dist = (1 - cossim(embeddings_enc, self.prev_embeddings_enc)).mean() / lr
        self.logger.info("Average encoder emb dist change: %.8f" % avg_dist)
        avg_dist = (1 - cossim(embeddings_dec, self.prev_embeddings_dec)).mean() / lr
        self.logger.info("Average decoder emb dist change: %.8f" % avg_dist)
        sys.stdout.flush()

        self.prev_embeddings_enc, self.prev_embeddings_dec = self.get_embeddings(model)


        # _ = logs.pop("total_flos", None)
        # if state.is_local_process_zero:
        #     self.logger.info(logs)
        #     sys.stdout.flush()

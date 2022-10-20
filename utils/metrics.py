# metrics.py
import os
import sys
import numpy as np
from datasets import load_metric


acc = load_metric('accuracy')
def compute_acc(eval_pred):
    logits, labels = eval_pred

    predictions = logits.argmax(axis=-1)
    predictions = predictions.flatten()
    labels = labels.flatten()

    mask = labels != -100

    labels = labels[mask]
    predictions = predictions[mask]

    return acc.compute(predictions=predictions, references=labels)



bleu = load_metric('sacrebleu')
def compute_bleu(eval_pred, step_num, args):
    predictions, labels = eval_pred

    char_preds = []
    char_labels = []
    for b in range(predictions.shape[0]):
        pred = predictions[b]
        if (pred==args.tok.eos_token_id).sum() > 0:
            eos_idx = np.argmax(pred==args.tok.eos_token_id)
            pred = pred[:eos_idx]
        else:
            pred = pred[pred!=-100]

        lab = labels[b]
        if (lab==args.tok.eos_token_id).sum() > 0:
            eos_idx = np.argmax(lab==args.tok.eos_token_id)
            lab = lab[:eos_idx]
        else:
            lab = lab[lab!=-100]

        if b < 5:
            print("P:", pred)
            print("L:", lab)
        char_preds.append(args.tok.decode(pred, skip_special_tokens=True).strip())
        char_labels.append([args.tok.decode(lab, skip_special_tokens=True).strip()])

    print("\npred:", char_preds[0:5])
    print("\nlabel:", char_labels[0:5])

    bleu_score = {"bleu": bleu.compute(predictions=char_preds, references=char_labels)['score']}

    write_lines(char_preds, char_labels, step_num, args)

    sys.stdout.flush()
    return bleu_score
    
def write_lines(preds, labels, step, args):
    test_set = "test" if args.eval_only else "valid"

    ref_path = args.save_path + "ref.%s" % test_set 
    if not os.path.exists(ref_path):
        with open(ref_path, 'w', encoding='utf8') as file:
            for i in range(len(labels)):
                file.write(labels[i][0] + "\n")

    out_path = args.save_path + "out.%s.%s" % (test_set, step)
    with open(out_path, 'w', encoding='utf8') as file:
        for i in range(len(preds)):
            file.write(preds[i] + "\n")
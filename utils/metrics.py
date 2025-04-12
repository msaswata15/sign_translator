from nltk.translate.bleu_score import corpus_bleu
from jiwer import wer

def compute_bleu(references, hypotheses):
    refs = [[ref.split()] for ref in references]
    hyps = [hyp.split() for hyp in hypotheses]
    return corpus_bleu(refs, hyps)

def compute_wer(references, hypotheses):
    return wer(references, hypotheses)
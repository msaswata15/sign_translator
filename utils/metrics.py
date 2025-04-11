from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer

def compute_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def compute_wer(reference, hypothesis):
    return wer(reference, hypothesis)

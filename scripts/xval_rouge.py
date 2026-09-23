import sys, os; sys.path.insert(0, os.getcwd())
import random
from rouge_score import rouge_scorer
import lamp_harness as H

sc = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=False)
random.seed(0)
vocab = "deep learning neural network for graph transformer model efficient sparse attention a the of on".split()
cases = [
    ("the cat sat on the mat", "the cat sat on a mat"),
    ("Attention Is All You Need", "attention is all you need"),
    ("A Survey of Graph Neural Networks", "Graph Neural Networks: A Comprehensive Survey"),
    ("cat cat", "cat"),
    ("abc", "xyz"),
    ("a b c d", "d c b a"),
    ("Deep Learning, for: Sparse!! Attention?", "deep learning for sparse attention"),
]
for _ in range(300):
    a = " ".join(random.choices(vocab, k=random.randint(1,14)))
    b = " ".join(random.choices(vocab, k=random.randint(1,14)))
    cases.append((a,b))

max1 = maxL = 0.0
for pred, ref in cases:
    ref_ = sc.score(ref, pred)   # (target, prediction)
    m1 = H.rouge_1(pred, ref); mL = H.rouge_l(pred, ref)
    for key, mine, theirs in (("r1", m1, ref_["rouge1"]), ("rL", mL, ref_["rougeL"])):
        for fld in ("precision","recall","fmeasure"):
            d = abs(mine[fld] - getattr(theirs, fld))
            if key=="r1": max1 = max(max1, d)
            else: maxL = max(maxL, d)
            assert d < 1e-9, (key, fld, pred, ref, mine[fld], getattr(theirs, fld))
print(f"cases compared: {len(cases)}")
print(f"max abs deviation ROUGE-1: {max1:.3e}")
print(f"max abs deviation ROUGE-L: {maxL:.3e}")
print("rouge-score version:", __import__("rouge_score").__file__.split('/')[-2])

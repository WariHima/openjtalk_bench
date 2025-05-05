# %%
from pathlib import Path
import jaconv

text = Path('src_yomi/aozora_01.tsv').read_text(encoding = "utf-8")


text = text.split("\n")
text = text[: len(text)//100 ]
i = 0
label = []
data = []

for row in text:
    yomi = row.split("\t")[2] 
    yomi = jaconv.hira2kata(yomi)

    kanji = row.split("\t")[1]
    kanji = kanji.replace(" ", "")

    data.append(kanji)
    label.append(yomi)

# %%
import pyopenjtalk
from tqdm.autonotebook import tqdm

ref = []
hyp = []
for i, d in enumerate(tqdm(data)):
    hyp.append(list(pyopenjtalk.g2p(d, kana=True)))
    ref.append([list(label[i])])

# %%
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

smooth = SmoothingFunction()

corpus_score_1 = corpus_bleu(ref, hyp, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
corpus_score_2 = corpus_bleu(ref, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
corpus_score_3 = corpus_bleu(ref, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
corpus_score_4 = corpus_bleu(ref, hyp, smoothing_function=smooth.method1)

# %%
print("### aozora score:")
print(corpus_score_1)
print(corpus_score_2)
print(corpus_score_3)
print(corpus_score_4)
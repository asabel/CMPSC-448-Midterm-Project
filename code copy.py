!pip install -q datasets scikit-learn pandas matplotlib seaborn
import os
import torch; print(torch.cuda.is_available()) 

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
from datasets import load_dataset
from collections import Counter
import pandas as pd

# Load the dataset and parse out into model, input, and output. Stops downloading the data after 150,000 valid data records.
ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
rows = []
for ex in ds:
    if ex["language"] != "English" or ex["turn"] != 1 or ex["redacted"]:
        continue
    conv = ex["conversation"]
    rows.append({"model": ex["model"],
                 "input": conv[0]["content"],
                 "output": conv[1]["content"]})
    if len(rows) >= 150000: break 
df = pd.DataFrame(rows)
print(Counter(df.model).most_common(25))

# Identify the family of the models.
def family(m):
    m = m.lower()
    if "gpt" in m: return "gpt"
    if "claude" in m: return "claude"
    if "llama" in m: return "llama"
    if "mpt" in m: return "mpt"
    if "vicuna" in m: return "vicuna"   # fine-tuned llama, but a different "family" in practice
    return None
df["family"] = df.model.map(family)
df = df.dropna(subset=["family"])

# Clean, remove duplicates, balance classes, and save
df = df[(df.output.str.len() > 20) & (df.input.str.len() > 5)]
df = df.drop_duplicates(subset=["family", "input", "output"])
N = min(df.family.value_counts().min(), 3000)
df = df.groupby("family").sample(N, random_state=42).reset_index(drop=True)
df.to_csv("data.csv", index=False)
print(df.family.value_counts())

from sklearn.model_selection import GroupShuffleSplit
# Splitting the data into train, validation, and test sets grouped by the input prompt.
groups = df.input.factorize()[0]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss.split(df, groups=groups))
trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
tr_idx, val_idx = next(gss2.split(trainval, groups=trainval.input.factorize()[0]))
train, val = trainval.iloc[tr_idx], trainval.iloc[val_idx]
# ~70 train / 10 validation / 20 test
print(len(train), len(val), len(test))   

import re, torch, numpy as np
from collections import Counter
# Tokenizer
def tok(text):
    return re.findall(r"\n|\w+|[^\w\s]", text.lower())

# Construct a vocabulary out of the training data only.
cnt = Counter(t for s in train.output for t in tok(s))
cnt.update(t for s in train.input for t in tok(s))
vocab = {"<pad>":0, "<unk>":1, "<sep>":2}
for w, c in cnt.most_common(30000):
    if c >= 3: vocab[w] = len(vocab)

MAXLEN_OUT, MAXLEN_IN = 256, 64
# Convert a list of tokens to ids.
def encode(tokens, L):
    ids = [vocab.get(t, 1) for t in tokens][:L]
    return ids + [0]*(L-len(ids))

# Builds the model input from the input prompt and/or output response. 
def make_X(d, mode):
    X = []
    for i, o in zip(d.input, d.output):
        if mode == "output":  X.append(encode(tok(o), MAXLEN_OUT))
        elif mode == "input": X.append(encode(tok(i), MAXLEN_IN))
        else:  # Input and output
            ids = [vocab.get(t,1) for t in tok(i)][:MAXLEN_IN] + [2] + [vocab.get(t,1) for t in tok(o)]
            ids = ids[:MAXLEN_IN+1+MAXLEN_OUT]
            X.append(ids + [0]*(MAXLEN_IN+1+MAXLEN_OUT-len(ids)))
    return torch.tensor(X)

# Assigns unique number to each of the model family names and builds the label.
classes = sorted(df.family.unique())
c2i = {c:i for i,c in enumerate(classes)}
def make_y(d): return torch.tensor(d.family.map(c2i).values)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=50000, lowercase=True)
X_train = vec.fit_transform(train.output); X_test = vec.transform(test.output)
lr = LogisticRegression(max_iter=1000).fit(X_train, train.family)
p = lr.predict(X_test)
print("acc", accuracy_score(test.family, p), "macroF1", f1_score(test.family, p, average="macro"))

import torch.nn as nn, torch.nn.functional as F
dev = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self, V, n_cls, emb=128, nf=100, ks=(3,4,5), drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, nf, k) for k in ks])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(nf*len(ks), n_cls)
    def forward(self, x):
        # (B, emb, L)
        e = self.emb(x).transpose(1,2)                      
        h = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(h, 1)))

class BiLSTM(nn.Module):
    def __init__(self, V, n_cls, emb=128, hid=128, drop=0.5):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid*2, n_cls)
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.lstm(self.emb(x))
        # Max-pool over real tokens.
        h = h.masked_fill(~mask, -1e9).max(dim=1).values    
        return self.fc(self.drop(h))

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

def run(model_cls, mode, seed=0, epochs=15, lr=1e-3, bs=64, train_df=None, test_df=None):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = train if train_df is None else train_df
    te = test if test_df is None else test_df
    X_train, y_train = make_X(tr, mode), make_y(tr)
    X_val, y_val = make_X(val, mode), make_y(val)
    X_test, y_test = make_X(te, mode), make_y(te)
    model = model_cls(len(vocab), len(classes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)

    def predict(X):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                out.append(model(X[i:i+256].to(dev)).argmax(1).cpu())
        return torch.cat(out)

    best, best_state, patience = 0, None, 0
    for ep in range(epochs):
        model.train()
        for x_batch, y_batch in dl:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x_batch), y_batch).backward(); opt.step()
        va = (predict(X_val) == y_val).float().mean().item()
        print(f"epoch {ep+1} val_acc {va:.3f}")
        if va > best: best, best_state, patience = va, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            patience += 1
            # Early stopping if validation accuracy isn't improving.
            if patience >= 3: break          
    model.load_state_dict(best_state)
    pred = predict(X_test)
    return {"acc": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, average="macro"),
            "cm": confusion_matrix(y_test, pred)}

results = {}
for name, cls in [("CNN", CNN), ("BiLSTM", BiLSTM)]:
    for mode in ["input", "output", "both"]:
        for seed in [0, 1, 2]:
            results[(name, mode, seed)] = run(cls, mode, seed)

import numpy as np, pandas as pd

modes = [("input", "Input only"), ("output", "Output only"), ("both", "Input + Output")]
acc_tbl, f1_tbl = {}, {}
for name in ["CNN", "BiLSTM"]:
    acc_row, f1_row = {}, {}
    for mode, label in modes:
        a = [results[(name, mode, s)]["acc"] for s in [0, 1, 2]]
        f = [results[(name, mode, s)]["f1"] for s in [0, 1, 2]]
        acc_row[label] = f"{np.mean(a):.3f} ± {np.std(a):.3f}"
        f1_row[label] = f"{np.mean(f):.3f} ± {np.std(f):.3f}"
    acc_tbl[name], f1_tbl[name] = acc_row, f1_row

acc_df = pd.DataFrame(acc_tbl).T
f1_df = pd.DataFrame(f1_tbl).T
print("TEST ACCURACY (mean ± std over 3 seeds)"); print(acc_df, "\n")
print("TEST MACRO-F1 (mean ± std over 3 seeds)"); print(f1_df)
print("\nChance level:", 1/len(classes))
acc_df.to_csv("rq2_accuracy.csv"); f1_df.to_csv("rq2_f1.csv")

import matplotlib.pyplot as plt
labels = [l for _, l in modes]
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, name in enumerate(["CNN", "BiLSTM"]):
    means = [np.mean([results[(name, m, s)]["acc"] for s in [0,1,2]]) for m, _ in modes]
    stds  = [np.std([results[(name, m, s)]["acc"] for s in [0,1,2]]) for m, _ in modes]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4, label=name)
ax.axhline(1/len(classes), color="red", linestyle="--", label="Chance (20%)")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Test accuracy"); ax.set_title("RQ2: Which input does the classifier see?")
ax.legend(); ax.set_ylim(0, 0.8)
plt.savefig("rq2_bar.png", dpi=200, bbox_inches="tight")

#for RQ 4
df["len_words"] = df.output.str.split().str.len()
df["ttr"] = df.output.map(lambda s: len(set(tok(s)))/max(len(tok(s)),1))
df["n_bullets"] = df.output.str.count(r"\n\s*[-*•\d]+[.)]?\s")
df["has_code"] = df.output.str.contains("```").astype(int)
print(df.groupby("family")[["len_words","ttr","n_bullets","has_code"]].mean())

import pickle
pickle.dump(results, open("results.pkl", "wb"))

print(df.groupby(["family", "model"]).size())   # check whether gpt4all got labeled "gpt"
print(df.family.value_counts())
print("train/val/test:", len(train), len(val), len(test))
print(train.family.value_counts())

import seaborn as sns

def cm_report(name, mode):
    cm = sum(results[(name, mode, s)]["cm"] for s in [0, 1, 2])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{name}, {mode} (row-normalized, 3 seeds)")
    plt.savefig(f"cm_{name}_{mode}.png", dpi=200, bbox_inches="tight"); plt.show()
    tp = np.diag(cm); prec = tp / cm.sum(axis=0); rec = tp / cm.sum(axis=1)
    print(name, mode)
    print(pd.DataFrame({"precision": prec, "recall": rec, "f1": 2*prec*rec/(prec+rec),
                        "n": cm.sum(axis=1)}, index=classes).round(3), "\n")

for name in ["CNN", "BiLSTM"]:
    cm_report(name, "output")
cm_report("CNN", "input")

# TF-IDF accuracy by response length
t = test.copy(); t["correct"] = (p == t.family.values)
t["len"] = t.output.str.split().str.len()
t["bucket"] = pd.qcut(t.len, 4, duplicates="drop")
print(t.groupby("bucket", observed=True).correct.mean().round(3))

d = df.copy()
d["in_len"] = d.input.str.split().str.len()
d["has_code_kw"] = d.input.str.contains(r"def |python|function|code|```", case=False, regex=True)
d["non_ascii"] = d.input.map(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1))
print(d.groupby("family")[["in_len", "has_code_kw", "non_ascii"]].agg(["mean", "median"]).round(3))

vi = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=30000)
li = LogisticRegression(max_iter=1000).fit(vi.fit_transform(train.input), train.family)
print("input-only TF-IDF acc:", accuracy_score(test.family, li.predict(vi.transform(test.input))))
in_names = np.array(vi.get_feature_names_out())
for c, coefs in zip(li.classes_, li.coef_):
    print(c, list(in_names[np.argsort(coefs)[-10:]]))

def task_of(s):
    return "code" if re.search(r"```|\b(python|javascript|java|sql|html|css|function|code|script|c\+\+)\b", s, re.I) else "other"

tr_t, va_t, te_t = [x.assign(task=x.input.map(task_of)) for x in (train, val, test)]
print(te_t.task.value_counts())

rq3 = {}
for src, tgt in [("other", "code"), ("code", "other")]:
    kw = dict(tr_df=tr_t[tr_t.task == src], va_df=va_t[va_t.task == src])
    within = [run_ablated(CNN, "output", s, None, te_df=te_t[te_t.task == src], **kw) for s in [0, 1, 2]]
    cross  = [run_ablated(CNN, "output", s, None, te_df=te_t[te_t.task == tgt], **kw) for s in [0, 1, 2]]
    rq3[f"train on {src}"] = {f"test {src} (in-domain)": f"{np.mean(within):.3f}",
                              f"test {tgt} (unseen)": f"{np.mean(cross):.3f}"}
print(pd.DataFrame(rq3).T)
# hinglish_hate_pipeline_save.py
# ---------------------------------------------------------------
#  Train LSTM, HAN, Enhanced XLM-R ― save the models ―
#  show severity predictions for all three.
# ---------------------------------------------------------------

import os, re, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from tqdm import tqdm
from transformers import (
    XLMRobertaTokenizer, XLMRobertaModel,
    get_linear_schedule_with_warmup
)
from imblearn.over_sampling import RandomOverSampler

# ----------------------------- CONFIG --------------------------
MPS = torch.backends.mps.is_available()
DEVICE_TXT  = torch.device("mps"  if MPS else ("cuda" if torch.cuda.is_available() else "cpu"))
DEVICE_XLMR = torch.device("cpu")           # keep transformer on CPU

DATA_PATH = "data/raw/Hate-speech-dataset/hate_speech.tsv"

BATCH_SIZE   = 4
EPOCHS_BASE  = 3
EPOCHS_XLMR  = 6
LR_BASE      = 1e-3
LR_XLMR      = 5e-5
DROPOUT      = 0.3
GRAD_ACCUM   = 2
WARMUP_RATIO = 0.1

# ------------------------ text helpers -------------------------
def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^a-zA-Z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def severity(prob: float) -> int:
    if prob < 0.5:  return 0   # No hate
    if prob < 0.7:  return 1   # Low
    if prob < 0.85: return 2   # Moderate
    return 3                   # High

# -------------------------- dataset ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None):
        self.texts  = [clean(t) for t in texts]
        self.labels = labels
        self.vocab  = vocab or self._build_vocab()
        self.enc    = [self._encode(t) for t in self.texts]

    def _build_vocab(self):
        c = Counter()
        for t in self.texts: c.update(t.split())
        v = {w: i+2 for i, (w, _) in enumerate(c.most_common())}
        v["<PAD>"] = 0; v["<UNK>"] = 1
        return v

    def _encode(self, txt):
        return [self.vocab.get(w, 1) for w in txt.split()]

    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        return torch.tensor(self.enc[i]), torch.tensor(self.labels[i])

def pad_batch(batch):
    xs, ys = zip(*batch)
    lens = [len(x) for x in xs]
    xs_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return xs_pad, torch.tensor(ys), torch.tensor(lens)

def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t", header=None,
                     usecols=[0, 1], names=["text", "label"], quoting=3)
    df = df[df["label"].isin(["yes", "no"])]
    texts  = df["text"].astype(str).tolist()
    labels = df["label"].map({"no": 0, "yes": 1}).astype(int).tolist()

    X_res, y_res = RandomOverSampler().fit_resample(
        np.array(texts).reshape(-1, 1), labels)
    texts = X_res.flatten().tolist(); labels = list(y_res)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print("Train:", Counter(y_train), " Val:", Counter(y_val))
    return X_train, X_val, y_train, y_val

# ------------------- classic models ----------------------------
class LSTM(nn.Module):
    def __init__(self, vocab, emb=100, hid=128):
        super().__init__(); self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.dp   = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(hid, 2)
    def forward(self, x, lengths):
        p = nn.utils.rnn.pack_padded_sequence(
            self.emb(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(p)
        return self.fc(self.dp(h[-1]))

class HAN(nn.Module):
    def __init__(self, vocab, emb=100, hid=128):
        super().__init__(); self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), emb, padding_idx=0)
        self.wgru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.watt = nn.Linear(hid*2, 1)
        self.sgru = nn.GRU(hid*2, hid, batch_first=True, bidirectional=True)
        self.satt = nn.Linear(hid*2, 1)
        self.dp   = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(hid*2, 2)
    def _att(self, x, a):
        w = torch.softmax(a(x), 1)
        return (w * x).sum(1)
    def forward(self, x, lengths):
        w_out, _ = self.wgru(self.emb(x))
        s_vec = self._att(w_out, self.watt).unsqueeze(1)
        d_out, _ = self.sgru(s_vec)
        d_vec = self._att(d_out, self.satt)
        return self.fc(self.dp(d_vec))

# ---------------- enhanced XLM-R -------------------------------
class XLMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        # freeze bottom 4 layers + embeddings
        for n, p in self.enc.named_parameters():
            if "embeddings" in n or any(f"layer.{i}." in n for i in range(4)):
                p.requires_grad = False
        self.dp  = nn.Dropout(DROPOUT)
        hid = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.LayerNorm(hid//2),
            nn.Dropout(DROPOUT),
            nn.Linear(hid//2, 2)
        )
    def forward(self, ids, mask):
        cls = self.enc(ids, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.head(self.dp(cls))

# -------------------- train / evaluate -------------------------
def train_epoch(model, loader, opt, crit, device, sched=None):
    model.train(); opt.zero_grad(); tot = 0
    for step, batch in enumerate(tqdm(loader, desc="Train")):
        if isinstance(model, XLMR):
            ids, mask, lab = [t.to(device) for t in batch]
            out = model(ids, mask)
        else:
            x, lab, lens = batch
            x, lab, lens = x.to(device), lab.to(device), lens.to(device)
            out = model(x, lens)
        loss = crit(out, lab); loss.backward()
        if (step+1) % GRAD_ACCUM == 0:
            opt.step(); opt.zero_grad(); sched and sched.step()
        tot += loss.item()
    return tot / len(loader)

def evaluate(model, loader, device):
    model.eval(); p, l = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval"):
            if isinstance(model, XLMR):
                ids, mask, lab = [t.to(device) for t in batch]
                out = model(ids, mask)
            else:
                x, lab, lens = batch
                x, lab, lens = x.to(device), lab.to(device), lens.to(device)
                out = model(x, lens)
            p.extend(torch.argmax(out, 1).cpu().numpy())
            l.extend(lab.cpu().numpy())
    print("\n" + classification_report(l, p))
    pr, rc, f1, _ = precision_recall_fscore_support(l, p, average="weighted")
    print(f"Weighted Precision {pr:.4f}  Recall {rc:.4f}  F1 {f1:.4f}")

# -------------------- severity helpers ------------------------
def predict_classic(model, vocab, text):
    ids = [vocab.get(w, 1) for w in clean(text).split()]
    x = torch.tensor([ids]).to(DEVICE_TXT)
    ln = torch.tensor([len(ids)]).to(DEVICE_TXT)
    with torch.no_grad():
        probs = torch.softmax(model(x, ln), 1)
    ph = probs[0, 1].item()
    return ("Hate" if ph >= 0.5 else "Non-hate"), ph, severity(ph)

def predict_xlmr(model, tok, text):
    model.eval()
    enc = tok([text], return_tensors="pt", padding=True,
              truncation=True).to(DEVICE_XLMR)
    with torch.no_grad():
        probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), 1)
    ph = probs[0, 1].item()
    return ("Hate" if ph >= 0.5 else "Non-hate"), ph, severity(ph)

# ----------------------------- MAIN ---------------------------
if name == "__main__":
    Xtr, Xv, ytr, yv = load_data()

    # ======== LSTM & HAN ========
    train_ds = TextDataset(Xtr, ytr)
    val_ds   = TextDataset(Xv,  yv, vocab=train_ds.vocab)
    tr_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=pad_batch)
    vl_ld = DataLoader(val_ds,  BATCH_SIZE, collate_fn=pad_batch)
    cw = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
    crit_base = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float,
                                                        device=DEVICE_TXT))

    lstm = LSTM(train_ds.vocab).to(DEVICE_TXT)
    opt_l = torch.optim.Adam(lstm.parameters(), lr=LR_BASE)
    print("\nLSTM training")
    for ep in range(EPOCHS_BASE):
        loss = train_epoch(lstm, tr_ld, opt_l, crit_base, DEVICE_TXT)
        print(f"  Epoch {ep+1}/{EPOCHS_BASE}  Loss {loss:.4f}")
    evaluate(lstm, vl_ld, DEVICE_TXT)

    han = HAN(train_ds.vocab).to(DEVICE_TXT)
    opt_h = torch.optim.Adam(han.parameters(), lr=LR_BASE)
    print("\nHAN training")
    for ep in range(EPOCHS_BASE):
        loss = train_epoch(han, tr_ld, opt_h, crit_base, DEVICE_TXT)
        print(f"  Epoch {ep+1}/{EPOCHS_BASE}  Loss {loss:.4f}")
    evaluate(han, vl_ld, DEVICE_TXT)

    # ======== XLM-Roberta ========
    tok = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    def enc(texts, labels):
        e = tok(texts, padding=True, truncation=True, return_tensors="pt")
        return DataLoader(list(zip(e["input_ids"], e["attention_mask"],
                                   torch.tensor(labels))),
                          batch_size=BATCH_SIZE, shuffle=True)
    tr_ld_r = enc(Xtr, ytr); vl_ld_r = enc(Xv, yv)

    crit_r = nn.CrossEntropyLoss(weight=torch.tensor(
        cw, dtype=torch.float, device=DEVICE_XLMR))

    xlmr = XLMR().to(DEVICE_XLMR)
    opt_r = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                     xlmr.parameters()), lr=LR_XLMR)
    total = len(tr_ld_r) // GRAD_ACCUM * EPOCHS_XLMR
    sched = get_linear_schedule_with_warmup(opt_r,
                                            int(total * WARMUP_RATIO), total)
    print("\nXLM-Roberta training")
    for ep in range(EPOCHS_XLMR):
        loss = train_epoch(xlmr, tr_ld_r, opt_r, crit_r, DEVICE_XLMR, sched)
        print(f"  Epoch {ep+1}/{EPOCHS_XLMR}  Loss {loss:.4f}")
    evaluate(xlmr, vl_ld_r, DEVICE_XLMR)

    # ======== SAVE MODELS ========
    torch.save({"state_dict": lstm.state_dict(), "vocab": train_ds.vocab},
               "lstm_hinglish.pt")
    torch.save({"state_dict": han.state_dict(),  "vocab": train_ds.vocab},
               "han_hinglish.pt")
    torch.save({"state_dict": xlmr.state_dict()}, "xlmr_hinglish.pt")
    tok.save_pretrained("xlmr_tokenizer")           # folder with tokenizer files
    print("\n✔ Models saved to disk")

    # ======== Severity Demo ========
    examples = [
        "Aap bahut achhe ho",
        "Thoda stupid comment hai",
        "Yeh log hamesha ganda bolte hain",
        "Tum jaise logo ko jinda jalana chahiye"
    ]
    print("\nSeverity predictions (LSTM / HAN / XLM-R):")
    for txt in examples:
        lbl_l, p_l, s_l = predict_classic(lstm, train_ds.vocab, txt)
        lbl_h, p_h, s_h = predict_classic(han,  train_ds.vocab, txt)
        lbl_x, p_x, s_x = predict_xlmr(xlmr, tok, txt)
        print(f"\n\"{txt}\"\n"
              f"  LSTM → {lbl_l}  p={p_l:.3f}  sev={s_l}\n"
              f"  HAN  → {lbl_h}  p={p_h:.3f}  sev={s_h}\n"
              f"  XLM-R→ {lbl_x}  p={p_x:.3f}  sev={s_x}")

    print("\n✅ Done!")
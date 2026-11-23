"""Lipi: a small character-level Transformer that turns Banglish into Bangla script.

  python lipi.py train --minutes 15          # train (safe limits below), saves checkpoints/lipi.pt
  python lipi.py eval                        # character error rate vs. the Avro phonetic baseline
  python lipi.py "ami tomake valobashi"      # convert text

Mac safety: runs at low CPU priority, caps GPU memory at 40% of RAM, stops at the time limit,
stops if free disk drops under 5 GB, keeps one small checkpoint (~25 MB).
"""
import math, os, shutil, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
DATA, CKPT = HERE / "data", HERE / "checkpoints" / "lipi.pt"
PAD, BOS, EOS = 0, 1, 2
MAX_LEN = 104
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def read(split):
    return [l.rstrip("\n").split("\t") for l in (DATA / f"{split}.tsv").open()]


class Vocab:
    def __init__(self, chars):
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def enc(self, s):
        return [BOS] + [self.stoi.get(c, 3) for c in s] + [EOS]

    def dec(self, ids):
        return "".join(self.itos[i] for i in ids if i > 3)


class Lipi(nn.Module):
    def __init__(self, n_src, n_tgt, d=256, heads=4, layers=3, ff=1024, drop=0.1):
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(n_src, d), nn.Embedding(n_tgt, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.tf = nn.Transformer(d, heads, layers, layers, ff, drop, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, n_tgt)
        self.d = d

    def embed(self, emb, x):
        return emb(x) * math.sqrt(self.d) + self.pos(torch.arange(x.size(1), device=x.device))

    def forward(self, src, tgt_in):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        h = self.tf(self.embed(self.src_emb, src), self.embed(self.tgt_emb, tgt_in), tgt_mask=mask,
                    src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD,
                    memory_key_padding_mask=src == PAD, tgt_is_causal=True)
        return self.out(h)

    @torch.no_grad()
    def convert(self, texts, sv, tv):
        """Greedy decoding, a batch at a time."""
        self.eval()
        src = pad([sv.enc(t[:MAX_LEN - 2]) for t in texts]).to(DEV)
        mem = self.tf.encoder(self.embed(self.src_emb, src), src_key_padding_mask=src == PAD)
        ys = torch.full((len(texts), 1), BOS, device=DEV)
        done = torch.zeros(len(texts), dtype=torch.bool, device=DEV)
        for _ in range(MAX_LEN - 1):
            mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=DEV)
            h = self.tf.decoder(self.embed(self.tgt_emb, ys), mem, tgt_mask=mask,
                                memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, -1]).argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        outs = []
        for row in ys[:, 1:].tolist():
            outs.append(tv.dec(row[:row.index(EOS)] if EOS in row else row))
        return outs


def pad(seqs):
    out = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def cer(pred, gold):
    """Character error rate: edits needed / length of the right answer."""
    prev = list(range(len(gold) + 1))
    for i, p in enumerate(pred, 1):
        cur = [i]
        for j, g in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != g)))
        prev = cur
    return prev[-1] / max(len(gold), 1)


def load():
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    sv, tv = Vocab([]), Vocab([])
    sv.itos, tv.itos = ck["src_itos"], ck["tgt_itos"]
    sv.stoi, tv.stoi = {c: i for i, c in enumerate(sv.itos)}, {c: i for i, c in enumerate(tv.itos)}
    m = Lipi(len(sv.itos), len(tv.itos)).to(DEV)
    m.load_state_dict(ck["model"])
    return m, sv, tv, ck


def train(minutes):
    os.nice(10)  # be polite to the rest of the Mac
    if DEV == "mps":
        torch.mps.set_per_process_memory_fraction(0.4)
    torch.manual_seed(0)
    tr, va = read("train"), read("val")
    sv, tv = Vocab({c for r, _ in tr for c in r}), Vocab({c for _, b in tr for c in b})
    m = Lipi(len(sv.itos), len(tv.itos)).to(DEV)
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M weights | {len(tr)} train pairs | device {DEV}")
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.01)
    sched = lambda s: min(1, s / 1000)  # warm-up
    tr.sort(key=lambda p: len(p[0]))  # similar lengths per batch = less padding
    batches = [tr[i:i + 128] for i in range(0, len(tr), 128)]
    CKPT.parent.mkdir(exist_ok=True)
    log = (HERE / "checkpoints" / "train_log.tsv").open("w")
    log.write("step\tminutes\ttrain_loss\tval_cer\n")
    t0, step, best = time.time(), 0, 9.9
    g = torch.Generator().manual_seed(0)
    while True:
        for bi in torch.randperm(len(batches), generator=g).tolist():
            b = batches[bi]
            src = pad([sv.enc(r) for r, _ in b]).to(DEV)
            tgt = pad([tv.enc(t) for _, t in b]).to(DEV)
            logits = m.train()(src, tgt[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            frac = min((time.time() - t0) / 60 / minutes, 1)
            for grp in opt.param_groups:  # warm-up, then cosine decay to 5% over the time budget
                grp["lr"] = 5e-4 * sched(step) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
            opt.step(); step += 1
            mins = (time.time() - t0) / 60
            if step % 1000 == 0 or mins >= minutes:
                sample = va[:300]
                preds = m.convert([r for r, _ in sample], sv, tv)
                v = sum(cer(p, b) for p, (_, b) in zip(preds, sample)) / len(sample)
                print(f"step {step:6d} | {mins:5.1f} min | loss {loss.item():.3f} | val CER {v:.3f} | "
                      f"'{sample[0][0]}' -> '{preds[0]}'", flush=True)
                log.write(f"{step}\t{mins:.2f}\t{loss.item():.4f}\t{v:.4f}\n"); log.flush()
                if v < best:
                    best = v
                    torch.save({"model": m.state_dict(), "src_itos": sv.itos, "tgt_itos": tv.itos,
                                "step": step, "val_cer": v}, CKPT)
                if shutil.disk_usage(HERE).free < 5e9:
                    sys.exit("stopped: less than 5 GB disk free")
            if mins >= minutes:
                print(f"time limit reached. best val CER {best:.3f}, saved {CKPT}")
                return


def evaluate():
    import avro
    m, sv, tv, ck = load()
    te = read("test")
    preds = []
    for i in range(0, len(te), 256):
        preds += m.convert([r for r, _ in te[i:i + 256]], sv, tv)
    av = [avro.parse(r) for r, _ in te]
    c_lipi = [cer(p, b) for p, (_, b) in zip(preds, te)]
    c_avro = [cer(a, b) for a, (_, b) in zip(av, te)]
    ours, base = sum(c_lipi) / len(te), sum(c_avro) / len(te)
    ex_l = sum(p == b for p, (_, b) in zip(preds, te)) / len(te)
    ex_a = sum(a == b for a, (_, b) in zip(av, te)) / len(te)
    wins = sum(l < a for l, a in zip(c_lipi, c_avro)) / len(te)
    short = [i for i, (r, _) in enumerate(te) if len(r) <= 40]
    s_l, s_a = (sum(c[i] for i in short) / len(short) for c in (c_lipi, c_avro))
    lines = [f"# Test results (step {ck['step']}, {len(te)} pairs, sentences never seen in training)", "",
             "| | Avro phonetic | Lipi |", "|---|---|---|",
             f"| Character error rate (all) | {base:.3f} | {ours:.3f} |",
             f"| Character error rate (short, <= 40 chars, {len(short)} pairs) | {s_a:.3f} | {s_l:.3f} |",
             f"| Exact sentences | {ex_a:.1%} | {ex_l:.1%} |",
             f"| Sentences where Lipi beats Avro | | {wins:.1%} |", "", "## Examples", ""]
    for i in list(range(0, len(te), max(len(te) // 12, 1)))[:12]:
        lines += [f"- `{te[i][0]}`", f"  - Avro: {av[i]}", f"  - Lipi: {preds[i]}", f"  - Right answer: {te[i][1]}"]
    out = "\n".join(lines) + "\n"
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "eval.md").write_text(out)
    print(out)


if __name__ == "__main__":
    assert cer("abc", "abc") == 0 and cer("abd", "abc") == 1 / 3  # self-check
    if sys.argv[1:2] == ["train"]:
        train(float(sys.argv[3]) if len(sys.argv) > 3 else 15)
    elif sys.argv[1:2] == ["eval"]:
        evaluate()
    else:
        m, sv, tv, _ = load()
        print(m.convert([" ".join(sys.argv[1:])], sv, tv)[0])

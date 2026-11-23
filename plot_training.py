"""Draw the training curve from checkpoints/train_log.tsv into results/training_curve.png."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = [l.split("\t") for l in (HERE / "checkpoints" / "train_log.tsv").read_text().splitlines()[1:]]
mins, loss, cer = ([float(r[i]) for r in rows] for i in (1, 2, 3))

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(mins, cer, color="#690AFF", lw=2.5, label="Lipi, validation")
# Avro on the same 300 validation pairs the training loop scores
import avro
from lipi import read, cer as char_err
va = read("val")[:300]
avro_cer = sum(char_err(avro.parse(r), b) for r, b in va) / len(va)
ax.axhline(avro_cer, color="#999", ls="--", lw=1.5, label=f"Avro phonetic rules ({avro_cer:.3f})")
ax.set(xlabel="Training time on a 16 GB Mac (minutes)", ylabel="Character error rate (lower is better)",
       title="Lipi learning Banglish to Bangla", ylim=(0, min(1.0, max(cer) * 1.05)))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
(HERE / "results").mkdir(exist_ok=True)
fig.savefig(HERE / "results" / "training_curve.png")
print("saved results/training_curve.png")

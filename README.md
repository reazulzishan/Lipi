# Lipi: Banglish to Bangla

Lipi turns Bangla written in English letters ("ami tomake bhalobashi") into Bangla script (আমি তোমাকে ভালবাসি). It's a small character-level Transformer, 5.6 million parameters, trained from scratch on a laptop in 150 minutes.

On 6,523 test sentences it never saw in training, it makes 0.188 character errors per character, against 0.278 for Avro phonetic (the rule-based converter most people type with). It gets 8.2% of sentences exactly right, against Avro's 0.8%, and beats Avro on 78.7% of them. The full write-up is in [REPORT.md](REPORT.md).

```
$ python lipi.py "kal ki class ache?"
কাল কি ক্লাস আছে?
```

## Why it does better than rules

Avro maps letters to letters, so it can't tell that "kemon" should be কেমন and not কেমন্, or that "ashsen" is আসছেন. Lipi learns those spellings from examples:

| Banglish | Avro | Lipi |
|---|---|---|
| apnara kothay theke ashsen? | আপ্নারা কথায় থেকে আশসেন? | আপনারা কোথায় থেকে আসেন? |
| amar shondeho ase | আমার শন্দেহ আসে | আমার সন্দেহ আছে |
| guli koiro na! | গুলি কইর না! | গুলি করো না! |

## Run it

```bash
pip install -r requirements.txt
python prepare_data.py          # downloads the pairs and makes train / val / test
python lipi.py train 150        # trains for 150 minutes (about 4.4 steps a second on an M-series Mac)
python lipi.py eval             # writes results/eval.md
python lipi.py "apni kemon achen"
```

The trained model (`lipi.pt`) is attached to the [latest release](https://github.com/reazulzishan/Lipi/releases). Put it in `checkpoints/` to skip training.

Training is gentle on a laptop: it runs at low priority, caps GPU memory at 40% and stops if free disk space drops under 5 GB.

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads the dataset, repairs broken conjuncts, drops pairs that are translations rather than transliterations, splits by Bangla sentence |
| `lipi.py` | The model, training, evaluation and conversion |
| `plot_training.py` | Draws `results/training_curve.png` from the training log |
| `results/` | Test results, the training curve and an early 30-minute test run |

## Data and licence

Data: [istiaqfuad/bangla-english-banglish-pairs](https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs), CC BY 4.0. Code: MIT, see [LICENSE](LICENSE).

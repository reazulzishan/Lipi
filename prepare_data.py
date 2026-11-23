"""Download the Banglish -> Bangla pairs and split them into train / val / test.

Source: istiaqfuad/bangla-english-banglish-pairs (CC BY 4.0), file banglish_pairs_filtered.jsonl.
Each Bangla sentence comes with up to 3 Banglish spellings (standard, alternate, short).
The split is by Bangla sentence, so no test sentence is ever seen in training, in any spelling.
"""
import json, random, re, urllib.request
import avro  # only used to filter out translation pairs
from pathlib import Path

URL = ("https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs/"
       "resolve/main/banglish_pairs_filtered.jsonl")
DATA = Path(__file__).parent / "data"
BN, LAT = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")
CLEAN_BN = re.compile(r"[ঀ-৿।॥‌‍0-9 .,!?;:'\"()\-–—‘’“”%&/@#]+")  # Bangla + punctuation only
CLEAN_RM = re.compile(r"[ -~‘’“”–—]+")  # plain keyboard text
MAX_LEN = 100  # characters; keeps training fast on a laptop
# Vowel signs and hasanta removed before the filter check, so spelling style ("ki"/"কি" vs "কী") does not count.
LOOSE = re.compile(r"[া-্ঁ-ঃ\s।.,!?'\"-]")


def loose_cer(a, b):
    a, b = LOOSE.sub("", a), LOOSE.sub("", b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(len(b), 1)


def main():
    DATA.mkdir(exist_ok=True)
    raw = DATA / "banglish_pairs.jsonl"
    if not raw.exists():
        print("downloading (179 MB)...")
        urllib.request.urlretrieve(URL, raw)

    by_bn, dropped = {}, 0
    for line in raw.open():
        d = json.loads(line)
        bn, rm = (" ".join(d[k].split()) for k in ("anchor", "positive"))  # no tabs/newlines inside
        bn = re.sub(r"\s*্\s*", "্", bn)  # repair "বিশ ্ বাস" -> "বিশ্বাস" (broken conjuncts in the source)
        if (CLEAN_BN.fullmatch(bn) and CLEAN_RM.fullmatch(rm) and BN.search(bn) and LAT.search(rm) and not BN.search(rm)
                and len(bn) <= MAX_LEN and len(rm) <= MAX_LEN and 0.6 < len(rm) / len(bn) < 1.7):
            # drop translations posing as transliterations ("painful punishment" -> "মর্মন্তুদ শাস্তি"):
            # if even a letter-by-letter phonetic reading is mostly wrong, the pair is not a transliteration
            if loose_cer(avro.parse(rm), bn) <= 0.6:
                by_bn.setdefault(bn, set()).add(rm)
            else:
                dropped += 1

    sents = sorted(by_bn)
    random.Random(0).shuffle(sents)
    n = len(sents)
    splits = {"test": sents[:2000], "val": sents[2000:3000], "train": sents[3000:]}
    for name, part in splits.items():
        with (DATA / f"{name}.tsv").open("w") as f:
            for bn in part:
                for rm in sorted(by_bn[bn]):
                    f.write(f"{rm}\t{bn}\n")
    assert not set(splits["test"]) & set(splits["train"]), "test leaked into train"
    print(f"dropped {dropped} translation-like pairs")
    print(f"{n} Bangla sentences -> " + ", ".join(
        f"{k}: {sum(len(by_bn[b]) for b in v)} pairs" for k, v in splits.items()))


if __name__ == "__main__":
    main()

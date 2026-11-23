# Lipi: report

## Question

Can a small model, trained on one laptop, convert Banglish to Bangla script better than Avro phonetic, the rule-based converter most Bangla typists use?

## Data

The [bangla-english-banglish-pairs](https://huggingface.co/datasets/istiaqfuad/bangla-english-banglish-pairs) dataset (CC BY 4.0). Each Bangla sentence comes with up to three Banglish spellings: standard, alternate and short.

An early 30-minute run showed two problems in the data, and `prepare_data.py` fixes both:

- Some Bangla answers had broken conjuncts, such as "বিশ ্ বাস" with stray spaces. These are repaired.
- Some pairs were translations, not transliterations ("painful punishment" next to "মর্মন্তুদ শাস্তি"). Any pair where Avro's output and the answer differ too much, after ignoring vowel signs, is dropped. That removed 9,335 pairs.

After cleaning and keeping sentences up to 100 characters: 130,746 training pairs, 3,320 validation pairs and 6,523 test pairs. The split is by Bangla sentence, so no test sentence appears in training in any of its spellings.

## Method

- Character-level encoder-decoder Transformer: 3 encoder and 3 decoder layers, width 256, 4 attention heads, 5.6 million parameters.
- AdamW (learning rate 5e-4, weight decay 0.01), 1,000 warm-up steps then cosine decay, label smoothing 0.1, batches of 128 sorted by length.
- Trained for 150 minutes (26,728 steps) on a 16 GB M-series Mac. The checkpoint with the lowest validation error is kept.
- Metric: character error rate (CER), the edit distance divided by the length of the right answer. Lower is better.

## Results

Test set, 6,523 sentences never seen in training:

| | Avro phonetic | Lipi |
|---|---|---|
| Character error rate, all sentences | 0.278 | 0.188 |
| Character error rate, short sentences (40 characters or fewer, 3,907) | 0.272 | 0.167 |
| Sentences exactly right | 0.8% | 8.2% |
| Sentences where Lipi beats Avro | | 78.7% |

Validation error fell from 1.16 at step 1,000 to 0.199 at the end. It flattened after about two hours, so longer training on the same data would add little. The curve is in `results/training_curve.png` and more examples are in `results/eval.md`.

## Limits

- 8.2% exact sentences is still low. Most errors are one or two characters, such as সম্পান্য for সম্পন্ন.
- The model sometimes adds a full stop (।) that wasn't in the input, because many training answers end with one.
- Some test answers are themselves loose translations ("barra pathano hobe naki" against "বার্তা প্রেরণ করা হবে কি"), so the true error is a little lower than the table shows.
- Sentences longer than 100 characters were left out of training, so very long input is less reliable.

## Next steps

- Clean the remaining translation-style pairs with a stricter filter, then retrain.
- Stop the model from adding punctuation the input didn't have.
- Try a larger model now that the data pipeline is fixed.
- Package it as a keyboard or browser helper.

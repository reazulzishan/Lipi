# Test run (30 minutes, before data cleaning)

Date: 2026-11-21. Data: 138,457 training pairs (no conjunct repair, no translation filter).
Test set: 6,922 pairs from sentences never seen in training.

| System | Character error rate (lower is better) |
|---|---|
| Avro phonetic (rules) | 0.306 |
| Lipi after 30 min (step 4,500) | 0.451 |

What it showed:
- Short everyday sentences already worked: "shoyar kemn acen?" -> সয়ার কেমন আছেন? (exact), where Avro gave শয়ার কেম্ন আচেন?
- Long sentences failed, for two data reasons: answers with broken conjuncts ("বিশ ্ বাস") and pairs that were translations, not transliterations ("painful punishment" -> "মর্মন্তুদ শাস্তি").
- Both were fixed in prepare_data.py before the full run: conjuncts repaired, 9,335 translation-like pairs dropped.
- Speed on a 16 GB M-series Mac (MPS): about 4.4 training steps per second, one pass over the data in about 4 minutes.

# Test results (step 26728, 6523 pairs, sentences never seen in training)

| | Avro phonetic | Lipi |
|---|---|---|
| Character error rate (all) | 0.278 | 0.188 |
| Character error rate (short, <= 40 chars, 3907 pairs) | 0.272 | 0.167 |
| Exact sentences | 0.8% | 8.2% |
| Sentences where Lipi beats Avro | | 78.7% |

## Examples

- `gosol er ayojon koro!`
  - Avro: গসল এর আয়জন কর!
  - Lipi: গোসলের আয়োজন করো!
  - Right answer: গোসলের আয়োজন করো!
- `badike punorai bitrin korun`
  - Avro: বাদিকে পুনরাই বিত্রিন করুন
  - Lipi: বাদিকে পুনরায় বিত্রিন করুন
  - Right answer: বাঁদিকে পুনরায় বিতরণ করুন:
- `ei operation shompanno korar jonno apnar on-line e thaka aboshsho`
  - Avro: এই অপেরাতিওন শম্পান্ন করার জন্ন আপ্নার অন-লিনে এ থাকা আবশশ
  - Lipi: এই অপারেশন সম্পান্য করার জন্য আপনার অনলাইনে থাকা বসহসে।
  - Right answer: এই অপারেশন সম্পন্ন করার জন্য আপনার অন-লাইনে থাকা আবশ্যক
- `asol e onekei khuje`
  - Avro: আসল এ অনেকেই খুজে
  - Lipi: আসলে অনেকেই খুঁজে।
  - Right answer: আসলে অনেকেই খুজছে।
- `apnara kothay theke ashsen?`
  - Avro: আপ্নারা কথায় থেকে আশসেন?
  - Lipi: আপনারা কোথায় থেকে আসেন?
  - Right answer: - আপনারা কোথা থেকে আসছেন?
- `chobi: komay`
  - Avro: ছবিঃ কমায়
  - Lipi: ছবিঃ কমায়
  - Right answer: ছবি: কোমায়ে
- `amar shondeho ase tara tomake ki poyna porishod kore!`
  - Avro: আমার শন্দেহ আসে তারা তমাকে কি পয়না পরিশদ করে!
  - Lipi: আমার সন্দেহ আছে তারা তোমাকে কি পয়না পরিষদ করে!
  - Right answer: আমার সন্দেহ আছে তারা তোমাকে কি পাওনা পরিশোধ করে!
- `apel ar badam er pie`
  - Avro: আপেল আর বাদাম এর পিএ
  - Lipi: আপেল আর বাদামের পেয়ে।
  - Right answer: আপেল আর বাদামের পাই
- `mailing-liste barra pathano hobe naki?`
  - Avro: মাইলিং-লিস্তে বাররা পাথান হবে নাকি?
  - Lipi: মেইলিং-লিস্টে বারা পাঠানো হবে নাকি?
  - Right answer: মেইলিং-লিস্টে বার্তা প্রেরণ করা হবে কি?
- `guli koiro na!`
  - Avro: গুলি কইর না!
  - Lipi: গুলি করো না!
  - Right answer: গুলি কোরো না!
- `jamal, edik as`
  - Avro: জামাল, এদিক আস
  - Lipi: জামাল, এদিক আসো।
  - Right answer: জামাল, এদিকে আয়।
- `ei site er jonno & kdapi noy`
  - Avro: এই সিতে এর জন্ন & কদাপি নয়
  - Lipi: এই সাইটের জন্য & কোড নয়।
  - Right answer: এই সাইটের জন্য & কদাপি নয়

# Sentences Script for Voice Recordings

This document contains the exact 10 training sentences and 2 test sentences designed for voice recording. Each sentence includes at least one named entity covering **Person (`PER`)**, **Organization (`ORG`)**, and **Location (`LOC`)**, with a mix of single-token and multi-token entities to train and evaluate both `B-` (Beginning) and `I-` (Inside) tags.

---

## 🎙️ Guidelines for Recording Your Audio

1. **Format:** Either **`.mp3`** or **`.wav`** format is completely fine! AssemblyAI and our pipeline support both natively.
1. **Audio Quality:** Speak clearly in a quiet environment at a natural pace.
1. **Naming Convention:**
   - Training sentences: `audio/train/train_01.mp3` (or `.wav`) to `audio/train/train_10.mp3` (or `.wav`)
   - Test sentences: `audio/test/test_01.mp3` (or `.wav`) and `audio/test/test_02.mp3` (or `.wav`)

---

## 1. Training Sentences (10 Recordings)

### Sentence 1 (`train_01`)
>
> **"Elon Musk founded SpaceX in California."**

- **Entities:**
  - `Elon`: `B-PER`
  - `Musk`: `I-PER`
  - `SpaceX`: `B-ORG`
  - `California`: `B-LOC`

### Sentence 2 (`train_02`)
>
> **"Satya Nadella leads Microsoft from Seattle."**

- **Entities:**
  - `Satya`: `B-PER`
  - `Nadella`: `I-PER`
  - `Microsoft`: `B-ORG`
  - `Seattle`: `B-LOC`

### Sentence 3 (`train_03`)
>
> **"Sundar Pichai visited Google headquarters in London."**

- **Entities:**
  - `Sundar`: `B-PER`
  - `Pichai`: `I-PER`
  - `Google`: `B-ORG`
  - `London`: `B-LOC`

### Sentence 4 (`train_04`)
>
> **"Tim Cook announced Apple products in New York."**

- **Entities:**
  - `Tim`: `B-PER`
  - `Cook`: `I-PER`
  - `Apple`: `B-ORG`
  - `New`: `B-LOC`
  - `York`: `I-LOC`

### Sentence 5 (`train_05`)
>
> **"Jeff Bezos established Amazon in Washington."**

- **Entities:**
  - `Jeff`: `B-PER`
  - `Bezos`: `I-PER`
  - `Amazon`: `B-ORG`
  - `Washington`: `B-LOC`

### Sentence 6 (`train_06`)
>
> **"Marie Curie conducted groundbreaking research at Sorbonne University in Paris."**

- **Entities:**
  - `Marie`: `B-PER`
  - `Curie`: `I-PER`
  - `Sorbonne`: `B-ORG`
  - `University`: `I-ORG`
  - `Paris`: `B-LOC`

### Sentence 7 (`train_07`)
>
> **"Mark Zuckerberg manages Meta across North America."**

- **Entities:**
  - `Mark`: `B-PER`
  - `Zuckerberg`: `I-PER`
  - `Meta`: `B-ORG`
  - `North`: `B-LOC`
  - `America`: `I-LOC`

### Sentence 8 (`train_08`)
>
> **"Sam Altman directs OpenAI in San Francisco."**

- **Entities:**
  - `Sam`: `B-PER`
  - `Altman`: `I-PER`
  - `OpenAI`: `B-ORG`
  - `San`: `B-LOC`
  - `Francisco`: `I-LOC`

### Sentence 9 (`train_09`)
>
> **"Jensen Huang unveiled Nvidia chips in Taiwan."**

- **Entities:**
  - `Jensen`: `B-PER`
  - `Huang`: `I-PER`
  - `Nvidia`: `B-ORG`
  - `Taiwan`: `B-LOC`

### Sentence 10 (`train_10`)
>
> **"Demis Hassabis develops artificial intelligence for DeepMind in Cambridge."**

- **Entities:**
  - `Demis`: `B-PER`
  - `Hassabis`: `I-PER`
  - `DeepMind`: `B-ORG`
  - `Cambridge`: `B-LOC`

---

## 2. Test Sentences (2 Recordings)

### Test Sentence 1 (`test_01`)
>
> **"Sundar Pichai discussed technology with Tesla in California."**

- **Entities:**
  - `Sundar`: `B-PER`
  - `Pichai`: `I-PER`
  - `Tesla`: `B-ORG`
  - `California`: `B-LOC`
- *Purpose:* Evaluates how well the model recognizes entities in a new combination of context.

### Test Sentence 2 (`test_02`)
>
> **"Dr. Kalam delivered an inspiring lecture at Oxford University in England."**

- **Entities:**
  - `Kalam`: `B-PER`
  - `Oxford`: `B-ORG`
  - `University`: `I-ORG`
  - `England`: `B-LOC`
- *Purpose:* Evaluates model behavior with unseen entities and Out-of-Vocabulary (OOV) tokens, demonstrating the challenges of small datasets.

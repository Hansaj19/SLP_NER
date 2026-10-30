# Technical Analysis Report: Named Entity Recognition on a Small, Self-Created Audio Dataset

## 1. Executive Summary
This project investigates the design, implementation, and evaluation of an end-to-end Named Entity Recognition (NER) system operating on speech input. The system couples a cloud-grade Speech-to-Text (STT) layer powered by **AssemblyAI** with a sequence-labeling Recurrent Neural Network (**SimpleRNN / GRU / Bidirectional LSTM**) implemented in **PyTorch**. The dataset comprises a self-recorded corpus of 10 training sentences and 2 test sentences annotated using the standard BIO tagging schema (`B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `O`).

This report provides an in-depth analysis of the technical hurdles encountered when training deep sequence models on extreme low-resource audio datasets.

---

## 2. Key Challenges in Audio-to-NER Pipelines

### 2.1 The Cascading Acoustic Error Effect
In a cascaded architecture ($\text{Audio} \xrightarrow{\text{STT}} \text{Text} \xrightarrow{\text{RNN}} \text{Tags}$), the downstream NER model is strictly dependent on the acoustic accuracy of the speech-to-text front-end:
- **Acoustic Misinterpretation:** Proper nouns (e.g., regional names like *Kalam*, or specialized company names like *DeepMind*) have lower prior probabilities in general-purpose language models. If an STT engine mistranscribes *"Kalam"* as *"column"*, the word vector shifts from an entity candidate to a common noun, causing the RNN to misclassify it as `O`.
- **Punctuation and Capitalization Drift:** Many traditional NER taggers rely heavily on uppercase capitalization as an indicator of named entities. If STT fails to output proper casing (or attaches punctuation such as `"California."`), tokenization and classification degrade.

### 2.2 The Out-of-Vocabulary (OOV) Bottleneck
When a model is trained on only 10 sentences, the vocabulary consists of roughly 60–70 unique word tokens. 
- During inference on new voice recordings (e.g., `test_02`), unseen words such as *"Oxford"*, *"England"*, and *"inspiring"* are encountered for the first time.
- In a closed-vocabulary RNN, these tokens are mapped to the generic `<UNK>` (Unknown) token index.
- If both an unseen entity (*"Oxford"*) and an unseen common word (*"inspiring"*) map to `<UNK>`, the model has no lexical features to distinguish between an organization and an ordinary adjective. It must rely entirely on surrounding syntactic context.

### 2.3 Severe Class Imbalance
In natural English text, non-entity words (`O`) constitute 75% to 85% of all tokens:
- With only 10 training sentences (~85 total tokens), the model sees very few positive instances of `B-LOC`, `I-LOC`, or `I-ORG`.
- An unweighted loss function naturally pushes the network toward a local minimum where predicting `O` for everything yields ~80% raw token accuracy while achieving 0% entity recall.

### 2.4 Overfitting vs. Memorization
Deep neural networks have high parameter capacity. When parameterized with 64-dimensional embeddings and 64 hidden units (~50,000 parameters) on a training corpus of under 100 tokens, the network quickly memorizes exact word-tag mappings rather than learning generalizable grammatical patterns. Consequently, familiar words in new syntactic arrangements may be misclassified.

---

## 3. Architecture Comparison: SimpleRNN vs. GRU vs. Bi-LSTM

| Metric / Dimension | SimpleRNN | Gated Recurrent Unit (GRU) | Bidirectional LSTM (BiLSTM) |
|---|---|---|---|
| **Gradient Flow** | Suffers from vanishing gradients over long sentences | Gating units (reset/update) prevent vanishing gradients | Separate cell state ($c_t$) maintains stable gradient highways |
| **Contextual Awareness** | Unidirectional (Past only) | Directional | **Bidirectional (Past + Future simultaneously)** |
| **Multi-token Entities** | Struggles with `I-` tags | Good boundary retention | **Superior boundary detection (`B-` followed by `I-`)** |
| **Training Stability** | Oscillates on small batches | Smooth convergence | Smooth, stable convergence |

**Why Bi-LSTM is the Preferred Choice:**
In sequence labeling, the label of a word often depends on the words that appear *after* it. For example, in *"Sorbonne University"*, knowing that *"University"* follows *"Sorbonne"* strongly hints that *"Sorbonne"* is the beginning of an organization (`B-ORG`). A unidirectional RNN reading left-to-right only sees *"Sorbonne"* without future context, whereas a Bi-LSTM reads both forward and backward.

---

## 4. Evaluation and Empirical Findings

Testing on the two held-out voice recordings revealed:
1. **Familiar Entity Generalization (`test_01`):**
   - The model accurately recognized `"Sundar Pichai"` (`B-PER`, `I-PER`) and `"California"` (`B-LOC`) because these tokens appeared in the training set.
   - However, `"Tesla"` was unseen in training and mapped to `<UNK>`. The model relied solely on the surrounding preposition `"with"` to infer an organization candidate.
2. **Unseen Entity and Boundary Challenge (`test_02`):**
   - Multi-token entity `"Oxford University"`: While `"University"` appeared in training, `"Oxford"` was unseen. The model struggled to anchor the beginning tag `B-ORG` on the unknown token, illustrating the OOV vulnerability.

---

## 5. Engineering Recommendations for Future Iterations

1. **Pre-trained Word Embeddings:**
   - Incorporating pre-trained GloVe (50d/100d) or FastText vectors provides semantic representations for words unseen in the small training split. FastText's subword n-grams would allow the model to recognize unseen morphological variants.
2. **Pre-trained Transformer Fine-Tuning:**
   - Utilizing a small pre-trained transformer (e.g., `distilbert-base-uncased` or `MobileBERT`) with subword tokenization largely eliminates the OOV issue.
3. **Acoustic Data Augmentation:**
   - Adding background noise, pitch shifting, and speed variation to the self-recorded audio increases speech-to-text robustness.


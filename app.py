"""
Streamlit Web Application for Voice-Driven Named Entity Recognition (NER).
Deployable directly to Streamlit Community Cloud.
"""
import os
import sys
import json
import tempfile
import streamlit as st
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.stt_service import STTService
from src.data_utils import tokenize_text, load_annotations, load_vocab, build_vocab, save_vocab, TAG2IDX
from src.train import run_training_pipeline, load_model_checkpoint
from src.evaluate import predict_sentence_tags, evaluate_test_set
from src.pipeline import extract_entities_from_bio

st.set_page_config(
    page_title="Voice NER System | RNN & AssemblyAI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .entity-badge {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .entity-per { background-color: #2563EB; color: white; }
    .entity-org { background-color: #059669; color: white; }
    .entity-loc { background-color: #D97706; color: white; }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4F46E5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def ensure_models_and_vocab():
    """Ensures at least one baseline checkpoint exists so app works out of the box."""
    vocab_path = "data/vocab.json"
    model_path = "models/ner_bilstm.pt"

    if not os.path.exists(vocab_path):
        data = load_annotations("data/annotations.json")
        w2i, _ = build_vocab(data["train"])
        save_vocab(w2i, vocab_path)

    if not os.path.exists(model_path):
        run_training_pipeline(cell_type="lstm", num_epochs=45, save_path=model_path)

ensure_models_and_vocab()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎙️ Audio NER Config")
    st.caption("AssemblyAI + PyTorch Recurrent Networks")
    st.markdown("---")

    model_choice = st.selectbox(
        "🧠 Select Model Architecture",
        ["Bidirectional LSTM (Recommended)", "Gated Recurrent Unit (GRU)", "SimpleRNN"],
        index=0
    )

    model_file_map = {
        "Bidirectional LSTM (Recommended)": ("lstm", "models/ner_bilstm.pt"),
        "Gated Recurrent Unit (GRU)": ("gru", "models/ner_gru.pt"),
        "SimpleRNN": ("rnn", "models/ner_rnn.pt")
    }
    cell_type, model_ckpt = model_file_map[model_choice]

    # Ensure selected model checkpoint exists
    if not os.path.exists(model_ckpt):
        with st.spinner(f"Training {model_choice} checkpoint..."):
            run_training_pipeline(cell_type=cell_type, num_epochs=45, save_path=model_ckpt)

    st.success(f"Loaded: `{os.path.basename(model_ckpt)}`")

    st.markdown("---")
    st.markdown("### 🔑 STT API Status")
    stt = STTService()
    if stt.is_api_configured:
        st.success("AssemblyAI API: Connected ✅")
    else:
        st.warning("AssemblyAI API: Using Mock Fallback ⚠️")
        st.caption("Set `ASSEMBLYAI_API_KEY` in `.env` or Streamlit Secrets.")

    st.markdown("---")
    st.info(
        "**BIO Schema Labels:**\n"
        "- `PER`: Person\n"
        "- `ORG`: Organization\n"
        "- `LOC`: Location\n"
        "- `O`: Outside / Non-entity"
    )

# --- MAIN PAGE TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Live Audio Inference",
    "📁 Dataset & Audio Corpus",
    "📊 Test Evaluation & Metrics",
    "📑 Challenge Analysis Report"
])

# --- TAB 1: LIVE AUDIO INFERENCE ---
with tab1:
    st.header("Voice-to-Named-Entity Recognition")
    st.write(
        "Record your voice or upload an audio file (`.mp3`, `.wav`, `.m4a`). "
        "The system will transcribe your speech via **AssemblyAI** and predict BIO tags using the **PyTorch RNN**."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Audio Input")
        input_mode = st.radio(
            "Choose Audio Input Mode:",
            ["Record from Microphone", "Upload Audio File", "Select from Pre-recorded Corpus"],
            horizontal=True
        )

        audio_bytes = None
        input_filename = "temp_input.wav"
        fallback_text = None

        if input_mode == "Record from Microphone":
            # Streamlit audio input widget
            mic_audio = st.audio_input("Click to record your voice:")
            if mic_audio:
                audio_bytes = mic_audio.read()
                input_filename = "recorded_mic.wav"

        elif input_mode == "Upload Audio File":
            uploaded_file = st.file_uploader("Upload audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])
            if uploaded_file:
                audio_bytes = uploaded_file.read()
                input_filename = uploaded_file.name

        else: # Select from pre-recorded corpus
            corpus_files = []
            for split in ["train", "test"]:
                split_dir = os.path.join("audio", split)
                if os.path.exists(split_dir):
                    for f in os.listdir(split_dir):
                        if f.endswith((".wav", ".mp3", ".m4a")):
                            corpus_files.append((f"{split}/{f}", os.path.join(split_dir, f)))

            if corpus_files:
                selected_label, selected_path = st.selectbox(
                    "Choose sample recording:",
                    corpus_files,
                    format_func=lambda x: x[0]
                )
                with open(selected_path, "rb") as f:
                    audio_bytes = f.read()
                input_filename = selected_path
            else:
                st.info("No recordings found in `audio/train` or `audio/test`. You can record or upload one above!")
                # Provide quick test text
                fallback_text = st.text_input(
                    "Or type a test sentence:",
                    value="Sundar Pichai discussed technology with Tesla in California."
                )

        if audio_bytes:
            st.audio(audio_bytes)

    with col2:
        st.subheader("2. Pipeline Execution")
        run_btn = st.button("🚀 Transcribe & Extract Named Entities", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Processing speech and running neural model..."):
                temp_path = None
                if audio_bytes:
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, os.path.basename(input_filename))
                    with open(temp_path, "wb") as f:
                        f.write(audio_bytes)
                else:
                    temp_path = "mock.wav"

                # 1. Transcribe
                stt_res = stt.transcribe(temp_path, fallback_text=fallback_text)
                transcribed_text = stt_res.get("text", "")

                # 2. Tokenize
                tokens = tokenize_text(transcribed_text)

                # 3. Predict Tags
                w2i, _ = load_vocab("data/vocab.json")
                model = load_model_checkpoint(model_ckpt)
                pred_tags = predict_sentence_tags(model, tokens, w2i)

                # 4. Extract Entities
                entities = extract_entities_from_bio(tokens, pred_tags)

            st.markdown("### 🗣️ Transcribed Speech")
            st.info(f"**\"{transcribed_text}\"** (Confidence: {stt_res.get('confidence', 1.0):.1%})")

            st.markdown("### 🏷️ Identified Named Entities")
            if entities:
                html_badges = ""
                for ent in entities:
                    ent_class = f"entity-{ent['type'].lower()}"
                    html_badges += f'<span class="entity-badge {ent_class}"><b>{ent["entity"]}</b> ({ent["type"]})</span> '
                st.markdown(html_badges, unsafe_allow_html=True)
            else:
                st.warning("No named entities identified.")

            st.markdown("### 🔍 Token-by-Token Sequence Classification")
            df = pd.DataFrame({
                "Token": tokens,
                "Predicted BIO Tag": pred_tags,
                "Entity Category": [
                    "Person" if "PER" in t else "Organization" if "ORG" in t else "Location" if "LOC" in t else "None (O)"
                    for t in pred_tags
                ]
            })
            st.dataframe(df, use_container_width=True)

# --- TAB 2: DATASET & AUDIO CORPUS ---
with tab2:
    st.header("Dataset & Spoken Sentences Corpus")
    st.write("Browse the 10 training sentences and 2 test sentences with ground-truth BIO annotations.")

    annot_data = load_annotations("data/annotations.json")
    train_items = annot_data.get("train", [])
    test_items = annot_data.get("test", [])

    st.subheader("Training Sentences (10 Recordings)")
    for i, item in enumerate(train_items, 1):
        with st.expander(f"Sentence {i:02d}: \"{item['text']}\""):
            c1, c2 = st.columns([1, 2])
            with c1:
                # Check for audio file
                audio_file = None
                for ext in [".wav", ".mp3", ".m4a"]:
                    cand = os.path.join("audio", "train", f"{item['id']}{ext}")
                    if os.path.exists(cand):
                        audio_file = cand
                        break
                if audio_file:
                    st.audio(audio_file)
                else:
                    st.caption("⚠️ Audio not uploaded yet (`audio/train/" + item['id'] + ".mp3`)")
            with c2:
                tokens_df = pd.DataFrame({"Token": item["tokens"], "Tag": item["tags"]}).T
                st.dataframe(tokens_df, use_container_width=True)

    st.subheader("Test Sentences (2 Recordings)")
    for i, item in enumerate(test_items, 1):
        with st.expander(f"Test Sentence {i:02d}: \"{item['text']}\""):
            c1, c2 = st.columns([1, 2])
            with c1:
                audio_file = None
                for ext in [".wav", ".mp3", ".m4a"]:
                    cand = os.path.join("audio", "test", f"{item['id']}{ext}")
                    if os.path.exists(cand):
                        audio_file = cand
                        break
                if audio_file:
                    st.audio(audio_file)
                else:
                    st.caption("⚠️ Audio not uploaded yet (`audio/test/" + item['id'] + ".mp3`)")
            with c2:
                tokens_df = pd.DataFrame({"Token": item["tokens"], "Tag": item["tags"]}).T
                st.dataframe(tokens_df, use_container_width=True)

# --- TAB 3: TEST EVALUATION & METRICS ---
with tab3:
    st.header("Evaluation on Test Recordings")
    st.write("Compare model predictions directly against actual ground-truth BIO annotations on the test set.")

    if st.button("Run Full Evaluation Benchmark", type="secondary"):
        with st.spinner("Computing precision, recall, and F1 across test recordings..."):
            eval_res = evaluate_test_set(model_path=model_ckpt)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Chunk Precision", f"{eval_res['precision']:.2%}")
        col_m2.metric("Chunk Recall", f"{eval_res['recall']:.2%}")
        col_m3.metric("Chunk F1-Score", f"{eval_res['f1']:.2%}")

        st.subheader("Classification Report (Seqeval / CoNLL Standard)")
        st.code(eval_res["report"])

        for rec in eval_res["records"]:
            st.markdown(f"#### Results for `{rec['id']}`")
            comp_df = pd.DataFrame({
                "Token": rec["tokens"],
                "Actual Tag": rec["true_tags"],
                "Predicted Tag": rec["pred_tags"],
                "Match": ["✅ Match" if t == p else "❌ Mismatch" for t, p in zip(rec["true_tags"], rec["pred_tags"])]
            })
            st.dataframe(comp_df, use_container_width=True)

# --- TAB 4: CHALLENGE ANALYSIS REPORT ---
with tab4:
    st.header("Technical Challenge Analysis")
    report_file = "docs/analysis_report.md"
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            report_md = f.read()
        st.markdown(report_md)
    else:
        st.info("Analysis report available in `docs/analysis_report.md`.")


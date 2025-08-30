import streamlit as st
import os, pathlib
from dotenv import load_dotenv
from qa import QASystem

load_dotenv()

st.set_page_config(page_title="EduSage", layout="centered")
st.title("📘 EduSage – Your Personal AI Tutor for PDFs")
st.caption("From Documents to Insights – Instantly")


pathlib.Path("uploads").mkdir(exist_ok=True)
pathlib.Path("outputs").mkdir(exist_ok=True)


if "qa" not in st.session_state:
    st.session_state.qa = QASystem()
if "history" not in st.session_state:
    st.session_state.history = [] 

qa = st.session_state.qa


with st.sidebar:
    st.header("Upload PDFs")
    pdfs = st.file_uploader("Select academic PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Index PDFs", type="primary", use_container_width=True, disabled=not pdfs):
        for up in pdfs:
            save_path = os.path.join("uploads", up.name)
            with open(save_path, "wb") as f:
                f.write(up.read())
            qa.index_pdf(save_path)
        st.success("✅ PDFs indexed successfully")

    st.divider()
    st.header("More Features")
    if st.button("📑 Summarize PDF(s)", use_container_width=True):
        with st.spinner("Summarizing..."):
            summary = qa.summarize_all()
        st.session_state.summary = summary  

    if st.button("🃏 Generate Knowledge Bites", use_container_width=True):
        with st.spinner("Generating Knowledge Bites..."):
            cards = qa.flashcards()
        st.session_state.cards = cards  


st.subheader("Q&A")
for i, turn in enumerate(st.session_state.history, start=1):
    st.markdown(f"**Q{i}: {turn['q']}**")
    st.markdown(turn["a"])
    st.markdown("---")


with st.form("ask_form", clear_on_submit=True):
    q = st.text_input("Ask an education-related question from the uploaded PDFs:", key="question_box")
    submit = st.form_submit_button("Get Answer")
if submit and q.strip():
    with st.spinner("Thinking..."):
        a = qa.answer(q.strip())
    st.session_state.history.append({"q": q.strip(), "a": a})
    st.rerun()


if "summary" in st.session_state:
    st.subheader("📑 Summary")
    st.write(st.session_state.summary)

if "cards" in st.session_state:
    st.subheader("🃏 Knowledge Bites")
    for i, c in enumerate(st.session_state.cards, 1):
        st.markdown(f"**Q{i}:** {c['question']}")
        st.markdown(f"**A{i}:** {c['answer']}")

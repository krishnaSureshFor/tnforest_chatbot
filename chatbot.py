def run():
    import streamlit as st
    import os
    import glob
    import requests
    import numpy as np
    import docx
    import PyPDF2
    from sklearn.feature_extraction.text import TfidfVectorizer

    # CONFIG
    DATA_DIR = "data/docs"
    TOP_K = 4
    CHUNK_SIZE = 700
    CHUNK_OVERLAP = 150

    HF_API_KEY = st.secrets.get("HF_API_KEY", None)
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

    HF_EMBED_API = "https://router.huggingface.co/hf-inference"
    GROQ_CHAT_API = "https://api.groq.com/openai/v1/chat/completions"

    # SAFETY CHECKS
    if HF_API_KEY is None:
        st.error("HF_API_KEY missing in secrets.")
        return

    if GROQ_API_KEY is None:
        st.error("GROQ_API_KEY missing in secrets.")
        return

    # LOAD FILES
    def load_files():
        files = glob.glob(f"{DATA_DIR}/*")
        texts = []
        if len(files) == 0:
            st.error("No documents inside data/docs/")
            return []

        for fp in files:
            if fp.lower().endswith(".pdf"):
                try:
                    with open(fp, "rb") as f:
                        reader = PyPDF2.PdfReader(f, strict=False)
                        txt = ""
                        for p in reader.pages:
                            t = p.extract_text()
                            if t:
                                txt += t + "\n"
                    texts.append((os.path.basename(fp), txt))
                except:
                    st.error(f"PDF error: {fp}")

            elif fp.lower().endswith(".docx"):
                try:
                    doc = docx.Document(fp)
                    txt = "\n".join([p.text for p in doc.paragraphs])
                    texts.append((os.path.basename(fp), txt))
                except:
                    st.error(f"DOCX error: {fp}")

            else:
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append((os.path.basename(fp), f.read()))
                except:
                    st.error(f"Text file error: {fp}")

        return texts

    # CHUNKING
    def chunk(text):
        parts = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            parts.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return parts

    # EMBEDDINGS
    def embed(texts):
        try:
            r = requests.post(
                HF_EMBED_API,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "inputs": texts
                },
                timeout=60
            )
            if r.status_code != 200:
                st.error(f"HF API error {r.status_code}: {r.text}")
                return None
            return np.array(r.json())

        except Exception as e:
            st.error(f"HF exception: {e}")
            return None

    # BUILD INDEX
    def build_index():
        raw = load_files()
        if not raw:
            return None

        corpus, sources = [], []
        for fname, txt in raw:
            for c in chunk(txt):
                corpus.append(c)
                sources.append(fname)

        try:
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(corpus)
        except Exception as e:
            st.error(f"TF-IDF error: {e}")
            return None

        return corpus, sources, tfidf, tfidf_matrix

    # RETRIEVE
    def retrieve(query, corpus, sources, tfidf, tfidf_matrix):

        scores = (tfidf_matrix @ tfidf.transform([query]).T).toarray().ravel()
        idx = scores.argsort()[::-1][:20]

        narrowed = [corpus[i] for i in idx]
        narrowed_sources = [sources[i] for i in idx]

        q_emb = embed([query])
        if q_emb is None:
            return []

        c_emb = embed(narrowed)
        if c_emb is None:
            return []

        q_emb = q_emb[0]
        sim = c_emb @ q_emb / (
            np.linalg.norm(c_emb, axis=1) * np.linalg.norm(q_emb)
        )

        top = sim.argsort()[::-1][:TOP_K]
        return [(narrowed[i], narrowed_sources[i]) for i in top]

    # GROQ LLM
    def ask(context, question):
        try:
            r = requests.post(
                GROQ_CHAT_API,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system",
                         "content": "Answer ONLY from the context."},
                        {"role": "user",
                         "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
                    ]
                },
                timeout=60
            )
            if r.status_code != 200:
                st.error(f"Groq error {r.status_code}: {r.text}")
                return "Groq API error."
            return r.json()["choices"][0]["message"]["content"]

        except Exception as e:
            st.error(f"Groq exception: {e}")
            return "Groq API failed."

    # =======================
    # MAIN UI
    # =======================

    st.title("📘 Free AI Chatbot (Docs + HF + Groq)")

    if "index" not in st.session_state:
        st.session_state.index = None

    if st.session_state.index is None:
        if st.button("📦 Build Document Index"):
            with st.spinner("Building index..."):
                st.session_state.index = build_index()
                if st.session_state.index:
                    st.success("Index ready!")
        return

    corpus, sources, tfidf, tfidf_matrix = st.session_state.index

    if "history" not in st.session_state:
        st.session_state.history = []

    query = st.chat_input("Ask your question...")

    if query:
        st.session_state.history.append(("user", query))

        chunks = retrieve(query, corpus, sources, tfidf, tfidf_matrix)
        if chunks:
            context = "\n\n---\n\n".join([f"[{s}]\n{t}" for t, s in chunks])
            answer = ask(context, query)
        else:
            answer = "No relevant information found."

        st.session_state.history.append(("assistant", answer))

    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

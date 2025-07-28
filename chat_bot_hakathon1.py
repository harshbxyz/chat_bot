import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision"

st.title("PDF Chatbot (Optimized RAG with Llama 3.2 Vision via Ollama)")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    # Extract text from PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Split text into larger chunks for speed
    chunk_size = 1500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Cache embeddings to avoid recomputation
    if "embeddings" not in st.session_state or st.session_state.get("last_pdf") != pdf_file.name:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
        st.session_state["embeddings"] = chunk_embeddings
        st.session_state["chunks"] = chunks
        st.session_state["embedder"] = embedder
        st.session_state["last_pdf"] = pdf_file.name
    else:
        chunk_embeddings = st.session_state["embeddings"]
        chunks = st.session_state["chunks"]
        embedder = st.session_state["embedder"]

    st.success("PDF processed! Ask your questions below.")
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        with st.spinner("Thinking..."):
            # Embed the question
            question_embedding = embedder.encode(user_question, convert_to_tensor=True)
            # Find the most similar chunk
            scores = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
            top_k = 1  # Only use the single most relevant chunk
            top_indices = scores.topk(top_k).indices.tolist()
            context = "\n\n".join([chunks[i] for i in top_indices])

            # Build prompt for Llama 3.2 Vision
            prompt = (
                "You are a helpful assistant. Use the following document context to answer the user's question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {user_question}\nAnswer:"
            )

            # Query Ollama
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
            )
            if response.ok:
                answer = response.json()["response"]
                st.write(answer.strip())
            else:
                st.error("Error communicating with Ollama.")
else:
    st.info("Please upload a PDF to get started.")
import os
import streamlit as st
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
import faiss
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
    return text

def load_category_pdfs(folder_path: str = "data") -> Dict[str, str]:
    category_files = {
        "Auto": "auto_policy.pdf",
        "Health": "health_policy.pdf",
        "Home": "home_policy.pdf",
        "Life": "life_policy.pdf"
    }
    texts = {}
    for category, filename in category_files.items():
        full_path = os.path.join(folder_path, filename)
        if os.path.exists(full_path):
            texts[category] = extract_text_from_pdf(full_path)
    return texts


def chunk_text(text: str, chunk_size: int = 500, subcategory: str = "") -> List[str]:
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks, current_chunk = [], ""
    for sentence in sentences:
        sentence += "."
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + "\n"
        else:
            chunks.append((subcategory + ": " + current_chunk.strip()) if subcategory else current_chunk.strip())
            current_chunk = sentence + "\n"
    if current_chunk:
        chunks.append((subcategory + ": " + current_chunk.strip()) if subcategory else current_chunk.strip())
    return chunks

def create_faiss_index(texts: Dict[str, str], model: SentenceTransformer) -> Tuple[faiss.Index, List[str], List[str]]:
    chunks, metadata, all_embeddings = [], [], []
    dim = model.get_sentence_embedding_dimension()
    for category, text in texts.items():
        if text:
            category_chunks = chunk_text(text, subcategory=category)
            embeds = model.encode(category_chunks, batch_size=32)
            all_embeddings.extend(embeds)
            chunks.extend(category_chunks)
            metadata.extend([category] * len(category_chunks))
    embeddings_array = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_array)
    return index, chunks, metadata


class HuggingFaceEngine:
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def generate(self, question: str, context: List[str]) -> str:
        sentence_lines = []
        for chunk in context:
            for sent in chunk.split("."):
                sent = sent.strip()
                if sent:
                    sentence_lines.append(sent + ".")
        joined_context = "\n".join(sentence_lines)

        prompt = (
            "You are an expert insurance assistant. Based only on the following context, "
            "answer the user's question in simple language, formatted using **3-5 bullet points, "
            "with short examples and emojis where helpful. Do not add anything beyond the context.\n\n"
            f"### CONTEXT:\n{joined_context}\n\n"
            f"### QUESTION:\n{question}\n\n"
            "### RESPONSE:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


st.set_page_config(page_title="InsuranceBot: Your Insurance AI Assistant", layout="wide")
st.title("🛡️ InsuranceBot: Your Insurance AI Assistant")

if 'step' not in st.session_state:
    st.session_state.update({
        'step': 1,
        'category': None,
        'subcategory': None,
        'index': None,
        'chunks': None,
        'metadata': None
    })

SUBCATEGORIES = {
    "Policy Types": "List the different policy types with key features",
    "Coverage Options": "Explain what is covered under this policy",
    "Premiums": "Describe premium information step-by-step",
    "Claim Process": "Explain the step-by-step claim process"
}

@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = load_category_pdfs("D:\\docsShruthi\\hackathonDocs")
    index, chunks, metadata = create_faiss_index(texts, embed_model)
    return embed_model, index, chunks, metadata

embed_model, faiss_index, chunks, metadata = load_resources()
llm = HuggingFaceEngine()


def select_category(category):
    st.session_state.category = category
    st.session_state.step = 2

def select_subcategory(subcat):
    st.session_state.subcategory = subcat
    st.session_state.step = 3

def go_back_to_step(step):
    st.session_state.step = step


def step1():
    st.subheader("1. Select Insurance Type")
    cols = st.columns(4)
    for i, category in enumerate(["Auto", "Health", "Home", "Life"]):
        with cols[i]:
            st.button(category, on_click=select_category, args=(category,))

def step2():
    st.subheader(f"2. What do you need to know about {st.session_state.category} insurance?")
    cols = st.columns(2)
    for i, (subcat, query) in enumerate(SUBCATEGORIES.items()):
        with cols[i % 2]:
            st.button(subcat, on_click=select_subcategory, args=(subcat,))
    st.button("← Back", on_click=go_back_to_step, args=(1,))

def step3():
    category = st.session_state.category
    subcategory = st.session_state.subcategory
    query = SUBCATEGORIES[subcategory]
    query_vec = embed_model.encode([query])
    _, indices = faiss_index.search(query_vec, k=5)
    context = [chunks[i] for i in indices[0] if metadata[i] == category][:2]
    st.subheader(f"{category} - {subcategory}")
    with st.spinner("🤖 AI analyzing policy documents..."):
        answer = llm.generate(query, context)
        st.markdown(f"*AI Answer:*\n{answer}")
        with st.expander("📄 Source Excerpts"):
            for i, chunk in enumerate(context):
                st.markdown(f"*Excerpt {i+1}:* {chunk[:200]}...")
    st.button("← Back", on_click=go_back_to_step, args=(2,))
    st.button("🔄 New Query", on_click=go_back_to_step, args=(1,))

if st.session_state.step == 1:
    step1()
elif st.session_state.step == 2:
    step2()
elif st.session_state.step == 3:
    step3()

# Sidebar
with st.sidebar:
    st.markdown("### How it works:")
    st.markdown("1. Select insurance type\n2. Choose your query\n3. Get AI-generated answer")
    st.markdown("---")
    st.markdown("*Tech Stack:*\n- HuggingFace flan-t5\n- FAISS\n- SentenceTransformers")
    st.markdown("*Need more help?*\n📧 support@insuranceai.com")
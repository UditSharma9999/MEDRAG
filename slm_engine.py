# # slm_engine.py

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load your fine-tuned SLM model (can replace with any path or HuggingFace model)
# MODEL_PATH = "distilgpt2"  # Replace with path to fine-tuned model if available

# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# def get_slm_response(query, context_chunks, max_tokens=300):
#     # Combine top 3 retrieved chunks with user query
#     context = "\n\n".join(context_chunks[:3])
#     prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=inputs["input_ids"].shape[1] + max_tokens,
#         num_return_sequences=1,
#         pad_token_id=tokenizer.eos_token_id
#     )

#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Extract only answer (remove prompt)
#     if "Answer:" in generated_text:
#         return generated_text.split("Answer:")[-1].strip()
#     return generated_text.strip()


# rag_with_slm.py
import os
import fitz
import re
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize ChromaDB and embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_papers")

# Load fine-tuned SLM model
slm_tokenizer = AutoTokenizer.from_pretrained("slm_disease_finetuned")
slm_model = AutoModelForCausalLM.from_pretrained("slm_disease_finetuned")
slm_model.eval()

# Preprocessing PDF chunks
def preprocess_text(text, chunk_size=1000, overlap=100):
    cleaned = re.sub(r"\s+", " ", text).strip()
    chunks = []
    for i in range(0, len(cleaned), chunk_size - overlap):
        chunk = cleaned[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        page_chunks = preprocess_text(text)
        for chunk in page_chunks:
            chunks.append((chunk, i + 1))
    return chunks

def store_chunks(chunks, file_name):
    for idx, (chunk, page_no) in enumerate(chunks):
        embedding = embedding_model.encode([chunk])
        metadata = {'source': file_name, 'page': page_no}
        collection.add(
            documents=[chunk],
            embeddings=embedding,
            metadatas=[metadata],
            ids=[f"{file_name}_p{page_no}_c{idx}"]
        )

def process_pdfs(data_folder):
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        path = os.path.join(data_folder, pdf_file)
        chunks = extract_chunks_from_pdf(path)
        store_chunks(chunks, pdf_file)

def retrieve_context(query, k=5):
    query_embedding = embedding_model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    docs = [doc for sublist in results['documents'] for doc in sublist]
    return docs

def get_slm_response(query, context_chunks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slm_model.to(device)

    context = "\n\n".join(context_chunks[:3])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = slm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = slm_model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=slm_tokenizer.eos_token_id
    )

    answer = slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()

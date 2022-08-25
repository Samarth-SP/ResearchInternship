import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import faiss
import pickle
data = pd.read_json('ArXiv.json')
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
st.header('ArXiv Search Interface')
st.write('This is a small search interface which uses semantic search tools to match user queries to the top 10 research papers under the astro-ph category on ArXiv. The searching employs SBERT embeddings and Hugging Face Transformers along with faiss KNN to get results.')
query = st.text_input(label='Input Box', value='')
query_embedding = embedder.encode([query])
with open('corpus_embeddings.pickle', 'rb') as pkl:
    sentence_embeddings = pickle.load(pkl)
n = 25
dimension = sentence_embeddings.shape[1]
quantizer = faiss.IndexFlatL2(dimension)
KNN = faiss.IndexIVFFlat(quantizer, dimension, n)
KNN.train(sentence_embeddings)
KNN.add(sentence_embeddings)
D, I = KNN.search(query_embedding, n)
for val in I[0]:
    st.write(data.iloc[val]['title'])

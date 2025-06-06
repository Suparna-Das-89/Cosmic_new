# ─────────────── 1. Imports ───────────────
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import feedparser
import arxiv
import re
import numpy as np
import matplotlib.pyplot as plt
import ephem
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer, util

# ─────────────── 2. Page config and CSS ───────────────
st.set_page_config(page_title="🌌 Cosmic Chatbot", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://apod.nasa.gov/apod/image/2305/MWandAurora_Odegard_960.jpg");
    background-size: cover;
    background-position: center;
}
html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ─────────────── 3. API Keys and LLM ───────────────
os.environ["GROQ_API_KEY"] = "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"
NASA_API_KEY = "rD8cgucyU9Rgcn1iTaOeh7mo1CPd6oN4CYThCdjg"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
Settings.embed_model = embed_model
llm = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])
Settings.llm = llm
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────── 4. Helper Functions ───────────────
def get_apod_image():
    try:
        res = requests.get(f"https://api.nasa.gov/planetary/apod?api_key={NASA_API_KEY}")
        data = res.json()
        return data.get("title", ""), data.get("url", ""), data.get("explanation", "")
    except:
        return "", "", "Could not load image."

def get_nasa_news():
    try:
        feed_url = "https://www.nasa.gov/news-release/rss"
        feed = feedparser.parse(feed_url)
        return [(entry.title, entry.link) for entry in feed.entries[:5]]
    except:
        return [("Could not fetch NASA news.", "#")]

def get_solar_activity():
    try:
        res = requests.get("https://services.swpc.noaa.gov/json/flares.json")
        data = res.json()
        if data:
            flare = data[0]
            return f"☀️ Solar flare: class {flare['classType']} at {flare['beginTime']}"
    except:
        return "No solar activity data."

def get_next_full_moon():
    try:
        next_full = ephem.next_full_moon(ephem.now())
        return f"🌕 Next full moon: {next_full.datetime().strftime('%Y-%m-%d %H:%M UTC')}"
    except Exception as e:
        return f"Lunar data unavailable: {e}"

def get_topic_embedding_match(query):
    known_topics = [
        "black hole", "white dwarf", "milky way", "solar system", "event horizon",
        "dark matter", "neutron star", "cosmic microwave background", "supernova",
        "gravitational wave", "red giant", "pulsar", "x-ray binary", "solar flare",
        "aurora borealis", "exoplanet", "hot jupiter", "super-earth", "ice giant",
        "terrestrial planet", "planet nine", "oort cloud", "kuiper belt", "asteroid belt",
        "dwarf planet", "pluto", "mars", "venus", "jupiter", "saturn", "uranus", "neptune",
        "moon", "earth", "space shuttle", "spacex", "starlink", "mars rover", "perseverance",
        "curiosity", "space debris", "iss", "apollo program", "voyager 1", "hubble space telescope",
        "james webb space telescope", "solar eclipse", "lunar eclipse", "solar neutrino"
    ]
    parts = re.split(r"[,.;:!?&]| and | or ", query.lower())
    best_topic, best_score = None, -1
    for part in parts:
        part = part.strip()
        if not part:
            continue
        query_emb = sbert_model.encode(part, convert_to_tensor=True)
        topic_embs = sbert_model.encode(known_topics, convert_to_tensor=True)
        similarities = util.cos_sim(query_emb, topic_embs)[0]
        top_idx = int(np.argmax(similarities))
        score = float(similarities[top_idx])
        if score > best_score:
            best_score = score
            best_topic = known_topics[top_idx]
    return best_topic

def get_wikipedia_summary(topic):
    try:
        topic = topic.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", ""), data.get("thumbnail", {}).get("source", ""), data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return "", "", ""
    except:
        return "", "", ""

def search_arxiv(query, max_results=5):
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        return [f"{res.title}

{res.summary}" for res in search.results()]
    except:
        return []

def plot_cmb_example():
    x = np.linspace(0.1, 10, 100)
    y = 1 / (x ** 2)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(x, y)
    ax.set_title("CMB Intensity Curve")
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")
    st.pyplot(fig)

# ─────────────── 5. Streamlit App ───────────────
st.title("🌌 Ask the Cosmos")
st.markdown("Type a space-related question (e.g., *When is the next full moon?* or *What is Jupiter?*)")

query = st.text_input("Ask your question about the universe:")

st.subheader("📸 NASA Astronomy Picture of the Day")
title, img_url, desc = get_apod_image()
if img_url:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_url, caption=title, use_container_width=True)
        st.markdown(f"<p style='text-align: center; font-size: 14px;'>{desc}</p>", unsafe_allow_html=True)

st.subheader("📰 Latest NASA News")
for title, link in get_nasa_news():
    st.markdown(f"- [{title}]({link})")

st.sidebar.header("🔭 Solar & Lunar Updates")
st.sidebar.markdown(get_solar_activity())
st.sidebar.markdown(get_next_full_moon())

if query:
    with st.spinner("🔄 Retrieving answer from the cosmos..."):
        topic = get_topic_embedding_match(query)
        wiki_context, image_url, page_url = get_wikipedia_summary(topic)
        live_context = get_solar_activity() + "\n" + get_next_full_moon()
        arxiv_texts = search_arxiv(query)
        docs = [Document(text=t) for t in arxiv_texts]
        index = VectorStoreIndex.from_documents(docs)
        nodes = index.as_retriever().retrieve(query)
        arxiv_context = "\n\n".join([n.get_content()[:500] for n in nodes])
        final_context = wiki_context + "\n\n" + arxiv_context + "\n\n" + live_context
        prompt = f"""
You are a helpful and knowledgeable cosmic assistant that answers space-related questions clearly, accurately, and in an educational tone suitable for curious learners.

Use the information provided in the context below as your primary source. If the answer isn't fully covered there, you may also draw from general scientific knowledge to complete your reply — but remain truthful and clearly mark any uncertainty.

If the topic is a well-known scientific concept (e.g. black holes, dark matter, cosmic microwave background), expand the explanation slightly beyond the raw definition to include historical context, scientific significance, or how it's observed.

Respond in a way that balances scientific accuracy and accessibility — make it understandable for non-experts without oversimplifying key facts.

Context:
{final_context}

Question: {query}

Answer:
"""
        response = llm.complete(prompt=prompt)
        st.subheader("💬 Cosmic Answer")
        st.markdown(response.text)
        if image_url:
            st.image(image_url, caption=f"Wikipedia image for {topic}", width=300)
        if page_url:
            st.markdown(f"[🔗 Read more on Wikipedia]({page_url})")
        if topic.lower() == "cosmic microwave background":
            st.markdown("📊 Sample visual of CMB intensity vs wavelength:")
            plot_cmb_example()
else:
    st.info("Enter a question about the cosmos to begin your journey! 🚀")

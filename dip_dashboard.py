import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import networkx as nx
import math


# Streamlit page config
st.set_page_config(page_title="DIP Analytics Dashboard", layout="wide", page_icon="📊")


# Helper / util functions
@st.cache_data
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

@st.cache_data
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

@st.cache_data
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Thematic categories 
THEMATIC_KEYWORDS = {
    'Architecture & Monuments': ['architecture', 'monument', 'temple', 'fort', 'palace', 'building', 'structure', 'heritage', 'gateway'],
    'Performing Arts': ['dance', 'music', 'theatre', 'performance', 'sangeet', 'natyam', 'kathak', 'bharatanatyam', 'drama'],
    'Visual Arts & Crafts': ['painting', 'craft', 'sculpture', 'art', 'handicraft', 'textile', 'pottery', 'weaving', 'painting'],
    'Religious Traditions': ['religion', 'ritual', 'worship', 'prayer', 'pilgrimage', 'deity', 'shrine'],
    'Festivals & Celebrations': ['festival', 'celebration', 'diwali', 'holi', 'eid', 'christmas', 'puja', 'fair', 'mela'],
    'Food & Cuisine': ['food', 'cuisine', 'dish', 'cooking', 'recipe', 'spice', 'culinary', 'meal'],
    'Traditional Livelihoods': ['livelihood', 'occupation', 'trade', 'farming', 'agriculture', 'fishery', 'artisan'],
    'Social Customs': ['custom', 'tradition', 'marriage', 'wedding', 'social', 'community', 'family'],
    'Language & Literature': ['language', 'literature', 'folk', 'story', 'poem', 'oral', 'script']
}

# Hardcoded cultural keywords 
HARDCODED_CULTURAL_WORDS = [
    'temple', 'festival', 'dance', 'music', 'craft', 'ritual', 'marriage', 'cuisine',
    'palace', 'handicraft', 'painting', 'folk', 'poetry', 'heritage', 'tradition', 'ceremony'
]

# Simple sentiment lexicons 
POSITIVE_WORDS = set(['good', 'beautiful', 'vibrant', 'rich', 'festive', 'celebration', 'delicious', 'unique', 'historic'])
NEGATIVE_WORDS = set(['poor', 'neglect', 'decline', 'lost', 'threat', 'danger', 'problem', 'polluted'])

INDIAN_STATES = [
    'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
    'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka',
    'kerala', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram',
    'nagaland', 'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu',
    'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal',
    'delhi', 'jammu and kashmir', 'ladakh', 'puducherry', 'chandigarh'
]

EXTENDED_STOPWORDS = set([
    "the","and","for","you","are","was","were","is","am","be","being","been",
    "this","that","these","those","with","from","have","has","had","their","there",
    "would","could","should","about","which","such","into","through","more","than",
    "also","some","what","when","where","who","whom","why","how","very","only","then",
    "they","them","your","our","its","it's","i","we","he","she","it","his","her","my",
    "me","mine","yours","ours","theirs","at","on","in","to","of","as","by","an","a",
    "but","if","or","because","so","up","out","over","under","too","can","will","just",
    "like","said","use","used","one","two","may","might","not","all","any","now","get","know","made","work","people",
    "time","other","content","research","india","hai","food","fish",
    "different","cuisine","interviewer","many","also","like","one",
    "could","would","really","make","use","used","using","see","well","even","way","good",
    "new","first","last","much","still","back","day","years","year","say","says","going","take","taken","most", "lot", "while", "during", "don", "after", "here", "come", "https", "www","however","yeah","ok","okay","right","uh","um","oh","hmm","mmm","ah","hey","yes","same","still","though","let","lets","shall","may","might","must","ought","need","dare","used","also","even","ever","never","always","sometimes","usually","often"
])


WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")


# Text processing helpers
def categorize_text_counts(text):
    text_lower = text.lower()
    scores = {cat: sum(text_lower.count(k) for k in kws) for cat, kws in THEMATIC_KEYWORDS.items()}
    return scores


def extract_indian_states_from_text(text):
    text_lower = text.lower()
    found = {}
    for s in INDIAN_STATES:
        c = text_lower.count(s)
        if c > 0:
            found[s.title()] = c
    return found


def extract_hardcoded_cultural_counts(text, keywords=HARDCODED_CULTURAL_WORDS):
    text_lower = text.lower()
    return {k: text_lower.count(k) for k in keywords}


def tokenize_and_count(text):
    words = WORD_RE.findall(text.lower())
    words = [w for w in words if w not in EXTENDED_STOPWORDS]
    return Counter(words)


def extract_entities_spacy(text, nlp_model, allowed_types=None):
    if nlp_model is None:
        return []
    doc = nlp_model(text)
    ents = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
    if allowed_types:
        ents = [e for e in ents if e[1] in allowed_types]
    return ents


# Embeddings & clustering
def get_document_embeddings(docs, embeddings_model):
    texts = [doc.page_content for doc in docs]
    embeddings = embeddings_model.embed_documents(texts)
    return np.array(embeddings)


def create_tsne_and_clusters(embeddings, docs, n_clusters=None):
    n = len(embeddings)
    if n == 0:
        return None, None, None
    # choose k automatically if not provided
    if n_clusters is None:
        n_clusters = min(6, max(2, int(math.sqrt(n))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
    emb2 = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'x': emb2[:, 0],
        'y': emb2[:, 1],
        'cluster': clusters,
        'text': [doc.page_content[:300] + '...' for doc in docs],
        'source': [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in docs]
    })
    return df, clusters, n_clusters


# Cluster naming & summaries
def name_clusters_by_keywords(df_chunks, top_n=3):
    cluster_labels = {}
    for c in sorted(df_chunks['cluster'].unique()):
        texts = " ".join(df_chunks[df_chunks['cluster'] == c]['text'].tolist())
        counts = tokenize_and_count(texts)
        
        candidates = []
        for theme, kws in THEMATIC_KEYWORDS.items():
            for k in kws:
                if counts.get(k, 0) > 0:
                    candidates.append(k)
        # fallback to top tokens
        if not candidates:
            candidates = [w for w, _ in counts.most_common(top_n)]
        label = " & ".join(candidates[:2]) if candidates else f"Cluster {c}"
        cluster_labels[c] = label.title()
    return cluster_labels


def cluster_summary_table(df_chunks, docs_chunks, top_k_words=5):
    rows = []
    for c in sorted(df_chunks['cluster'].unique()):
        subset = df_chunks[df_chunks['cluster'] == c]
        texts = " ".join(subset['text'].tolist())
        word_counts = tokenize_and_count(texts)
        top_words = ", ".join([w for w, _ in word_counts.most_common(top_k_words)])
        top_papers = subset['source'].value_counts().head(3).index.tolist()
        rows.append({
            'Cluster': c,
            'Top Words': top_words,
            'Top Papers (3)': ", ".join(top_papers),
            'Num Chunks': len(subset)
        })
    return pd.DataFrame(rows)


# Entity network (co-occurrence)
def build_entity_network_from_chunks(chunks_df, n_top_entities=50, min_cooccurrence=2):
    
    all_entities = Counter()
    for ents in chunks_df['entities']:
        uniq = set([e[0] for e in ents])
        all_entities.update(uniq)
    top_entities = set([e for e, _ in all_entities.most_common(n_top_entities)])

    # co-occurrence counts
    cooc = Counter()
    for ents in chunks_df['entities']:
        names = [e[0] for e in ents if e[0] in top_entities]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                pair = tuple(sorted([names[i], names[j]]))
                cooc[pair] += 1

    G = nx.Graph()
    # add nodes with frequency
    for e, freq in all_entities.items():
        if e in top_entities:
            G.add_node(e, freq=freq)
    # add edges if above threshold
    for (a, b), cnt in cooc.items():
        if cnt >= min_cooccurrence:
            G.add_edge(a, b, weight=cnt)
    return G


# Simple sentiment scoring per topic/state
def chunk_sentiment(text):
    words = WORD_RE.findall(text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return "neutral"
    return "positive" if pos >= neg else "negative"

def simple_sentiment_score(text):
    words = WORD_RE.findall(text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos + neg == 0:
        return 0
    return (pos - neg) / (pos + neg)


# UI / App
nlp = load_spacy_model()

if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'docs_chunks' not in st.session_state:
    st.session_state.docs_chunks = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None

# Header
st.markdown("""
# Discover India Program — Analytics
Comprehensive, de-duplicated visualizations focused on insights (thematic, geographic, cultural, and clustering).
""")

# Sidebar: load docs + settings
with st.sidebar:
    st.header("Configuration")
    directory = st.text_input("Documents Directory", value="
from fastapi import FastAPI
from pydantic import BaseModel
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import nltk
from scipy.spatial import distance

# Initialize FastAPI application
app = FastAPI()

# Define the request model using Pydantic
class TextPair(BaseModel):
    text1: str
    text2: str

# Download necessary NLTK datasets
nltk.download('punkt')

# Pre-processing function
def tokenize_text(text):
    return word_tokenize(text.lower())

# Function to train Word2Vec model
def train_word2vec(dataset_path: str):
    data = pd.read_csv(dataset_path)
    # Tokenize texts
    sentences = data['text1'].apply(tokenize_text).tolist() + data['text2'].apply(tokenize_text).tolist()
    # Train Word2Vec model
    w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    return w2v_model

# Train Word2Vec model with the dataset at startup
word2vec_model = train_word2vec('DataNeuron_Text_Similarity.csv')

# Function to calculate text similarity using Word2Vec embeddings
def calculate_similarity(text1, text2):
    # Ensure the model has been trained
    if word2vec_model is None:
        raise ValueError("Word2Vec model has not been trained.")
    
    # Tokenize and vectorize the texts using Word2Vec
    vector1 = sum(word2vec_model.wv[token] for token in tokenize_text(text1) if token in word2vec_model.wv.key_to_index)
    vector2 = sum(word2vec_model.wv[token] for token in tokenize_text(text2) if token in word2vec_model.wv.key_to_index)
    
    # Compute cosine similarity as 1 - cosine distance
    sim_score = 1 - distance.cosine(vector1, vector2)
    
    # Convert sim_score to a native Python float
    return float(sim_score)

# API endpoint for processing text and returning similarity score
@app.post('/text_similarity')
def text_similarity_endpoint(text_pair: TextPair):
    # Calculate the similarity score
    similarity_score = calculate_similarity(text_pair.text1, text_pair.text2)
    # Return the similarity score as a JSON-compatible type
    return {'similarity score': similarity_score}

# To run the app, use the following command:
# uvicorn script_name:app --host 0.0.0.0 --port 8000

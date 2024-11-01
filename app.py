from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load and preprocess your dataset
df_clothing = pd.read_csv('Data/Fashion_Dataset.csv')
def clean_keywords(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Clean the descriptions (example function)
def clean_description(text):
    # Check if the text is a string; if not, replace with an empty string
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only alphanumeric
    return text.lower()  # Convert to lowercase


df_clothing['cleaned_description'] = df_clothing['description'].apply(clean_description)

# Vectorize descriptions for cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
description_vectors = vectorizer.fit_transform(df_clothing['cleaned_description'])

# Define Pydantic model for request
class KeywordRequest(BaseModel):
    keywords: str

def get_recommendations(keywords, top_n=5):
    keywords_cleaned = clean_keywords(keywords) # instead of clean_description(keywords)
    keyword_vector = vectorizer.transform([keywords_cleaned])
    cosine_similarities = cosine_similarity(keyword_vector, description_vectors).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Select relevant columns and replace NaN values with an empty string
    recommendations = df_clothing.iloc[top_indices][['p_id', 'name', 'price', 'colour', 'brand', 'img', 'avg_rating']]
    recommendations = recommendations.fillna('')  # Replace NaN values with an empty string
    return recommendations

@app.post("/recommendations")

async def recommend_clothing(request: KeywordRequest):

    try:

        # Get recommendations based on cleaned keywords

        recommendations = get_recommendations(request.keywords)

        

        # Convert recommendations DataFrame to a list of dictionaries for JSON response

        return {

            "cleaned_keywords": clean_keywords(request.keywords),

            "results": recommendations.to_dict(orient="records")

        }

    except Exception as e:

        print(f"Error occurred: {e}")

        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

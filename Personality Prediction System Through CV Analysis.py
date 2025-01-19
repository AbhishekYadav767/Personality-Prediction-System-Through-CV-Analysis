# Install necessary libraries
!pip install nltk pandas

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Predefined personality trait dictionary
trait_keywords = {
    "Openness": ["creative", "curious", "imaginative", "adventurous", "original"],
    "Conscientiousness": ["organized", "efficient", "disciplined", "focused", "methodical"],
    "Extraversion": ["outgoing", "energetic", "sociable", "talkative", "assertive"],
    "Agreeableness": ["cooperative", "trustworthy", "empathetic", "kind", "helpful"],
    "Neuroticism": ["anxious", "nervous", "insecure", "sensitive", "moody"]
}

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Personality prediction function
def predict_personality(cv_text):
    # Preprocess the CV text
    tokens = preprocess_text(cv_text)
    # Initialize trait scores
    trait_scores = {trait: 0 for trait in trait_keywords}
    # Count matches for each trait
    for trait, keywords in trait_keywords.items():
        trait_scores[trait] += sum(1 for word in tokens if word in keywords)
    return trait_scores

# Example usage
cv_text = """
I am a highly creative and imaginative individual with a passion for solving problems. 
I am well-organized, methodical, and disciplined in my approach to work. 
I enjoy collaborating with others and value trust and empathy in teamwork.
"""

# Predict personality traits
trait_scores = predict_personality(cv_text)

# Display results
print("Personality Trait Analysis:")
for trait, score in trait_scores.items():
    print(f"{trait}: {score}")

# Output personality traits as a DataFrame
df = pd.DataFrame(trait_scores.items(), columns=["Trait", "Score"])
df.plot(kind="bar", x="Trait", y="Score", legend=False, title="Personality Trait Scores")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split into words and join back
    words = text.split()
    return ' '.join(words)

def load_and_preprocess_data(file_path):
    try:
        # Load the dataset with tab separator
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path, 
                        sep='\t',  # Use tab as separator
                        names=['text1', 'text2', 'label'],  # Specify column names
                        skiprows=1)  # Skip header row if it exists
        
        # Display basic information about the dataset
        print("\nDataset Info:")
        print(f"Number of samples: {len(df)}")
        print("\nColumns in dataset:")
        print(df.columns.tolist())
        
        # Handle missing values in label column
        df['label'] = df['label'].fillna(0).astype(int)
        
        # Preprocess the text columns
        print("\nPreprocessing texts...")
        df['text1_processed'] = df['text1'].apply(preprocess_text)
        df['text2_processed'] = df['text2'].apply(preprocess_text)
        
        # Remove rows with empty processed texts
        df = df[df['text1_processed'].str.len() > 0]
        df = df[df['text2_processed'].str.len() > 0]
        
        print(f"Final number of samples after preprocessing: {len(df)}")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def train_model(df):
    print("\nPreparing data for training...")
    # Combine the processed texts
    texts = df['text1_processed'] + ' ' + df['text2_processed']
    
    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = df['label']
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training the model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def check_plagiarism(text1, text2, model, vectorizer):
    # Preprocess the texts
    text1_processed = preprocess_text(text1)
    text2_processed = preprocess_text(text2)
    
    # Combine texts
    combined_text = text1_processed + ' ' + text2_processed
    
    # Transform using the vectorizer
    X = vectorizer.transform([combined_text])
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability

def main():
    try:
        # Check if dataset exists
        dataset_path = 'plagiarism_dataset.csv'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Load and preprocess the data
        df = load_and_preprocess_data(dataset_path)
        
        # Train the model
        model, vectorizer = train_model(df)
        
        while True:
            print("\n=== Plagiarism Checker ===")
            print("Enter your texts to compare (or 'quit' to exit):")
            
            # Get first text
            text1 = input("\nEnter first text: ").strip()
            if text1.lower() == 'quit':
                break
                
            # Get second text
            text2 = input("Enter second text: ").strip()
            if text2.lower() == 'quit':
                break
            
            # Check plagiarism
            prediction, probability = check_plagiarism(text1, text2, model, vectorizer)
            
            # Display results
            print("\nResults:")
            print(f"Text 1: {text1}")
            print(f"Text 2: {text2}")
            print(f"Prediction: {'Plagiarized' if prediction == 1 else 'Not Plagiarized'}")
            print(f"Confidence: {probability:.2f}")
            
            # Ask if user wants to continue
            choice = input("\nDo you want to check another pair of texts? (yes/no): ").strip().lower()
            if choice != 'yes':
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
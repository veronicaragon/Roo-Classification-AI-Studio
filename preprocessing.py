import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None

    def remove_stop_words(self, text):
        """Remove stop words from text."""
        words = [word for word in str(text).split() if word.lower() not in self.stop_words]
        return ' '.join(words)

    def lemmatize_text(self, text):
        """Lemmatize text."""
        words = [self.lemmatizer.lemmatize(word) for word in str(text).split()]
        return ' '.join(words)

    def is_english(self, text):
        """Check if text is English."""
        try:
            return detect(text) == 'en'
        except:
            return True

    def clean_roo_data(self, df):
        """Clean Roo conversation data."""
        # Remove automated responses
        df = df[~df['First_response'].str.contains(
            'Hey there|BTW, this chat is completely private|All health educators are currently busy|'
            'Live Chat is currently closed for the holiday|Live chat with health educators is currently closed|Hey, hey.',
            case=False, na=False
        )]

        # Remove prompt:livechatinstant
        df = df[df['First_prompt'].str.contains('prompt:livechatinstant')==False]

        # Filter English content
        df = df[df['First_prompt'].apply(self.is_english)]

        return df

    def process_text(self, df):
        """Apply full text preprocessing pipeline."""
        # Create copies of text columns
        df.loc[:, 'processed_prompt'] = df['First_prompt'].copy()
        df.loc[:, 'processed_response'] = df['First_response'].copy()

        # Remove stop words
        df.loc[:, 'processed_prompt'] = df['processed_prompt'].apply(self.remove_stop_words)
        df.loc[:, 'processed_response'] = df['processed_response'].apply(self.remove_stop_words)

        # Lemmatize
        df.loc[:, 'processed_prompt'] = df['processed_prompt'].apply(self.lemmatize_text)
        df.loc[:, 'processed_response'] = df['processed_response'].apply(self.lemmatize_text)

        return df

    def vectorize_text(self, df, max_features=1000):
        """Convert text to TF-IDF vectors."""
        # Combine prompt and response
        combined_text = df['processed_prompt'] + ' ' + df['processed_response']

        # Initialize and fit vectorizer if not already done
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=max_features)
            X = self.vectorizer.fit_transform(combined_text)
        else:
            X = self.vectorizer.transform(combined_text)

        return X

    def prepare_data(self, df):
        """Run complete preprocessing pipeline."""
        # Clean data
        df_cleaned = self.clean_roo_data(df)
        
        # Process text
        df_processed = self.process_text(df_cleaned)
        
        # Vectorize
        X = self.vectorize_text(df_processed)
        
        return X, df_processed

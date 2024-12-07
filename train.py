import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class ModelTrainer:
    def __init__(self, model_type='svm', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.tokenizer = None

    def prepare_labels(self, y):
        """Encode labels and compute class weights."""
        y_encoded = self.label_encoder.fit_transform(y)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
        return y_encoded, class_weight_dict

    def build_lstm_model(self, max_words, embedding_dim, max_len, num_classes):
        """Build LSTM model architecture."""
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_lstm_data(self, texts, max_words=10000, max_len=100):
        """Prepare text data for LSTM model."""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=max_words)
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_len)
        return padded_sequences

    def train(self, X, y, test_size=0.2):
        """Train selected model type."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        if self.model_type == 'svm':
            # Train SVM
            y_encoded, class_weight_dict = self.prepare_labels(y)
            self.model = SVC(
                kernel='linear',
                class_weight=class_weight_dict,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)

        elif self.model_type == 'logistic':
            # Train Logistic Regression
            y_encoded, class_weight_dict = self.prepare_labels(y)
            self.model = LogisticRegression(
                multi_class='multinomial',
                class_weight=class_weight_dict,
                max_iter=1000,
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)

        elif self.model_type == 'lstm':
            # Prepare data for LSTM
            X_padded = self.prepare_lstm_data(X)
            y_encoded = self.label_encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_padded, y_categorical, 
                test_size=test_size, 
                random_state=self.random_state
            )
            
            # Build and train LSTM
            self.model = self.build_lstm_model(
                max_words=10000,
                embedding_dim=100,
                max_len=100,
                num_classes=len(np.unique(y_encoded))
            )
            
            self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.2
            )

        return X_test, y_test

    def predict(self, X):
        """Make predictions with trained model."""
        if self.model_type == 'lstm':
            X_padded = self.prepare_lstm_data(X)
            y_pred = self.model.predict(X_padded)
            return self.label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
        else:
            return self.model.predict(X)

    def save_model(self, path):
        """Save trained model to disk."""
        if self.model_type == 'lstm':
            self.model.save(path)
        else:
            pd.to_pickle(self.model, path)

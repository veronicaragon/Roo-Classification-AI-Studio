# Roo Chatbot Error Detection
*Part of Break Through Tech AI at Cornell Tech Studio, Fall 2024*

## Project Overview
This project develops an error detection model for Roo, Planned Parenthood's AI chatbot, to identify and classify response accuracy. The goal is to improve the quality of automated responses by detecting and categorizing potential errors into False Positives (FP), False Negatives (FN), True Positives (TP), and True Negatives (TN).

### Business Impact
Online sexual and reproductive health information is critical for young people, including LGBTQ youth, but misinformation is rampant. This project aims to:
- Enhance user experience with high-quality, error-free interactions
- Build trust on sensitive topics through accurate responses
- Increase sexual literacy through reliable automated information

## Objectives and Goals
1. Create an error-identification model for real-time monitoring of Roo's performance
2. Analyze trends in how Roo makes errors to identify areas for improvement
3. Provide recommendations for enhancing Roo's response accuracy

## Methodology

### Data Understanding & Preprocessing
- Cleaned dataset of Roo conversations, including user messages and bot responses
- Removed automated initial messages and non-English content
- Applied text preprocessing techniques:
  - Stop word removal
  - Lemmatization
  - TF-IDF vectorization
- Handled class imbalance through weighted sampling

### Model Development
Implemented and evaluated multiple classification approaches:
- Support Vector Machine (SVM)
- Logistic Regression
- Long Short-Term Memory (LSTM) networks

### Feature Engineering
- Combined prompt and response text for comprehensive analysis
- Applied text vectorization techniques including TF-IDF and BERT embeddings
- Created custom text features to capture semantic meaning

## Results and Key Findings

### Model Performance
- Logistic Regression achieved 77% accuracy (best performing model)
- Strong performance on TP/FP classification
- Challenges with minority classes (FN/TN) due to limited examples

### Error Analysis
- Most misclassifications occurred between False Positives and True Positives
- Model showed stronger performance on abortion and sexual health topics
- Limited training data for crisis and emergency scenarios

### Visualizations
[Include links to confusion matrices and performance plots]

## Individual Contributions - Veronica Aragon
- Implemented Support Vector Machine (SVM) classifier achieving 71% accuracy
- Created preprocessing pipeline for text normalization and feature extraction
- Developed evaluation metrics and confusion matrix visualizations
- Contributed to data cleaning and Spanish language filtering
- Performed error analysis and model comparison studies

## Setup Instructions

### Prerequisites
```
python 3.8+
pandas
scikit-learn
nltk
tensorflow
```

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model
```python
python train_model.py --data_path data/roo_conversations.csv --model logistic
```

### Evaluating Performance
```python
python evaluate_model.py --model_path models/best_model.pkl --test_data data/test.csv
```

## Potential Next Steps
1. Collect additional data for underrepresented classes (FN/TN)
2. Explore ensemble methods combining strengths of different models
3. Implement active learning for efficient data labeling
4. Deploy model in production environment with monitoring

## License
Apache License 2.0

## Acknowledgments
- Michelle Bao, Data Scientist at PPFA
- Michael O'Keefe, Senior Director of Software Engineering at PPFA
- Ambreen Molitor, National Director of Innovation at PPFA
- Break Through Tech AI Program at Cornell Tech

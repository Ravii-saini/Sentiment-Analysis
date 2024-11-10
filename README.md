# IMDB Movie Sentiment Analysis

## Project Overview
This project is a sentiment analysis model designed for classifying IMDB movie reviews as positive or negative. By applying natural language processing (NLP) techniques and machine learning algorithms, the model provides a quick sentiment assessment, highlighting trends in audience reactions to movies.

## Project Objectives
- Develop an NLP pipeline to preprocess and clean text data from IMDB reviews.
- Extract meaningful features from the text data using vectorization techniques.
- Train and evaluate a sentiment classifier to distinguish between positive and negative reviews with high accuracy.

## Key Steps

### Data Collection
- Utilized the IMDB dataset, containing 50,000 movie reviews labeled as positive or negative.

### Data Preprocessing
- **Text Cleaning**: Removed stop words, punctuation, and applied lowercasing.
- **Tokenization and Stemming**: Broke down sentences into words, applied stemming and lemmatization to reduce words to their base forms.
- **Feature Vectorization**: Converted cleaned text into numerical data using methods like bag-of-words and TF-IDF (Term Frequency-Inverse Document Frequency).

### Feature Extraction
- Used n-grams and word embeddings to capture context within the text.

### Model Training and Evaluation
- Tested multiple machine learning algorithms, including Logistic Regression and LinearSVC.
- Evaluated model performance based on accuracy and F1-score, achieving an accuracy of over 90%.

## Tools and Technologies Used
- **Languages**: Python
- **Development Tools**: Jupyter Notebook, Spyder IDE
- **Libraries**: 
  - **NLP**: NLTK, SpaCy, Gensim
  - **Machine Learning**: Scikit-Learn, Keras, TensorFlow

## Results
- **Model Performance**: Achieved 90%+ accuracy on test data, demonstrating reliable classification of movie reviews.
- **Data Processing Techniques**: Employed techniques like TF-IDF, stop word removal, and n-grams to boost model performance.

## Sample Code Snippet

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Vectorizing reviews using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf_vectorizer.fit_transform(reviews)

# Training a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)



Installation and Usage
Clone the Repository:

bash
Copy code
git clone <repository-url>
Install Required Packages:

bash
Copy code
pip install -r requirements.txt
Run the Notebook: Open the project in Jupyter Notebook:

bash
Copy code
jupyter notebook
Future Improvements
Incorporate advanced models like BERT for deeper text analysis.
Optimize performance with hyperparameter tuning for the current models.
Explore additional datasets to enhance the model’s generalization to diverse review sources.
License
This project is intended for self-learning purposes.

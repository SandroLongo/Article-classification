from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords as sw 
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
import re 
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")   


SEED = 17


def main():
    random.seed(SEED)            # seed per il modulo random di Python
    np.random.seed(SEED)

    df = pd.read_csv("development.csv")


    df = df.drop_duplicates(subset=['article'], keep='first').reset_index(drop=True)

    mask_len = df['article'].str.len() >= 50
    df = df[mask_len].copy() 

    df = df.reset_index(drop=True)

    def clean_text(text):
        if not isinstance(text, str):
            return ""

        # capturing alt captions from HTML before stripping tags
        captions = " ".join(re.findall(r'alt="([^"]*)"', text))
        captions = re.sub(r'[^a-zA-Z\s]', ' ', captions)

        # removing the URL messages 
        text_no_links = re.sub(r'https?://\S+', ' ', text)

        # remove HTML tags
        text_without_tags = re.sub(r'<[^>]+>', ' ', text_no_links)

        # keep information about money simbols
        text_without_tags = re.sub(
            r'[$€£¥]',
            ' MONEYAMOUNT ',
            text_without_tags
        )

        # this removes punctuation, numbers
        text_final = re.sub(r"[^a-zA-Z\s]", ' ', text_without_tags)

        combined = text_final  + " " + captions

        # 6. lower casse and extra whitespace removal
        return re.sub(r'\s+', ' ', combined).strip().lower()

    def cleaner_applyer(df):
        df['title'] = df['title'].fillna('').apply(clean_text)
        df['article'] = df['article'].fillna('').apply(clean_text)
        df['source'] = df['source'].fillna('')
        df['combined_text'] = df['title'] + " " + df['article']
        df['combined_text2'] = df['title'] + " " + df['article']


    class TextCleanerTransformer(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass
        
        def fit(self, X, y=None):
            return self  # Non deve fare nulla in fase di fit
        
        def transform(self, X):
            # create a copy
            X = X.copy()
                
            cleaner_applyer(X)
            
            return X
        
    class LemmaTokenizerGood(object): 
        def __init__(self): 
            self.lemmatizer = WordNetLemmatizer() 
            self.stop_words = set(sw.words('english'))
        
        def __call__(self, document): 
            lemmas = [] 
            for t in word_tokenize(document): 
                t = t.lower().strip()
                
                # if t still contains some other character differente from letters, or it is the stopword set, or it is too shot, it is removed
                if t.isalpha() and t not in self.stop_words and len(t) > 2:
                    # lemmatizing to extract its root form
                    lemma = self.lemmatizer.lemmatize(t) 
                    lemmas.append(lemma)
                    
            return lemmas
        

    lemmaTokenizer = LemmaTokenizerGood()   
    text_transformer1 = TfidfVectorizer(
        tokenizer=lemmaTokenizer, 
        stop_words=None, 
        )
    
    cleaner = TextCleanerTransformer()
    df = cleaner.fit_transform(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['year'] = df['timestamp'].dt.year
    df['year'] = df['year'].fillna(0)
    df['article_len'] = df['article'].fillna('').str.len()
    X = df.drop(columns=['label'])
    y = df['label']

    source_transformer = OneHotEncoder(handle_unknown='ignore')
    text_transformer_classic_char = TfidfVectorizer(analyzer='char')
    
    num_transformer = StandardScaler()

    preprocessor_svc = ColumnTransformer(
        transformers=[
            ('text', text_transformer1, 'combined_text'),
            ('char', text_transformer_classic_char, 'combined_text2'), 
            ('source', source_transformer, ['source']),
            ('pagerank', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ]), ['page_rank']),
            ('year', OneHotEncoder(handle_unknown='ignore'), ['year']),
            ('lenght', MinMaxScaler(), ['article_len'])
            
        ], remainder='drop'
    )

    best_pipeline_params_svc = { 
        'preprocessor__text__sublinear_tf' : True,
        'preprocessor__text__ngram_range' : (1, 3),
        'preprocessor__text__min_df' : 5,
        'preprocessor__text__max_features' : 200_000,
        'preprocessor__text__max_df' : 0.15,
        'preprocessor__text__use_idf' : True,
        'preprocessor__text__binary': False,
        'preprocessor__text__norm' : 'l2',


        'clf__C' : 0.075,
        'clf__class_weight' : 'balanced', 

        # char preprocessor

        'preprocessor__char__sublinear_tf' : True,
        'preprocessor__char__ngram_range' : (3, 5),
        'preprocessor__char__min_df' : 5,
        'preprocessor__char__max_features' : 120_000,
        'preprocessor__char__max_df' : 0.06,
        'preprocessor__char__use_idf' : True,
        'preprocessor__char__norm' : 'l2',

    }

    linear_svc = LinearSVC()

    pipe_svc = Pipeline([
        ('preprocessor', preprocessor_svc),
        ("clf", linear_svc)
    ])


    pipe_svc = pipe_svc.set_params(**best_pipeline_params_svc)

    evaluation = pd.read_csv('evaluation.csv')

    pipe_svc.fit(X, y)
    evaluation = cleaner.fit_transform(evaluation)

    evaluation['timestamp'] = pd.to_datetime(evaluation['timestamp'], errors='coerce')
    evaluation['year'] = evaluation['timestamp'].dt.year
    evaluation['year'] = evaluation['year'].fillna(0)
    evaluation['article_len'] = evaluation['article'].fillna('').str.len()

    prediction = pipe_svc.predict(evaluation)

    results = pd.DataFrame({
        'Id': range(len(prediction)), 
        'Predicted': prediction
    })

    results.to_csv('submission.csv', index=False) #score public 0.714
    return results

    
if __name__ == "__main__":
    main()
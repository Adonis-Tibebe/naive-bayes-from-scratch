from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_transform_train(train_texts, min_df=5, max_df=0.85, ngram_range=(1, 1)):
    """
    Initializes and fits a CountVectorizer on the training data ONLY.
    Transforms the training text into a sparse feature matrix.
    
    Defaults rely on CountVectorizer's built-in lowercasing and standard 
    punctuation stripping. Stopwords are kept intact.

    Support ngrams for capturing word pairs
    
    Args:
        train_texts (iterable): The text corpus from the training set.
        min_df (int/float): Threshold to ignore rare words.
        max_df (int/float): Threshold to ignore overly common words.
        
    Returns:
        tuple: (fitted_vectorizer, X_train_sparse)
    """
    vectorizer = CountVectorizer(
        min_df=min_df, 
        max_df=max_df, 
        lowercase=True, 
        ngram_range=ngram_range,
        stop_words=None # Explicitly keeping stopwords to preserve negations
    )
    
    # Fit the vocabulary and transform the training text in one step
    X_train_sparse = vectorizer.fit_transform(train_texts)
    
    return vectorizer, X_train_sparse

def transform_new_data(vectorizer, new_texts):
    """
    Transforms new data (validation/test sets) using an ALREADY FITTED vectorizer.
    This guarantees that the feature space perfectly matches the training data
    and prevents data leakage.
    
    Args:
        vectorizer (CountVectorizer): The fitted vectorizer from the training step.
        new_texts (iterable): The text corpus from the validation or test set.
        
    Returns:
        scipy.sparse.csr_matrix: The transformed sparse feature matrix.
    """
    # STRICTLY .transform() only. Never .fit() or .fit_transform() here.
    X_new_sparse = vectorizer.transform(new_texts)
    
    return X_new_sparse


def fit_transform_tfidf_train(text_data, min_df=10, max_df=0.50, ngram_range=(1, 2)):
    """
    Fits a TfidfVectorizer to the training text and returns the sparse weight matrix.
    Uses continuous TF-IDF weighting instead of raw counts.
    """
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=None  # Preserves contextual bigrams like "not good"
    )
    X_sparse = vectorizer.fit_transform(text_data)
    return vectorizer, X_sparse
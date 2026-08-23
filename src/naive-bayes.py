import numpy as np
import pickle

def calculate_smoothed_log_likelihood(word_counts, total_words, n_features, alpha):
    """
    Calculates the log-likelihood of features given a class, applying 
    Laplace or Lidstone smoothing. 
    
    Args:
        word_counts (np.array): 1D array of word frequencies for a specific class.
        total_words (float/int): Total number of words in this class.
        n_features (int): Total vocabulary size (|V|).
        alpha (float): Smoothing parameter (e.g., 1.0 for Laplace).
        
    Returns:
        np.array: 1D array of smoothed log-likelihoods.
    """
    # Math: log P(w_i|c) = log( (count(w_i, c) + alpha) / (sum(count(w, c)) + alpha * |V|) )
    smoothed_counts = word_counts + alpha
    smoothed_total = total_words + (alpha * n_features)
    
    return np.log(smoothed_counts / smoothed_total)


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        """
        Multinomial Naive Bayes classifier.
        Args:
            alpha (float): Laplace/Lidstone smoothing parameter.
        """
        self.alpha = alpha
        self.classes_ = None
        
        # Stored as dictionaries for easy human inspection in notebooks
        self.log_priors_dict_ = {}      
        self.log_likelihoods_dict_ = {} 
        
        # Stored as arrays for fast NumPy matrix multiplication during predict
        self._log_priors_array = None
        self._log_likelihoods_array = None

    def fit(self, X, y):
        """
        Estimates class priors and per-feature likelihoods from the training data.
        """
        y = np.array(y)
        self.classes_ = np.unique(y)
        
        n_samples, n_features = X.shape
        
        self._log_priors_array = np.zeros(len(self.classes_))
        self._log_likelihoods_array = np.zeros((len(self.classes_), n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            
            # --- PRIOR PROBABILITY ---
            prior_c = X_c.shape[0] / n_samples
            log_prior_c = np.log(prior_c)
            
            self._log_priors_array[idx] = log_prior_c
            self.log_priors_dict_[c] = log_prior_c
            
            # --- LIKELIHOOD ---
            word_counts_c = np.asarray(X_c.sum(axis=0)).flatten()
            total_words_c = word_counts_c.sum()
            
            # Call the external smoothing function
            log_likelihood_c = calculate_smoothed_log_likelihood(
                word_counts=word_counts_c,
                total_words=total_words_c,
                n_features=n_features,
                alpha=self.alpha
            )
            
            self._log_likelihoods_array[idx, :] = log_likelihood_c
            self.log_likelihoods_dict_[c] = log_likelihood_c

    def predict(self, X):
        """
        Computes the log-space posterior scores and returns the argmax class.
        """
        # Math: score_c = log P(c) + sum( f_i * log P(w_i | c) )
        scores = (X @ self._log_likelihoods_array.T) + self._log_priors_array
        
        best_class_indices = np.argmax(scores, axis=1)
        return self.classes_[best_class_indices]

    def get_learned_parameters(self):
        return {
            "classes": self.classes_,
            "log_priors": self.log_priors_dict_,
            "log_likelihoods": self.log_likelihoods_dict_
        }

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
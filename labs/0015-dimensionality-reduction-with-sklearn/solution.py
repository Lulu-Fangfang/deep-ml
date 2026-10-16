import numpy as np
# You can use sklearn! For example:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import  check_is_fitted
# from sklearn.random_projection import GaussianRandomProjection

class MyReducer(TransformerMixin, BaseEstimator):
    """
    Implement dimensionality reduction to 10 dimensions.
    
    You have access to sklearn - explore different methods:
    - PCA (Principal Component Analysis)
    - TruncatedSVD (for sparse data)
    - GaussianRandomProjection
    - And more!
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        self.random_state = 42
        # Initialize your reducer here
    @staticmethod
    def _prepare(X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return X
    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Fit your dimensionality reduction
        X = self._prepare(X)
        self.pca_ = PCA(
            n_components = 10,
            svd_solver="randomized",
            random_state=self.random_state,
        )
        self.pca_.fit(X)
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Transform X to 10 dimensions
        check_is_fitted(self, "pca_")
        return self.pca_.transform(self._prepare(X))
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

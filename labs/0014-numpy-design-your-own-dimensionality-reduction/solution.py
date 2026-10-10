import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        # Add any attributes you need to store learned parameters

    def fit(self, X):
        """
        Learn the reduction from training data.
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # TODO: Analyze X and store what you need for transform()
        X = np.asarray(X, dtype = np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        covariance = (
            X_centered.T @ X_centered
        )/ (X.shape[0] - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        indices = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components_ = eigenvectors[: , indices]
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # TODO: Project X to 10 dimensions using parameters from fit()
        X = np.asarray(X, dtype = np.float64)

        return (X - self.mean_) @ self.components_
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

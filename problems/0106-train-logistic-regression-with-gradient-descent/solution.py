import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    # Your code here
    row, col = X.shape
    y = y.reshape(row, 1)
    X = np.column_stack((np.ones(row), X))
    theta = np.zeros((col + 1, 1))
    # b = 0.0
    losses = np.array([])
    sigmoid = lambda z_v : 1 / (1 + np.exp(-z_v))
     
    for i in range(iterations):
        y_pred = (X @ theta)
        prediction = sigmoid(y_pred)
        # p = np.clip(prediction, 1e-15, 1 - 1e-15)
        loss_cross_entropy = -np.sum(y * np.log(prediction) + (1-y) * np.log1p(-prediction))
        losses = np.append(losses, loss_cross_entropy)
        errors = prediction - y

        grad_theta = X.T @ errors
        theta -= learning_rate * grad_theta

    coefficients = theta.ravel().tolist()
    
    return (
        [round(c ,4) for c in coefficients],
        [round(loss ,4) for loss in losses],
    )

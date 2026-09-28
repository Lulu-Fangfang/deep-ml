import numpy as np


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	weights = initial_weights.astype(float).copy()
	bias = float(initial_bias)
	mse_values = []

	for _ in range(epochs):
		z = features @ weights + bias
		predictions = 1 / (1 + np.exp(-z))

		errors = predictions - labels
		mse = np.mean(errors ** 2)
		mse_values.append(round(float(mse), 4))

		d = 2 * errors * predictions * (1 - predictions)
		weight_grad = features.T @ d / len(labels)
		bia_grad = np.mean(d)

		weights -= learning_rate * weight_grad
		bias -= learning_rate * bia_grad
	
	updated_weights = np.round(weights, 4)
	updated_bias = np.round(float(bias), 4)
	return updated_weights, updated_bias, mse_values
import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	if std_dev <= 0:
		return 0
	
	coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
	expoent = -((x - mean)**2) / (2 * std_dev ** 2)
	val = coefficient * math.exp(expoent)
	return round(val,5)
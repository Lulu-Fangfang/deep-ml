import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    h_value = 0
    h_co = len(h_coeffs) - 1
    for h in h_coeffs:
        h_value += h * x ** h_co
        h_co -= 1
    g_value = 0
    g_co = len(g_coeffs) - 1
    for g in g_coeffs:
        g_value += g * x ** g_co
        g_co -= 1


    numerator_high = len(g_coeffs)-1
    numerator_d_value = 0
    idg = 0
    denumerator_high = len(h_coeffs)-1
    denumerator_d_value = 0
    idh = 0

    while numerator_high >0 :
        if numerator_high > 0:
            numerator_d_value += g_coeffs[idg] * numerator_high * x ** (numerator_high - 1)
            numerator_high -= 1
            idg += 1

    while denumerator_high >0:
        if denumerator_high > 0:
            denumerator_d_value += h_coeffs[idh] * denumerator_high * x ** (denumerator_high - 1)
            denumerator_high -= 1
            idh += 1

    result = ((numerator_d_value*h_value) - (g_value * denumerator_d_value))/(h_value**2)
    return float(result)
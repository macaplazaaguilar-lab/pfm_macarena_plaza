"""
bs_utils.py
===========

Módulo de utilidades para el modelo de Black–Scholes.

Contiene:
    1. Pricing analítico de opciones europeas.
    2. Griegas analíticas (Delta, Gamma, Vega, Rho, Theta).
    3. Simulación de trayectorias bajo el GBM riesgo-neutral.
    4. Estimadores Monte Carlo del precio.
    5. Estimadores Pathwise de Griegas (Delta, Vega, Rho).
    6. Estimadores Finite Differences de Griegas (Delta, Gamma, Vega, Rho, Theta)
       con Common Random Numbers (CRN).

Inputs:
    - CP : 'call' o 'put'.
    - tau = T - t : tiempo hasta el vencimiento.
    - sigma : volatilidad del subyacente (sigma_S en en el trabajo).
    - r : tipo de interés libre de riesgo (constante bajo BS).
"""

import numpy as np
from scipy.stats import norm


# =============================================================================
# 1. Pricing analítico Black–Scholes
# =============================================================================

def BS_Call_Put_Option_Price(CP, S0, K, sigma, t, T, r):
    """
    Precio en t de una opción europea (call o put) bajo Black–Scholes.

    Parameters
    ----------
    CP : 'call' o 'put'.
    S0 : float
        Precio del subyacente en t.
    K : float
        Strike.
    sigma : float
        Volatilidad del subyacente.
    t : float
        Tiempo actual (habitualmente 0).
    T : float
        Vencimiento.
    r : float
        Tipo libre de riesgo.

    Output
    -------
    float
        Precio de la opción.
    """
    tau = T - t

    # En vencimiento devolvemos el payoff intrínseco.
    if tau <= 0:
        if CP.lower() == "call":
            return max(S0 - K, 0.0)
        elif CP.lower() == "put":
            return max(K - S0, 0.0)
        else:
            raise ValueError("CP debe ser 'call' o 'put'")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if CP.lower() == "call":
        return norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP.lower() == "put":
        return norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S0
    else:
        raise ValueError("CP debe ser 'call' o 'put'")


# =============================================================================
# 2. Griegas analíticas Black–Scholes
# =============================================================================

def BS_delta(CP, S0, K, sigma, t, T, r):
    """Delta analítica: dV/dS0."""
    tau = T - t
    if tau <= 0:
        if CP.lower() == "call":
            return 1.0 if S0 > K else 0.0
        elif CP.lower() == "put":
            return -1.0 if S0 < K else 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * np.power(sigma,2.0)) * tau) / (sigma * np.sqrt(tau))

    if CP.lower() == "call":
        return norm.cdf(d1)
    elif CP.lower() == "put":
        return norm.cdf(d1) - 1.0
    else:
        raise ValueError("CP debe ser 'call' o 'put'")


def BS_gamma(S0, K, sigma, t, T, r):
    """Gamma analítica: d^2V/dS0^2 (igual para call y put)."""
    tau = T - t
    if tau <= 0:
        return 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.pdf(d1) / (S0 * sigma * np.sqrt(tau))


def BS_vega(S0, K, sigma, t, T, r):
    """Vega analítica: dV/dsigma (igual para call y put)."""
    tau = T - t
    if tau <= 0:
        return 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return S0 * norm.pdf(d1) * np.sqrt(tau)


def BS_rho(CP, S0, K, sigma, t, T, r):
    """Rho analítica: dV/dr."""
    tau = T - t
    if tau <= 0:
        return 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if CP.lower() == "call":
        return K * tau * np.exp(-r * tau) * norm.cdf(d2)
    elif CP.lower() == "put":
        return -K * tau * np.exp(-r * tau) * norm.cdf(-d2)
    else:
        raise ValueError("CP debe ser 'call' o 'put'")


def BS_theta(CP, S0, K, sigma, t, T, r):
    """Theta analítica: dV/dt."""
    tau = T - t
    if tau <= 0:
        return 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    term = -(S0 * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(tau))

    if CP.lower() == "call":
        return term - r * K * np.exp(-r * tau) * norm.cdf(d2)
    elif CP.lower() == "put":
        return term + r * K * np.exp(-r * tau) * norm.cdf(-d2)
    else:
        raise ValueError("CP debe ser 'call' o 'put'")


# =============================================================================
# 3. Simulación bajo GBM riesgo-neutral
# =============================================================================

def simulate_ST_exact(Z, T, r, sigma, S0):
    
    """Simulación exacta del precio final S_T bajo BS en un único paso.
    Z es un vector (o matriz) de N(0,1) i.i.d.
    Útil cuando solo necesitamos S_T y no toda la trayectoria.
    """
    return S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def GeneratePathsGBM_givenZ(Z, T, r, sigma, S0):
    """
    Genera trayectorias de GBM bajo Q a partir de una matriz Z dada.
    El uso de una Z fija permite CRN: si llamamos a esta función dos veces con Z idéntica y parámetros distintos, ambas
    simulaciones comparten el movimiento browniano subyacente, lo que reduce
    drásticamente la varianza del estimador FD de Griegas.

    Parameters
    ----------
    Z : ndarray, shape (NoOfPaths, NoOfSteps)
        Matriz de N(0,1) i.i.d.
    T, r, sigma, S0 : Parámetros del GBM.

    Outputs:
    -------
    dict con claves:
        'time' : ndarray de tiempos.
        'S'    : ndarray (NoOfPaths, NoOfSteps+1) de precios simulados."""
    
    NoOfPaths, NoOfSteps = Z.shape
    dt = T / float(NoOfSteps)

    X = np.zeros((NoOfPaths, NoOfSteps + 1))
    time = np.zeros(NoOfSteps + 1)
    X[:, 0] = np.log(S0)

    for i in range(NoOfSteps):
        Zi = Z[:, i]
        # Moment matching: estandarizar Z en cada paso para reducir varianza.
        if NoOfPaths > 1:
            Zi = (Zi - np.mean(Zi)) / np.std(Zi)

        X[:, i + 1] = X[:, i] + (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Zi
        time[i + 1] = time[i] + dt

    S = np.exp(X)
    return {"time": time, "S": S}


# =============================================================================
# 4. Estimador Monte Carlo del precio
# =============================================================================

def MC_price(CP, ST, K, r, T):
    """
    Estimador Monte Carlo del precio descontado.
    Devuelve el precio puntual y su error estándar.

    Parameters
    ----------
    CP : str
    ST : array
        Realizaciones del precio terminal.
    K, r, T : float

    Outputs
    -------
    price : float: Estimador Monte Carlo.
    se : float: Error estándar = std muestral / sqrt(N).
    """
    if CP.lower() == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif CP.lower() == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("CP debe ser 'call' o 'put'")

    discounted = np.exp(-r * T) * payoff
    price = np.mean(discounted)
    se = np.std(discounted, ddof=1) / np.sqrt(len(discounted))

    return price, se


def opcion_europea(CP, ST, K, r, T):
    """Versión simple del estimador MC: devuelve solo el precio puntual."""
    price, _ = MC_price(CP, ST, K, r, T)
    return price


# =============================================================================
# 5. Estimadores Pathwise
# =============================================================================

def PathwiseDelta(CP, S0, S, K, r, T):
    """
    Estimador pathwise de Delta para una opción europea.
        Para call: Delta = exp(-rT) * E[1_{S_T > K} * (S_T / S0)]
        Para put: Delta = -exp(-rT) * E[1_{S_T < K} * (S_T / S0)]

    Input
    ----------
    S : ndarray (NoOfPaths, NoOfSteps+1)
        Trayectorias simuladas. Solo se usa el valor en T (final).
    """
    ST = S[:, -1]

    if CP.lower() == "call":
        indicator = (ST > K).astype(float)
        return np.exp(-r * T) * np.mean((ST / S0) * indicator)
    elif CP.lower() == "put":
        indicator = (ST < K).astype(float)
        return -np.exp(-r * T) * np.mean((ST / S0) * indicator)
    else:
        raise ValueError("CP debe ser 'call' o 'put'")


def PathwiseVega(CP, S0, S, K, r, T, sigma):
    """
    Estimador pathwise de Vega para una opción europea.
    Vega = exp(-rT) * E[1_{ITM} * dS_T/dsigma] con dS_T/dsigma = S_T * (W_T - sigma*T), donde W_T se reconstruye desde S_T:
        W_T = (log(S_T/S_0) - (r - 0.5*sigma^2)*T) / sigma
    """
    ST = S[:, -1]

    if CP.lower() == "call":
        indicator = (ST > K).astype(float)
    elif CP.lower() == "put":
        indicator = (ST < K).astype(float)
    else:
        raise ValueError("CP debe ser 'call' o 'put'")

    # Reconstrucción de W_T (signo: r - 0.5*sigma^2 en el drift bajo Q).
    WT = (np.log(ST / S0) - (r - 0.5 * np.power(sigma,2)) * T) / sigma

    # dS_T/dsigma = S_T * (W_T - sigma*T)
    return np.exp(-r * T) * np.mean(indicator * ST * (WT - sigma * T))


def PathwiseRho(CP, S0, S, K, r, T):
    """Estimador pathwise de Rho para una opción europea.
    Se obtiene de derivar e^{-rT}*payoff respecto a r, usando que
    dS_T/dr = T * S_T.
    """
    ST = S[:, -1]

    if CP.lower() == "call":
        indicator = (ST > K).astype(float)
        payoff = np.maximum(ST - K, 0.0)
        rho_paths = T * indicator * ST - T * payoff
    elif CP.lower() == "put":
        indicator = (ST < K).astype(float)
        payoff = np.maximum(K - ST, 0.0)
        rho_paths = -T * indicator * ST - T * payoff
        
    else:
        raise ValueError("CP debe ser 'call' o 'put'")

    return np.exp(-r * T) * np.mean(rho_paths)


# =============================================================================
# 6. Estimadores Finite Differences (con CRN)
# =============================================================================
# Todos los estimadores FD reciben Z explícitamente para garantizar CRN entre las simulaciones perturbadas.

def FD_Delta(CP, S0, K, sigma, T, r, h, Z):
    """FD centrada de Delta: perturbamos S0."""
    paths_up = GeneratePathsGBM_givenZ(Z, T, r, sigma, S0 + h)
    paths_dn = GeneratePathsGBM_givenZ(Z, T, r, sigma, S0 - h)

    ST_up = paths_up["S"][:, -1]
    ST_dn = paths_dn["S"][:, -1]

    price_up = opcion_europea(CP, ST_up, K, r, T)
    price_dn = opcion_europea(CP, ST_dn, K, r, T)

    return (price_up - price_dn) / (2 * h)


def FD_Gamma(CP, S0, K, sigma, T, r, h, Z):
    """FD centrada de Gamma: segunda diferencia simétrica."""
    paths_up = GeneratePathsGBM_givenZ(Z, T, r, sigma, S0 + h)
    paths_md = GeneratePathsGBM_givenZ(Z, T, r, sigma, S0)
    paths_dn = GeneratePathsGBM_givenZ(Z, T, r, sigma, S0 - h)

    price_up = opcion_europea(CP, paths_up["S"][:, -1], K, r, T)
    price_md = opcion_europea(CP, paths_md["S"][:, -1], K, r, T)
    price_dn = opcion_europea(CP, paths_dn["S"][:, -1], K, r, T)

    return (price_up - 2 * price_md + price_dn) / (h ** 2)


def FD_Vega(CP, S0, K, sigma, T, r, h, Z):
    """FD centrada de Vega: perturbamos sigma."""
    paths_up = GeneratePathsGBM_givenZ(Z, T, r, sigma + h, S0)
    paths_dn = GeneratePathsGBM_givenZ(Z, T, r, sigma - h, S0)

    ST_up = paths_up["S"][:, -1]
    ST_dn = paths_dn["S"][:, -1]

    price_up = opcion_europea(CP, ST_up, K, r, T)
    price_dn = opcion_europea(CP, ST_dn, K, r, T)

    return (price_up - price_dn) / (2 * h)


def FD_Rho(CP, S0, K, sigma, T, r, h, Z):
    """
    FD centrada de Rho: perturbamos r.
    Nota: r aparece tanto en el drift como en el descuento. Hay que perturbar en los dos lugares.
    """
    paths_up = GeneratePathsGBM_givenZ(Z, T, r + h, sigma, S0)
    paths_dn = GeneratePathsGBM_givenZ(Z, T, r - h, sigma, S0)

    ST_up = paths_up["S"][:, -1]
    ST_dn = paths_dn["S"][:, -1]

    price_up = opcion_europea(CP, ST_up, K, r+h, T)
    price_dn = opcion_europea(CP, ST_dn, K, r-h, T)

    return (price_up - price_dn) / (2 * h)


def FD_Theta(CP, S0, K, sigma, T, r, h, Z):
    """
    FD centrada de Theta perturbando T.
        Theta = -[V(0, T+h) - V(0, T-h)] / (2h)   """
    
    paths_up = GeneratePathsGBM_givenZ(Z, T + h, r, sigma, S0)
    paths_dn = GeneratePathsGBM_givenZ(Z, T - h, r, sigma, S0)

    ST_up = paths_up["S"][:, -1]
    ST_dn = paths_dn["S"][:, -1]
    
    price_up = opcion_europea(CP, ST_up, K, r, T + h)
    price_dn = opcion_europea(CP, ST_dn, K, r, T - h)

    return -(price_up - price_dn) / (2 * h)

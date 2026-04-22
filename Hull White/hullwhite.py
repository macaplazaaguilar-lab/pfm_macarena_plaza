import numpy as np

def P0T(T):
    r_flat=0.05
    return np.exp(-r_flat * T)


def GeneratePathsHWAndAssetEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta, S0, sigma_S, rho, seed):
    """
    Simula trayectorias de:
    - r_t: bajo Hull–White
    - S_t: activo financiero bajo medida riesgo-neutral

    NoOfPaths : Número de trayectorias Monte Carlo.
    NoOfSteps : Número de pasos temporales.
    T : Horizonte temporal.
    P0T : Curva inicial de descuento P(0,T).
    lambd : Velocidad de reversión a la media del short rate.
    eta : Volatilidad del short rate.
    S0 : Precio inicial del activo.
    sigma_S : Volatilidad del activo.
    rho : Correlación entre el shock del short rate y el shock del activo.

    Devuelve
        "R": trayectorias del short rate
        "S": trayectorias del activo
    """

    if seed is not None:
        np.random.seed(seed)

    dt_diff = 1e-4

    f0T = lambda t: -(np.log(P0T(t + dt_diff)) - np.log(P0T(t - dt_diff))) / (2 * dt_diff)

    r0 = f0T(1e-5)

    theta = lambda t: (1.0 / lambd * (f0T(t + dt_diff) - f0T(t - dt_diff)) / (2.0 * dt_diff)
        + f0T(t)
        + eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t)))

    # =========================

    dt = T / float(NoOfSteps)

    time = np.zeros([NoOfSteps + 1])

    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    S = np.zeros([NoOfPaths, NoOfSteps + 1])

    R[:, 0] = r0
    S[:, 0] = S0

     # SIMULACIÓN

    for i in range(NoOfSteps):

        # Normales independientes
        Z1 = np.random.normal(0.0, 1.0, NoOfPaths)
        Z2 = np.random.normal(0.0, 1.0, NoOfPaths)

        # Normalización opcional
        if NoOfPaths > 1:
            Z1 = (Z1 - np.mean(Z1)) / np.std(Z1)
            Z2 = (Z2 - np.mean(Z2)) / np.std(Z2)

        # Correlacionamos
        Z_r = Z1
        Z_S = rho * Z1 + np.sqrt(1.0 - rho**2) * Z2

        # Short rate:
        R[:, i + 1] = (
            R[:, i]
            + lambd * (theta(time[i]) - R[:, i]) * dt
            + eta * np.sqrt(dt) * Z_r)

        # Activo:
        S[:, i + 1] = (
            S[:, i]
            * np.exp((R[:, i] - 0.5 * sigma_S**2) * dt + sigma_S * np.sqrt(dt) * Z_S))

        time[i + 1] = time[i] + dt

    return {"time": time, "R": R, "S": S}

def PriceEuropeanOptionHWMC(paths, K, T, option_type):

    R = paths["R"]
    S = paths["S"]

    NoOfSteps = R.shape[1] - 1
    dt = T / NoOfSteps

    # Aproximación de la integral del short rate
    integral_r = np.sum(R[:, :-1] * dt, axis=1)

    # Descuento estocástico
    discount = np.exp(-integral_r)

    # Precio final del activo
    ST = S[:, -1]

    # Selección del payoff según el tipo de opción
    option_type = option_type.lower()

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type debe ser 'call' o 'put'.")

    discounted_payoff = discount * payoff

    price = np.mean(discounted_payoff)
    stderr = np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff))

    return price, stderr
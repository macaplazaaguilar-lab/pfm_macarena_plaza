import numpy as np


# ============================================================
# Curva de descuento inicial
# ============================================================

def P0T(T):
    """
    Curva de descuento plana usada en los experimentos numericos.

        P(0,T) = exp(-r_flat * T),    r_flat = 0.05

    Equivale a una curva forward instantanea constante
    f^M(0,t) = r_flat, lo que simplifica la calibracion de theta(t)
    en el modelo Hull-White (ver Seccion 4.2 de la tesis).
    """
    r_flat = 0.05
    return np.exp(-r_flat * T)


# ============================================================
# Generador de trayectorias conjuntas (r_t, S_t) en Hull-White
# ============================================================

def GeneratePathsHWAndAssetEuler(NoOfPaths, NoOfSteps, T, P0T,
                                 lambd, eta, S0, sigma_S, rho, seed):
    """
    Simula trayectorias conjuntas del par (r_t, S_t) bajo la medida
    riesgo-neutral Q en el modelo Hull-White con activo subyacente:

        dr_t = lambda * (theta(t) - r_t) * dt + eta * dW_t^r
        dS_t = r_t * S_t * dt + sigma_S * S_t * dW_t^S
        <dW^S, dW^r> = rho * dt

    Convencion de la descomposicion de Cholesky (alineada con la
    Seccion 4.4 de la tesis):

        Delta W^r = sqrt(dt) * Z^(1)
        Delta W^S = sqrt(dt) * ( rho * Z^(1) + sqrt(1 - rho^2) * Z^(2) )

    con Z^(1), Z^(2) iid N(0,1).

    Parametros
    ----------
    NoOfPaths : int
        Numero de trayectorias Monte Carlo.
    NoOfSteps : int
        Numero de pasos temporales de la discretizacion.
    T : float
        Horizonte temporal.
    P0T : callable
        Curva inicial de descuento, P0T(t) -> precio de un bono cupon
        cero de vencimiento t. Se usa para calibrar theta(t).
    lambd : float
        Velocidad de reversion a la media del tipo corto.
    eta : float
        Volatilidad del tipo corto en Hull-White (NO confundir con sigma_S).
    S0 : float
        Precio inicial del activo subyacente.
    sigma_S : float
        Volatilidad del activo subyacente.
    rho : float
        Correlacion browniana entre tipo corto y activo, rho_{S,r} in (-1,1).
    seed : int or None
        Semilla del generador. Si None, se usa entropia del sistema.

    Devuelve
    --------
    dict con claves:
        "time" : array (NoOfSteps+1,)
            Malla temporal [0, T].
        "R" : array (NoOfPaths, NoOfSteps+1)
            Trayectorias del tipo corto.
        "S" : array (NoOfPaths, NoOfSteps+1)
            Trayectorias del activo subyacente.
    """

    # Generador moderno (consistente con el resto del proyecto)
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Calibracion de theta(t) a la curva inicial
    # --------------------------------------------------------
    # Forward instantanea inicial via derivada numerica de -log P(0,t)
    dt_diff = 1e-4
    f0T = lambda t: -(np.log(P0T(t + dt_diff)) - np.log(P0T(t - dt_diff))) / (2.0 * dt_diff)
    r0 = f0T(1e-5)

    # theta(t) calibrada a la curva: ver formula (4.x) de la tesis.
    # Con curva plana, theta(t) se reduce a una constante igual a r_flat.
    theta = lambda t: (
        (f0T(t + dt_diff) - f0T(t - dt_diff)) / (2.0 * dt_diff) / lambd
        + f0T(t)
        + eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    )

    # --------------------------------------------------------
    # Discretizacion temporal
    # --------------------------------------------------------
    dt = T / float(NoOfSteps)
    time = np.linspace(0.0, T, NoOfSteps + 1)

    R = np.zeros((NoOfPaths, NoOfSteps + 1))
    S = np.zeros((NoOfPaths, NoOfSteps + 1))
    R[:, 0] = r0
    S[:, 0] = S0

    # --------------------------------------------------------
    # Bucle de simulacion (Euler-Maruyama + log-Euler)
    # --------------------------------------------------------
    sqrt_dt = np.sqrt(dt)
    sqrt_1mrho2 = np.sqrt(1.0 - rho * rho)

    for i in range(NoOfSteps):
        # Normales independientes
        Z1 = rng.standard_normal(NoOfPaths)
        Z2 = rng.standard_normal(NoOfPaths)

        # Brownianos correlacionados (Cholesky):
        #   W^r impulsa al tipo corto (componente "independiente")
        #   W^S se compone de la parte correlacionada con W^r y una ortogonal
        dW_r = sqrt_dt * Z1
        dW_S = sqrt_dt * (rho * Z1 + sqrt_1mrho2 * Z2)

        # Euler-Maruyama para r_t (SDE lineal, admite tipos negativos)
        R[:, i + 1] = (R[:, i]
                       + lambd * (theta(time[i]) - R[:, i]) * dt
                       + eta * dW_r)

        # Log-Euler para S_t (proviene de aplicar Ito a ln S_t,
        # garantiza S_t > 0 para todo paso temporal)
        S[:, i + 1] = S[:, i] * np.exp(
            (R[:, i] - 0.5 * sigma_S * sigma_S) * dt + sigma_S * dW_S
        )

    return {"time": time, "R": R, "S": S}


# ============================================================
# Estimador Monte Carlo del precio con descuento estocastico
# ============================================================

def PriceEuropeanOptionHWMC(paths, K, T, option_type):
    """
    Estima el precio de una opcion europea por Monte Carlo en el
    modelo Hull-White conjunto, usando descuento estocastico:

        V_0 = E^Q [ exp( - int_0^T r_t dt ) * Phi(S_T) ]

    La integral del tipo corto se aproxima por la regla del punto
    izquierdo sobre la malla de simulacion:

        int_0^T r_t dt  ~  sum_{k=0}^{M-1} r_{t_k} * dt

    El uso de r_{t_k} (y no r_{t_{k+1}}) es coherente con la
    adaptabilidad de la filtracion: r_{t_k} es F_{t_k}-medible.

    Parametros
    ----------
    paths : dict
        Diccionario devuelto por GeneratePathsHWAndAssetEuler,
        con claves "R" y "S".
    K : float
        Precio de ejercicio.
    T : float
        Vencimiento.
    option_type : str
        "call" o "put".

    Devuelve
    --------
    price : float
        Estimador Monte Carlo del precio.
    stderr : float
        Error estandar del estimador (desviacion tipica muestral / sqrt(N)).
    """

    R = paths["R"]
    S = paths["S"]

    NoOfSteps = R.shape[1] - 1
    dt = T / NoOfSteps

    # Aproximacion del factor de descuento estocastico
    # (regla del punto izquierdo: se toman r_{t_0}, ..., r_{t_{M-1}})
    integral_r = np.sum(R[:, :-1] * dt, axis=1)
    discount = np.exp(-integral_r)

    # Precio final del activo
    ST = S[:, -1]

    # Payoff segun tipo de opcion
    option_type = option_type.lower()
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type debe ser 'call' o 'put'.")

    discounted_payoff = discount * payoff

    price = float(np.mean(discounted_payoff))
    stderr = float(np.std(discounted_payoff, ddof=1) / np.sqrt(len(discounted_payoff)))

    return price, stderr

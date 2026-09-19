"""
Traduzione Python del solver MATLAB per l'integrale di collisione del modello MMT
(Majda-McLaughlin-Tabak), a 4 onde, nello spazio delle frequenze omega=sqrt(k).

Fedele riga per riga al codice MATLAB originale (cartella integratore_MMT/):
BuildBasicVarsFun, CalcFlux, FluxAB, P2/P31/P32Integrand, Interaction, V, Weight,
PowerLaw, FindPow, IntegratePower.

Non tradotto: BarycentricIntegralFeeder — nello script originale barryon=False
sempre, quindi quell'integratore non viene mai usato a runtime; qui si usa
sempre np.trapz, come fa il codice MATLAB nel ramo effettivamente eseguito.
"""

import numpy as np

# np.trapz è stato rimosso in NumPy >= 2.0 in favore di np.trapezoid;
# questo wrapper garantisce compatibilità con entrambe le versioni.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Blocchi elementari (V.m, Weight.m, PowerLaw.m, Interaction.m)
# ---------------------------------------------------------------------------

def power_law(p, scale, x):
    """PowerLaw.m — modello n(k) = scale * k^p."""
    return scale * x ** p


def v_coeff(om1, om2, om3, om, beta):
    """V.m — coefficiente di interazione."""
    return (om1 * om2 * om3 * om) ** (-beta)


def weight(om1, om2, type_):
    """Weight.m — peso energia/azione."""
    if type_ == "Action":
        out = np.zeros_like(om1) + 0.5
    elif type_ == "Energy":
        out = om1 / (om1 + om2)
        # replica: if om1(1)==0 && om2(1)==0 -> OUT(1)=0
        if np.ndim(om1) > 0 and om1[0] == 0 and om2[0] == 0:
            out = out.copy()
            out[0] = 0.0
    else:
        raise ValueError(f"type sconosciuto: {type_}")
    return out


def interaction(om1, om2, om3, om4, n_func, n_params):
    """Interaction.m — termine di interazione della WKE (4 onde)."""
    om1 = om1 ** 2
    om2 = om2 ** 2
    om3 = om3 ** 2
    om4 = om4 ** 2

    n1 = n_func(*n_params, om1)
    n2 = n_func(*n_params, om2)
    n3 = n_func(*n_params, om3)
    n4 = n_func(*n_params, om4)

    return n1 * n2 * (n3 + n4) - (n2 + n1) * n3 * n4


# ---------------------------------------------------------------------------
# Correzione di singolarità (FindPow.m, IntegratePower.m)
# ---------------------------------------------------------------------------

def find_pow(f, x, i, location):
    """FindPow.m — stima esponente/scala dell'andamento vicino a un bordo."""
    xfind = np.flip(location - x)
    F = np.abs(f)
    if F[1] > F[0]:
        F = np.flip(F)

    slope = np.polyfit(np.log(xfind[:i]), np.log(F[:i]), 1)[0]
    est_func = xfind[0] ** slope
    scale = np.sign(f[0]) * F[0] / est_func
    return slope, scale


def integrate_power(a, b, p, scale, shift):
    """IntegratePower.m — integrale analitico di scale*(shift-x)^p tra a e b."""
    return -(scale / (p + 1)) * ((shift - b) ** (p + 1) - (shift - a) ** (p + 1))


# ---------------------------------------------------------------------------
# Integrandi (P2Integrand.m, P31Integrand.m, P32Integrand.m)
# ---------------------------------------------------------------------------

def p2_integrand(bv, y, om):
    a2 = bv["a2"]
    yi = 1.0 / y
    om1_ = (1 - yi + np.sqrt((yi - 1) * (yi + 3))) / 2
    om12 = om1_ + yi - 1

    W = weight(yi, om1_, bv["type"])
    v_part = v_coeff(om1_, yi, om12, 1, bv["beta"])
    power_part = yi ** 2 * (om1_ * yi * om12)
    interact = interaction(yi * om * a2, om1_ * om * a2, om12 * om * a2, om * a2,
                            bv["n_func"], bv["n_params"])
    delt_term = 1.0 / (2 * (2 * om1_ + yi - 1))
    return W, v_part, power_part, interact, delt_term


def p31_integrand(bv, y, om):
    a2 = bv["a2"]
    yi = 1.0 / y
    om2_ = yi / (yi - 1)
    om12 = om2_ + yi - 1

    W = weight(yi, om2_, bv["type"])
    v_part = v_coeff(yi, om2_, om12, 1, bv["beta"])
    power_part = yi ** 2 * (yi * om2_ * om12)
    interact = interaction(yi * om * a2, om2_ * om * a2, om12 * om * a2, om * a2,
                            bv["n_func"], bv["n_params"])
    delt_term = 1.0 / (2 * (yi - 1))
    return W, v_part, power_part, interact, delt_term


def p32_integrand(bv, y, om):
    a2 = bv["a2"]
    yi = 1.0 / y
    om2_ = yi / (yi - 1)
    om12 = om2_ + yi - 1

    W = weight(om2_, yi, bv["type"])
    v_part = v_coeff(yi, om2_, om12, 1, bv["beta"])
    power_part = yi ** 2 * (yi * om2_ * om12)
    interact = interaction(yi * om * a2, om2_ * om * a2, om12 * om * a2, om * a2,
                            bv["n_func"], bv["n_params"])
    delt_term = 1.0 / (2 * (yi - 1))
    return W, v_part, power_part, interact, delt_term


# ---------------------------------------------------------------------------
# Setup (BuildBasicVarsFun.m) e calcolo del flusso (CalcFlux.m)
# ---------------------------------------------------------------------------

def build_basic_vars(A, B, N, IntN, beta, n_func, n_params, type_):
    """BuildBasicVarsFun.m — costruisce il dizionario di variabili di base.

    A, B: estremi del dominio (in spazio omega=sqrt(k)), es. A=[a1,a2], B=[b1,b2]
    """
    a1, a2 = A
    b1, b2 = B
    oma, omb = a1 / a2, 1.0

    om1 = a2 / b1
    om2 = a2 / b2
    om = np.linspace(oma, omb, N)

    return {
        "om": om, "N": N, "Om1": om1, "Om2": om2, "beta": beta, "type": type_,
        "IntN": IntN, "n_func": n_func, "n_params": n_params, "a2": a2,
    }


def calc_flux(bv):
    """CalcFlux.m — calcola il flusso integrando su omega e sui due sotto-integrali."""
    om = bv["om"]
    Om1, Om2, a2 = bv["Om1"], bv["Om2"], bv["a2"]
    beta = bv["beta"]

    om_power = 5 - 4 * beta
    om_correction_power = 0  # come nel MATLAB: sovrascritto a 0

    n = len(om)
    contain2 = np.zeros(n)
    contain31 = np.zeros(n)
    contain32 = np.zeros(n)
    int_n = bv["IntN"]

    for i in range(n):
        omloc = om[i]
        om_om1 = Om1 * omloc
        om_om2 = Om2 * omloc
        om_p_cor = (a2 * omloc) ** om_correction_power
        with np.errstate(divide="ignore", invalid="ignore"):
            om_p_true = (a2 * omloc) ** (om_power - om_correction_power) if omloc != 0 else 0.0

        # --- P2 ---
        if om_om1 == om_om2:
            contain2[i] = 0.0
        else:
            y2 = np.linspace(om_om2, om_om1, int_n)
            if y2[0] == 0:
                y2[0] = 0.5 * (y2[0] + y2[1])
            if y2[-1] == 1:
                y2[-1] = 0.5 * (y2[-2] + y2[-1])

            W, VPart, PowerPart, Interact, DTerm = p2_integrand(bv, y2, omloc)
            y_terms = W * VPart * PowerPart * DTerm
            om_terms = om_p_cor * Interact
            integrand = y_terms * om_terms

            p2_cor = np.zeros_like(y2)
            p2_cor_val = 0.0
            if om_om1 > 0.9:
                slope, scale = find_pow(integrand, y2, 2, 1)
                if -1 < slope < 0:
                    p2_cor = scale * (1 - y2) ** slope
                    p2_cor_val = integrate_power(y2[0], y2[-1], slope, scale, 1)

            contain2[i] = om_p_true * (_trapz(integrand - p2_cor, y2) + p2_cor_val)

        # --- P31 ---
        lo31, hi31 = min(0.5, om_om2), min(0.5, om_om1)
        if lo31 == hi31:
            contain31[i] = 0.0
        else:
            y31 = np.linspace(lo31, hi31, int_n)
            if y31[0] == 0:
                y31[0] = 0.5 * (y31[0] + y31[1])
            W, VPart, PowerPart, Interact, DTerm = p31_integrand(bv, y31, omloc)
            integrand = W * VPart * PowerPart * DTerm * om_p_cor * Interact
            contain31[i] = om_p_true * _trapz(integrand, y31)

        # --- P32 ---
        lo32, hi32 = min(0.5, 1 - om_om1), min(0.5, 1 - om_om2)
        if lo32 == hi32:
            contain32[i] = 0.0
        else:
            y32 = np.linspace(lo32, hi32, int_n)
            if y32[0] == 0:
                y32[0] = 0.5 * (y32[0] + y32[1])
            W, VPart, PowerPart, Interact, DTerm = p32_integrand(bv, y32, omloc)
            integrand = W * VPart * PowerPart * DTerm * om_p_cor * Interact
            contain32[i] = om_p_true * _trapz(integrand, y32)

    om_integrand = -128 * np.pi * a2 * (contain2 + contain31 + contain32)
    om = om[:-1]
    om_integrand = om_integrand[:-1]

    if bv["type"] == "Action":
        with np.errstate(divide="ignore", invalid="ignore"):
            om_integrand = ((a2 * om) ** (-1)) * om_integrand
        om_integrand[0] = 0.0

    om_cor = np.zeros_like(om)
    om_cor_val = 0.0
    slope_om, scale_om = find_pow(om_integrand, om, 2, 1)
    if -1 < slope_om < 0:
        om_cor = scale_om * (1 - om) ** slope_om
        om_cor_val = integrate_power(om[0], om[-1], slope_om, scale_om, 1)

    flux = _trapz(om_integrand - om_cor, om) + om_cor_val
    return flux


def flux_ab(n_func, n_params, A_in, B_in, beta, type_, N=2000, IntN=1000):
    """FluxAB.m — flusso attraverso il dominio [A_in -> B_in] (in spazio k).

    A_in, B_in: array (n_calc, 2), o singola coppia [a1,a2]/[b1,b2]. Uso np.inf per infinito.
    """
    A_in = np.atleast_2d(A_in).astype(float)
    B_in = np.atleast_2d(B_in).astype(float)

    flux = np.zeros(A_in.shape[0])
    for i in range(A_in.shape[0]):
        A = np.sqrt(A_in[i])
        B = np.sqrt(B_in[i])
        if A[0] == A[1]:
            flux[i] = 0.0
        else:
            bv = build_basic_vars(A, B, N, IntN, beta, n_func, n_params, type_)
            flux[i] = calc_flux(bv)
    return flux

"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import re
from scipy.signal import find_peaks




# --------------------------
# constants / geometry 
# --------------------------
xlength = 200.0
ylength = 20.0
zlength = 20.0
x_act = 0.0
x_s   = 0.5 * xlength   # 100 mm

# material
rho = 1.780E-3
nu  = 0.3
E   = 100E3
G   = E / (2.0 * (1.0 + nu))
wave_modulus = E / ((1 + nu) * (1 - 2 * nu))
lam = wave_modulus * nu
k1  = lam + 2 * G
wave_speed_elastic = (wave_modulus * (1.0 - nu) / rho) ** 0.5

# --------------------------
# load CSV
# --------------------------
csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

t_np  = df["t_ms"].to_numpy()
u_act = df["u_act_mm"].to_numpy()
u_s   = df["u_s_mm"].to_numpy()

frequency = float(df["frequency_kHz"].iloc[0])
eta       = float(df["eta"].iloc[0])
xi        = float(df["xi"].iloc[0])
wave_tag  = str(df["wave_type"].iloc[0])
Dim       = int(df["Dim"].iloc[0])
fname = os.path.basename(csv_path)

m_nx = re.search(r"_nx(\d+)", fname)
m_nt = re.search(r"_nt(\d+)", fname)

if m_nx is None or m_nt is None:
    raise ValueError("nx or nt not found in filename. Expected '_nx###' and '_nt###'.")

nx = int(m_nx.group(1))
nt = int(m_nt.group(1))

print(f"Detected nx = {nx}, nt = {nt} from filename")


c  = eta
c1 = xi

if eta == 0 and xi == 0:
    model_str = r"Elastic medium"
elif eta != 0 and xi == 0:
    model_str = fr"c={eta}"
elif eta == 0 and xi != 0:
    model_str = fr"c_1={xi}"
else:
    model_str = fr"c={eta},\;c_1={xi}"


# output paths next to CSV
base = os.path.splitext(csv_path)[0]
out_png  = base + "_TIME_FEM.png"
out_info = base + "_INFO.csv"

def robust_peak_selection(t, u, frac=0.3, label="signal", auto=True):
    """
    Robust positive-peak detection using Gaussian ML statistics.
    Manual selection optional.
    """

    # --- positive peaks only ---
    peaks, _ = find_peaks(u)
    if len(peaks) == 0:
        raise RuntimeError(f"No positive peaks found for {label}")

    umax = np.max(u)
    peaks = peaks[u[peaks] >= frac * umax]
    if len(peaks) == 0:
        raise RuntimeError(f"No peaks above threshold for {label}")

    amps = u[peaks]

    # --- robust ML estimate ---
    mu0 = np.median(amps)
    mad = np.median(np.abs(amps - mu0))
    sigma0 = 1.4826 * mad if mad > 0 else np.std(amps, ddof=1)
    sigma0 = max(sigma0, 1e-16)

    mu = mu0
    for _ in range(3):
        w = np.exp(-0.5 * ((amps - mu) / sigma0) ** 2)
        mu = np.sum(w * amps) / np.sum(w)

    sigma = np.sqrt(np.sum(w * (amps - mu) ** 2) / np.sum(w))
    rel_amp_err = sigma / mu if mu > 0 else 0.0

    idx_ml = np.argmin(np.abs(amps - mu))
    p_ml = peaks[idx_ml]

    # ---------------- PLOT ----------------
    plt.figure(figsize=(12, 4))
    plt.plot(t, u, lw=2, label=label)
    plt.scatter(t[peaks], u[peaks], s=80, color="tab:orange", label="candidate peaks")
    plt.scatter(
        t[p_ml], u[p_ml],
        s=160, color="red", edgecolor="black",
        zorder=10, label="suggested (ML)"
    )

    plt.title(
        f"{label} – peak selection\n"
        f"μ={mu:.3e}, σ={sigma:.3e}, rel_err={rel_amp_err:.2%}"
    )
    plt.xlabel("Time [ms]")
    plt.ylabel("Displacement [mm]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------------- CONSOLE ----------------
    print(f"\nCandidate peaks for {label}:")
    for i, p in enumerate(peaks):
        tag = " <--- suggested" if i == idx_ml else ""
        print(f"[{i:2d}] t={t[p]:.6e} ms | u={u[p]:+.6e} mm{tag}")

    # ---------------- SELECT ----------------
    if auto:
        idx = idx_ml
    else:
        choice = input("\nChoose peak index (ENTER = suggested): ").strip()
        idx = idx_ml if choice == "" else int(choice)

    p = peaks[idx]

    return {
        "t": t[p],
        "u": u[p],
        "rel_amp_err": rel_amp_err,
        "peak_index": idx
    }


AUTO_PEAK = False
if len(sys.argv) > 2:
    AUTO_PEAK = (sys.argv[2].lower() != "manual")


# --------------------------
# robust peak detection (ML)
# --------------------------

p_s = robust_peak_selection(
    t_np,
    u_s,
    frac=0.20,
    label="Sensor",
    auto=AUTO_PEAK
)

def stable_amplitude_from_peaks(t, u, frac=0.1, n_last=8):
    peaks, _ = find_peaks(u)
    umax = np.max(u)
    peaks = peaks[u[peaks] >= frac * umax]

    if len(peaks) < n_last:
        raise RuntimeError("Not enough peaks for stable amplitude")

    amps = u[peaks][-n_last:]
    return np.median(amps), peaks[-n_last:]



# positive peaks actuator
p_act = robust_peak_selection(
    t_np,
    u_act,
    frac=0.30,
    label="Actuator",
    auto=AUTO_PEAK
)

if frequency >= 750.0:
    A_act, _ = stable_amplitude_from_peaks(
        t_np, u_act, frac=0.30, n_last=8
    )
    p_act["u"] = A_act


REL_AMP_ERR = p_s["rel_amp_err"]


TOF = p_s["t"] - p_act["t"]
vp_FEM = x_s / TOF
alpha_FEM = (1.0 / x_s) * np.log(abs(p_act["u"]) / abs(p_s["u"]))




# --------------------------
# FEM numerical error models
# --------------------------

def velocity_error(TOF, frequency, x_s, nx):
    """
    Error on FEM phase velocity v = x / TOF
    using mesh-based spatial resolution
    """
    TOF = max(TOF, 1e-12)

    dx = xlength / nx              # spatial resolution [mm]
    dt = 0.5 / (frequency*nt)      # temporal resolution [ms]

    dv_space = dx / TOF
    dv_time  = x_s / (TOF**2) * dt

    return dv_space + dv_time


def attenuation_error(u_act, u_s, x_s, nx, rel_amp_err):
    """
    Total error on FEM attenuation:
        alpha = ln(|u_act/u_s|) / x_s

    Contributions:
    - spatial discretization error (mesh)
    - amplitude statistical error (REL_AMP_ERR = sigma / mu)

    Returned value is the quadratic sum of the two.
    """

    # safety
    u_act = max(abs(u_act), 1e-12)
    u_s   = max(abs(u_s),   1e-12)
    x_s   = max(x_s, 1e-12)

    # -------------------------
    # 1) Spatial discretization
    # -------------------------
    dx = xlength / nx               # [mm]
    ln_ratio = np.log(u_act / u_s)

    dalpha_dx = abs(ln_ratio) / (x_s**2)
    err_space = dalpha_dx * dx

    # -------------------------
    # 2) Amplitude dispersion
    # -------------------------
    # propagation of relative amplitude error
    err_amp = (1.0 / x_s) * np.sqrt(2.0) * rel_amp_err

    # -------------------------
    # Total (quadratic sum)
    # -------------------------
    return np.sqrt(err_space**2 + err_amp**2)



# FEM numerical errors

vp_FEM_err = velocity_error(
    TOF=TOF,
    frequency=frequency,
    x_s=x_s,
    nx=nx
)

alpha_FEM_err = attenuation_error(
    u_act=p_act["u"],
    u_s=p_s["u"],
    x_s=x_s,
    nx=nx,
    rel_amp_err=REL_AMP_ERR
)




# ------------------------------------------------
# analytical k(w), Phase velocity and attenuation
# ------------------------------------------------
w = 2.0 * np.pi * frequency

def kw_elastic(w, rho, k1):
    return w / np.real(wave_speed_elastic)

def kw_visc(w, rho, k1, c):
    wc = c / rho
    z  = np.sqrt(rho / k1)
    rekw = w * z * (1 + (wc / w)**2)**0.25 * np.cos(-0.5 * np.arctan(wc / w))
    imkw = w * z * (1 + (wc / w)**2)**0.25 * np.sin(-0.5 * np.arctan(wc / w))
    return rekw + 1j*imkw

def kw_grad(w, rho, k1, c1):
    wc1 = k1 / c1
    z  = np.sqrt(rho / k1)
    rekw = w * z * (1/(1 + (w / wc1)**2))**0.25 * np.cos(-0.5 * np.arctan(w / wc1))
    imkw = w * z * (1 + (w / wc1)**2)**0.25 * np.sin(-0.5 * np.arctan(w / wc1))
    return rekw + 1j*imkw

def kw_general(w, rho, k1, c, c1):
    if eta == 0 and xi == 0:
        return kw_elastic(w, rho, k1)
    elif eta != 0 and xi == 0:
        return kw_visc(w, rho, k1, c)
    elif xi != 0 and eta == 0:
        return kw_grad(w, rho, k1, c1)
    else:
        raise ValueError("Choose elastic (eta=xi=0), viscous (eta>0, xi=0) or gradient (xi>0, eta=0).")

kw0 = kw_general(w, rho, k1, c, c1)
vp_ANA = w / np.real(kw0)
alpha_ANA = -np.imag(kw0)

vp_err_pct    = 100.0 * (vp_FEM - vp_ANA) / vp_ANA if vp_ANA != 0 else np.nan
alpha_err_pct = 100.0 * (alpha_FEM - alpha_ANA) / alpha_ANA if alpha_ANA != 0 else np.nan

def u_analytic(x, t):
    """
    Analytical harmonic wave at sensor position.
    """
    return -0.01 * (
        np.cos(w * t - np.real(kw0) * x) *
        np.exp(np.imag(kw0) * x)
    )



# --------------------------
# save INFO.csv
# --------------------------
pd.DataFrame([{
    "Dim": Dim,
    "frequency_kHz": frequency,
    "wave_type": wave_tag,
    "x_sensor_mm": x_s,
    "TOF_FEM_ms": TOF,

    "vp_FEM_mm/ms": vp_FEM,
    "vp_FEM_err_mm/ms": vp_FEM_err,
    "vp_ANA_mm/ms": vp_ANA,
    "vp_error_pct (FEM-ANA)": vp_err_pct,

    "alpha_FEM_1/mm": alpha_FEM,
    "alpha_FEM_err_1/mm": alpha_FEM_err,
    "alpha_ANA_1/mm": alpha_ANA,
    "alpha_error_pct (FEM-ANA)": alpha_err_pct,

    "t_act_ms": p_act["t"],
    "u_act_mm": p_act["u"],
    "t_s_ms": p_s["t"],
    "u_s_mm": p_s["u"],
}]).to_csv(out_info, index=False)



print("✔ Saved:", out_info)

# --------------------------
# save plot with markers
# --------------------------
plt.figure(figsize=(13, 6))

# --- curves ---
plt.plot(t_np, u_act, label="Actuator FEM", linewidth=2.0)
plt.plot(t_np, u_s,   label="Sensor FEM",   linewidth=2.0)

# --- labels & title ---
plt.xlabel("Time [ms]", fontsize=16)
plt.ylabel("Displacement [mm]", fontsize=16)
plt.title(
    fr"{Dim}D model {wave_tag} wave – {frequency} kHz, {model_str}  " 
    r"Global temporal plot",
    fontsize=16
)




plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.3)

# --- selected peaks ---
plt.scatter(
    p_act["t"], p_act["u"],
    s=140, color="tab:blue", edgecolor="black",
    zorder=6, label="Actuator peak"
)

plt.scatter(
    p_s["t"], p_s["u"],
    s=140, color="tab:orange", edgecolor="black",
    zorder=6, label="Sensor peak"
)


# --- vertical lines at peaks ---
plt.axvline(
    p_act["t"],
    color="tab:blue",
    linestyle=":",
    linewidth=2,
    alpha=0.8,
    label="Actuator start"
)

plt.axvline(
    p_s["t"],
    color="tab:orange",
    linestyle=":",
    linewidth=2,
    alpha=0.8,
    label="Sensor arrival"
)


# --- TOF distance with arrows (horizontal) ---
y_tof = 0.95 * max(np.max(u_act), np.max(u_s))

plt.annotate(
    "",
    xy=(p_s["t"], y_tof),
    xytext=(p_act["t"], y_tof),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black")
)

plt.text(
    0.5 * (p_act["t"] + p_s["t"]),
    y_tof,
    fr"TOF = {TOF:.3e} ms",
    ha="center",
    va="bottom",
    fontsize=12
)

# --- TOF distance with arrows (horizontal) ---
y_tof = 0.95 * max(np.max(u_act), np.max(u_s))

plt.annotate(
    "",
    xy=(p_s["t"], y_tof),
    xytext=(p_act["t"], y_tof),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black",
    mutation_scale=20)
)

plt.text(
    0.5 * (p_act["t"] + p_s["t"]),
    y_tof,
    fr"TOF = {TOF:.3e} ms",
    ha="center",
    va="bottom",
    fontsize=12
)


# --- horizontal displacement levels at peaks ---
plt.axhline(
    p_act["u"],
    color="tab:blue",
    linestyle="--",
    linewidth=1.5,
    alpha=0.6
)

plt.axhline(
    p_s["u"],
    color="tab:orange",
    linestyle="--",
    linewidth=1.5,
    alpha=0.6
)


# --- displacement difference with arrows (vertical) ---
x_du = p_s["t"] + 0.03 * (t_np[-1] - t_np[0])

plt.annotate(
    "",
    xy=(x_du, p_s["u"]),
    xytext=(x_du, p_act["u"]),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black",
    mutation_scale=20)
)

plt.text(
    x_du,
    0.5 * (p_act["u"] + p_s["u"]),
    fr"$A = {100.0 * (p_s['u'] - p_act['u']) / p_act['u']:+.2f}\%$",
    ha="left",
    va="center",
    fontsize=12
)




# INFO BOX (right side)

vp_err_pct = 100.0 * (vp_FEM - vp_ANA) / vp_ANA if vp_ANA != 0 else np.nan
alpha_err_pct = 100.0 * (alpha_FEM - alpha_ANA) / alpha_ANA if alpha_ANA != 0 else np.nan

textstr = (
    "Phase velocity:\n"
    f"  FEM = {vp_FEM:.3e} mm/ms\n"
    f"  ANA = {vp_ANA:.3e} mm/ms\n"
    f"  Δ%  = {vp_err_pct:+.2f} %\n\n"
    "Attenuation:\n"
    f"  FEM = {alpha_FEM:.3e} 1/mm\n"
    f"  ANA = {alpha_ANA:.3e} 1/mm\n"
    f"  Δ%  = {alpha_err_pct:+.2f} %"
)

plt.text(
    0.98, 0.02,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.9
    )
)


plt.legend(
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    fontsize=12,
    frameon=True
)


plt.tight_layout()
plt.savefig(out_png, dpi=180)
plt.close()

print("✔ FEM plot with peaks saved.")


# ==========================================================
# SENSOR-ONLY PLOT (FEM + ANALYTICAL, NO ACTUATOR)
# ==========================================================

plt.figure(figsize=(13, 6))

# FEM sensor
plt.plot(
    t_np,
    u_s,
    lw=2.8,
    color="tab:orange",
    label="Sensor FEM"
)

# Analytical sensor
u_an = u_analytic(x_s, t_np)
plt.plot(
    t_np,
    u_an,
    lw=2.2,
    ls="--",
    color="black",
    alpha=0.8,
    label="Sensor analytical"
)

# Selected sensor peak
plt.scatter(
    p_s["t"],
    p_s["u"],
    s=160,
    color="tab:orange",
    edgecolor="black",
    zorder=10,
    label="Selected peak"
)

plt.xlabel("Time [ms]", fontsize=16)
plt.ylabel("Displacement [mm]", fontsize=16)

plt.title(
    fr"{Dim}D {wave_tag} wave – sensor only"
    "\n"
    fr"$f={frequency}$ kHz, {model_str}",
    fontsize=16
)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(alpha=0.3)

plt.legend(
    loc="lower right",
    fontsize=12,
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()

out_png_sensor = base + "_SENSOR_ONLY_ANA.png"
plt.savefig(out_png_sensor, dpi=180)
plt.close()

print("✔ Sensor-only FEM vs analytical plot saved:", out_png_sensor)


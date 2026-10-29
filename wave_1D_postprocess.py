"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

"""
1D wave post–processing (MULTI–SENSOR)
• robust peak detection (Gaussian statistics)
• TOF, phase velocity, attenuation
• FEM error estimation
• analytical comparison
• ONE INFO_TABLE row per sensor
"""

# ==========================================================
# IMPORTS
# ==========================================================
import numpy as np
import pandas as pd
import argparse
import os
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



# ==========================================================
# ARGUMENTS
# ==========================================================
parser = argparse.ArgumentParser(
    description="Postprocess 1D wave FEM simulation (multi-sensor)"
)
parser.add_argument("csv", type=str,
                    help="*_TIME_HISTORY.csv")
parser.add_argument("--frac", type=float, default=0.3,
                    help="Peak threshold fraction (default=0.3)")
parser.add_argument(
    "--auto_peak",
    action="store_true",
    default = True,
    help="Automatically select suggested peak without manual input"
)


args = parser.parse_args()



# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv(args.csv)

t = df["t_ms"].to_numpy()
u_act = df["u_act_mm"].to_numpy()

# detect sensor columns automatically
sensor_cols = [
    c for c in df.columns
    if re.match(r"u_s_\d+mm", c)
]

if len(sensor_cols) == 0:
    raise RuntimeError("No sensor columns found (u_s_*mm)")

# global metadata
frequency = float(df["frequency_kHz"].iloc[0])
eta = float(df["eta"].iloc[0])
xi  = float(df["xi"].iloc[0])
nx  = int(df["nx"].iloc[0])
nt  = int(df["nt_factor"].iloc[0])

x_act = 0.0
xlength = 200.0


# ==========================================================
# MATERIAL CONSTANTS
# ==========================================================
rho = 1.780e-3
nu  = 0.0
E   = 100e3
G   = E / (2.0 * (1.0 + nu))
wave_modulus = E / ((1 + nu) * (1 - 2 * nu))
lam = wave_modulus * nu
k1  = lam + 2 * G

wave_speed_elastic = (wave_modulus * (1.0 - nu) / rho) ** 0.5




# ==========================================================
# PLOT STYLE CONSTANTS
# ==========================================================
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
TITLE_SIZE = 16
LEGEND_SIZE = 12


# ==========================================================
# ROBUST PEAK PICKING
# ==========================================================

def manual_peak_selection(t, u, frac, label="signal"):
    """
    Positive peaks only.

    Suggested peak = maximum likelihood under Gaussian model
    (Gaussian-weighted center, robust).

    If args.auto_peak is True, the suggested peak is automatically selected.
    Otherwise, the user can manually choose the peak index from the console.
    """

    # --- positive peaks only ---
    peaks, _ = find_peaks(u)
    if len(peaks) == 0:
        raise RuntimeError(f"No positive peaks found for {label}")

    umax = np.max(u)
    peaks = peaks[u[peaks] >= frac * umax]
    if len(peaks) == 0:
        raise RuntimeError(f"No positive peaks above threshold for {label}")

    amps = u[peaks]

    # --- robust initial estimate (median + MAD) ---
    mu0 = np.median(amps)
    mad = np.median(np.abs(amps - mu0))
    sigma0 = 1.4826 * mad if mad > 0 else np.std(amps, ddof=1)

    if sigma0 <= 0:
        sigma0 = 1e-16

    # --- Gaussian-weighted fixed-point iteration ---
    mu = mu0
    for _ in range(3):
        w = np.exp(-0.5 * ((amps - mu) / sigma0) ** 2)
        mu = np.sum(w * amps) / np.sum(w)

    # --- weighted sigma ---
    sigma = np.sqrt(np.sum(w * (amps - mu) ** 2) / np.sum(w))
    rel_amp_err = sigma / mu if mu > 0 else 0.0

    # --- maximum likelihood peak ---
    idx_suggested = np.argmin(np.abs(amps - mu))
    p_suggested = peaks[idx_suggested]

    # ---------------- plot ----------------
    plt.figure(figsize=(12, 4))
    plt.plot(t, u, lw=2, label=label)

    plt.scatter(
        t[peaks], u[peaks],
        s=80, color="tab:orange",
        label="candidate peaks"
    )

    plt.scatter(
        t[p_suggested], u[p_suggested],
        s=140, color="red", edgecolor="black",
        zorder=10, label="suggested (ML)"
    )

    plt.title(
        f"{label} – positive peak selection\n"
        f"μ_ML={mu:.3e}, σ={sigma:.3e}, rel_err={rel_amp_err:.2%}"
    )
    plt.xlabel("Time [ms]")
    plt.ylabel("Displacement [mm]")
    plt.grid(alpha=0.3)
    plt.legend(
        loc="lower right",
        fontsize=LEGEND_SIZE,
        frameon=True,
        framealpha=0.9
    )
    plt.tight_layout()
    plt.show()

    # ---------------- console ----------------
    print(f"\nCandidate positive peaks for {label}:")
    for i, p in enumerate(peaks):
        print(
            f"[{i:2d}] t={t[p]:.6e} ms | u={u[p]:+.6e} mm"
            + ("  <-- suggested" if p == p_suggested else "")
        )

    # --- automatic vs manual selection ---
        # --- automatic vs manual selection ---
    if args.auto_peak:
        p_final = p_suggested
        peak_index = idx_suggested
    else:
        choice = input("\nChoose peak index (ENTER = suggested): ").strip()
        if choice == "":
            p_final = p_suggested
            peak_index = idx_suggested
        else:
            peak_index = int(choice)
            p_final = peaks[peak_index]

    return {
        "t": t[p_final],
        "u": u[p_final],
        "rel_amp_err": rel_amp_err,
        "peak_index": peak_index
    }




# ==========================================================
# FEM ERROR MODELS
# ==========================================================
def velocity_error(TOF, frequency, x_s, nx):
    dx = xlength / nx
    dt = 0.5 / (frequency * nt)
    return dx / TOF + x_s * dt / TOF**2


def attenuation_error(u_act, u_s, x_s, nx, rel_amp_err):
    dx = xlength / nx
    ln_ratio = np.log(abs(u_act / u_s))
    err_space = abs(ln_ratio) * dx / x_s**2
    err_amp = np.sqrt(2.0) * rel_amp_err / x_s
    return np.sqrt(err_space**2 + err_amp**2)


# ==========================================================
# ANALYTICAL MODEL
# ==========================================================
w = 2.0 * np.pi * frequency

def kw_visc(w, rho, k1, c):
    wc = c / rho
    z = np.sqrt(rho / k1)
    re = w * z * (1 + (wc / w)**2)**0.25 * np.cos(-0.5*np.arctan(wc/w))
    im = w * z * (1 + (wc / w)**2)**0.25 * np.sin(-0.5*np.arctan(wc/w))
    return re + 1j*im

def kw_grad(w, rho, k1, c1):
    wc1 = k1 / c1
    z = np.sqrt(rho / k1)
    re = w * z * (1/(1+(w/wc1)**2))**0.25 * np.cos(-0.5*np.arctan(w/wc1))
    im = w * z * (1 + (w / wc1)**2)**0.25 * np.sin(-0.5*np.arctan(w/wc1))
    return re + 1j*im

def kw_general(w, rho, k1, eta, xi):
    if eta == 0 and xi == 0:
        return w / wave_speed_elastic
    elif eta != 0 and xi == 0:
        return kw_visc(w, rho, k1, eta)
    elif xi != 0 and eta == 0:
        return kw_grad(w, rho, k1, xi)
    else:
        raise ValueError("Invalid model")

# --- recompute analytical quantities at central frequency ---
kw0 = kw_general(w, rho, k1, eta, xi)

rekw_general = np.real(kw0)
imkw_general = np.imag(kw0)

# analytical phase velocity [mm/ms]
vp_ANA = w / rekw_general

# analytical attenuation [1/mm]
alpha_ANA = -imkw_general

def u_analytic(x, t):
    """
    Analytical harmonic wave (same frequency as FEM),
    no actuator plotted, only propagation + attenuation.
    """
    return -0.01 * (
        np.cos(w * t - rekw_general * x) *
        np.exp(imkw_general * x)
    )



# ==========================================================
# LOOP OVER SENSORS (SENSOR FIRST, THEN ACTUATOR)
# ==========================================================
rows = {}
sensor_peaks = {}
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
sensor_colors = {}

for i, col in enumerate(sensor_cols):
    x_s = float(re.findall(r"\d+", col)[0])
    sensor_colors[x_s] = color_cycle[i % len(color_cycle)]




for col in sensor_cols:

    x_s = float(re.findall(r"\d+", col)[0])
    u_s = df[col].to_numpy()

    # --- 1) SENSOR PEAK ---
    p_s = manual_peak_selection(
        t, u_s,
        frac=args.frac,
        label=f"Sensor @ {x_s:.0f} mm"
    )

    # --- 2) ACTUATOR PEAK (for this sensor) ---
    # --- actuator peaks (same detection, no selection) ---
    peaks_act, _ = find_peaks(u_act)
    peaks_act = peaks_act[u_act[peaks_act] >= args.frac * np.max(u_act)]

    idx = p_s["peak_index"]
    if idx >= len(peaks_act):
        raise RuntimeError(
            f"Actuator does not have peak #{idx} corresponding to sensor peak"
        )

    p_act = {
        "t": t[peaks_act[idx]],
        "u": u_act[peaks_act[idx]],
        "rel_amp_err": p_s["rel_amp_err"]  # stesso fronte → stessa incertezza
    }


    # store for plotting
    sensor_peaks[x_s] = {
        "sensor": p_s,
        "actuator": p_act
    }

    # --- 3) FEM quantities ---
    TOF = p_s["t"] - p_act["t"]
    vp_FEM = (x_s - x_act) / TOF
    alpha_FEM = (1.0 / (x_s - x_act)) * np.log(
        abs(p_act["u"]) / abs(p_s["u"])
    )

    vp_err = velocity_error(TOF, frequency, x_s, nx)
    alpha_err = attenuation_error(
        p_act["u"], p_s["u"], x_s, nx, p_s["rel_amp_err"]
    )

    rows[x_s] = {
        "frequency_kHz": frequency,
        "eta": eta,
        "xi": xi,
        "nx": nx,
        "nt_factor": nt,
        "sensor_x_mm": x_s,

        "t_act_ms": p_act["t"],
        "u_act_mm": p_act["u"],
        "t_s_ms": p_s["t"],
        "u_s_mm": p_s["u"],

        "TOF_FEM_ms": TOF,

        "vp_FEM_mmms": vp_FEM,
        "vp_FEM_err_mmms": vp_err,
        "vp_ANA_mmms": vp_ANA,

        "alpha_FEM_1mm": alpha_FEM,
        "alpha_FEM_err_1mm": alpha_err,
        "alpha_ANA_1mm": alpha_ANA
    }


# ==========================================================
# SAVE INFO_TABLE
# ==========================================================
INFO = pd.DataFrame(rows.values())


outfile = args.csv.replace("_TIME_HISTORY.csv", "_INFO_TABLE.csv")
INFO.to_csv(outfile, index=False)

print("\n✔ INFO_TABLE written:")
print(f"  → {outfile}")


# ==========================================================
# GLOBAL PLOT (FEM ONLY)
# ==========================================================


plt.figure(figsize=(13, 6))

# --- curves ---
plt.plot(
    t,
    u_act,
    lw=3.0,
    color="red",
    label="Actuator FEM"
)

for col in sensor_cols:
    x_s = float(re.findall(r"\d+", col)[0])
    c = sensor_colors[x_s]

    plt.plot(
        t,
        df[col],
        lw=2.2,
        color=c,
        label=f"Sensor @ {x_s:.0f} mm"
    )

# --- sensor & actuator peaks ---
for x_s, pair in sensor_peaks.items():
    c = sensor_colors[x_s]

    plt.scatter(
        pair["sensor"]["t"],
        pair["sensor"]["u"],
        s=160,
        color=c,
        edgecolor="black",
        zorder=10,
        label=f"Sensor peak @ {int(x_s)} mm"
    )

    plt.scatter(
        pair["actuator"]["t"],
        pair["actuator"]["u"],
        s=160,
        marker="x",
        color=c,
        linewidths=3,
        zorder=10,
        label=f"Actuator peak @ {int(x_s)} mm"
    )

# --- labels & style ---
plt.xlabel("Time [ms]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel("Displacement [mm]", fontsize=AXIS_LABEL_SIZE)

plt.title(
    fr"1D FEM – {frequency} kHz, $c={eta}$, $c_1={xi}$"
    "\nGlobal displacement over time",
    fontsize=TITLE_SIZE
)

plt.xticks(fontsize=TICK_LABEL_SIZE)
plt.yticks(fontsize=TICK_LABEL_SIZE)

plt.grid(alpha=0.3)
plt.legend(
    loc="lower right",
    fontsize=LEGEND_SIZE,
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()

png = outfile.replace(".csv", ".png")
plt.savefig(png, dpi=160)
plt.close()


# ==========================================
# SENSOR-ONLY PLOT (FEM + SENSOR PEAKS)  
# ==========================================

plt.figure(figsize=(13, 6))

# --- plot sensor signals ---
for col in sensor_cols:
    x_s = float(re.findall(r"\d+", col)[0])
    u_s = df[col].to_numpy()
    c = sensor_colors[x_s]

    plt.plot(t, u_s, lw=2.5, color=c, label=f"Sensor @ {x_s:.0f} mm")

    # selected peak marker
    if x_s in sensor_peaks:
        p = sensor_peaks[x_s]["sensor"]
        plt.scatter(p["t"], p["u"], s=160, color=c, edgecolor="black",
                    zorder=10, label=f"Peak @ {x_s:.0f} mm")



plt.xlabel("Time [ms]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel("Displacement [mm]", fontsize=AXIS_LABEL_SIZE)


plt.title(
    fr"1D FEM – Sensors only – {frequency} kHz, $c={eta}$, $c_1={xi}$"
    "\nGlobal displacement over time",
    fontsize=TITLE_SIZE
)

plt.xticks(fontsize=TICK_LABEL_SIZE)
plt.yticks(fontsize=TICK_LABEL_SIZE)
plt.grid(alpha=0.3)

handles, labels = plt.gca().get_legend_handles_labels()
seen = set()
h2, l2 = [], []
for h, lab in zip(handles, labels):
    if lab not in seen:
        h2.append(h); l2.append(lab); seen.add(lab)
plt.legend(h2, l2, loc="lower right", fontsize=LEGEND_SIZE, frameon=True, framealpha=0.9)

plt.tight_layout()
png_sensors = outfile.replace(".csv", "_SENSORS_ONLY.png")
plt.savefig(png_sensors, dpi=160)
plt.close()


# ==========================================================
# SENSOR-ONLY PLOT (FEM + ANALYTICAL + SENSOR PEAKS)
# ==========================================================
plt.figure(figsize=(11, 6))

for col in sensor_cols:
    x_s = float(re.findall(r"\d+", col)[0])
    u_s = df[col].to_numpy()
    c = sensor_colors[x_s]

    # --- FEM sensor ---
    plt.plot(
        t,
        u_s,
        lw=2.5,
        color=c,
        label=f"Sensor FEM @ {x_s:.0f} mm"
    )

    # --- ANALYTICAL sensor (behind, dashed) ---
    u_an = u_analytic(x_s, t)
    plt.plot(
        t,
        u_an,
        lw=2.0,
        ls="--",
        color=c,
        alpha=0.6,
        label=f"Sensor ANA @ {x_s:.0f} mm"
    )

    # --- peak ---
    if x_s in sensor_peaks:
        p = sensor_peaks[x_s]["sensor"]
        plt.scatter(
            p["t"],
            p["u"],
            s=160,
            color=c,
            edgecolor="black",
            zorder=10
        )

plt.xlabel("Time [ms]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel("Displacement [mm]", fontsize=AXIS_LABEL_SIZE)

plt.title(
    fr"1D FEM vs Analytical – Sensors only – {frequency} kHz"
    fr"$N_{{el}}={nx},\; N_{{ti}}={nt}$",
    fontsize=TITLE_SIZE
)



plt.xticks(fontsize=TICK_LABEL_SIZE)
plt.yticks(fontsize=TICK_LABEL_SIZE)

plt.grid(alpha=0.3)
plt.legend(
    loc="lower right",
    fontsize=LEGEND_SIZE,
    frameon=True,
    framealpha=0.9
)

plt.tight_layout()
plt.savefig(outfile.replace(".csv", "_SENSORS_ONLY_ANA.png"), dpi=160)
plt.close()

# ==========================================================
# SINGLE-SENSOR PLOTS (FEM vs ANALYTICAL)
# ==========================================================

for col in sensor_cols:

    x_s = float(re.findall(r"\d+", col)[0])
    u_s = df[col].to_numpy()
    c = sensor_colors[x_s]

    # analytical signal
    u_an = u_analytic(x_s, t)

    plt.figure(figsize=(12, 5))

    # FEM signal
    plt.plot(
        t, u_s,
        lw=2.8,
        color=c,
        label="FEM"
    )

    # Analytical signal
    plt.plot(
        t, u_an,
        lw=2.2,
        ls="--",
        color="black",
        alpha=0.7,
        label="Analytical"
    )

    # Peak (sensor)
    #if x_s in sensor_peaks:
        #p = sensor_peaks[x_s]["sensor"]
        #plt.scatter(
            #p["t"],
            #p["u"],
            #s=180,
            #color=c,
            #edgecolor="black",
            #zorder=10,
            #label="Selected peak"
        #)

    # Labels & style
    plt.xlabel("Time [ms]", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("Displacement [mm]", fontsize=AXIS_LABEL_SIZE)

    plt.title(
        fr"1D FEM vs Analytical – Sensor at $x={int(x_s)}$ mm"
        "\n"
        fr"$f={frequency}$ kHz, $c={eta}$, $c_1={xi}$"
        "\n"
        fr"$N_{{el}}={nx},\; N_{{ti}}={nt}$",
        fontsize=TITLE_SIZE
    )

    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)

    plt.grid(alpha=0.3)
    plt.legend(
        fontsize=LEGEND_SIZE,
        frameon=True,
        framealpha=0.9
    )

    plt.tight_layout()

    out_png = outfile.replace(
        ".csv", f"_SENSOR_{int(x_s)}mm_FEM_vs_ANA.png"
    )
    plt.savefig(out_png, dpi=160)
    plt.close()

    print(f"✔ Saved single-sensor comparison plot: {out_png}")


# ==========================================================
# PEAK AMPLITUDE vs DISTANCE (DECAY LAW)
# ==========================================================

# === peak data ===
xs_sorted = np.array(sorted(sensor_peaks.keys()), dtype=float)          # [50,100,150]
A_pts = np.array([abs(sensor_peaks[x]["sensor"]["u"]) for x in xs_sorted])  # |u_peak|

# ==========================================================
# OPTION 1: use of the alpha analytic or FEM
# ==========================================================
alpha_use = alpha_ANA  

x_ref = xs_sorted[0]
A_ref = A_pts[0]
A0_use = A_ref * np.exp(alpha_use * x_ref)

x_curve = np.linspace(xs_sorted.min(), xs_sorted.max(), 200)
A_curve = A0_use * np.exp(-alpha_use * x_curve)

# ==========================================================
#  OPTION 1: use of the alpha fit
# ==========================================================

eps = 1e-16
coeff = np.polyfit(xs_sorted, np.log(A_pts + eps), 1)  # lnA = m x + q
m, q = coeff
alpha_fit = -m
A0_fit = np.exp(q)

A_curve_fit = A0_fit * np.exp(-alpha_fit * x_curve)

plt.figure(figsize=(7, 5))

# punti FEM (picchi selezionati)
plt.scatter(
    xs_sorted, A_pts,
    s=90,
    facecolor="white",
    edgecolor="magenta",
    zorder=10,
    label=r"FEM peaks $|u_{\mathrm{peak}}|$"
)

# curva analitica
plt.plot(
    x_curve, A_curve,
    color="black",
    lw=2.5,
    ls="--",
    label=(
        rf"Analytical: $A_0 e^{{-\alpha x}}$"
        rf"$,\ \alpha={alpha_ANA:.3e}\ \mathrm{{mm^{{-1}}}}$"
    )
)

# curva fit FEM
plt.plot(
    x_curve, A_curve_fit,
    color="magenta",
    lw=2.5,
    label=(
        rf"FEM fit: $A_0 e^{{-\alpha x}}$"
        rf"$,\ \alpha={alpha_fit:.3e}\ \mathrm{{mm^{{-1}}}}$"
    )
)

plt.xlabel(r"Distance $x$ [mm]", fontsize=AXIS_LABEL_SIZE)
plt.ylabel(r"Peak amplitude $|u_{\mathrm{peak}}|$ [mm]", fontsize=AXIS_LABEL_SIZE)

plt.title(
    fr"Spatial attenuation – {frequency} kHz - $c={eta}$, $c_1={xi}$",
    fontsize=TITLE_SIZE
)

plt.grid(alpha=0.3)
plt.legend(fontsize=LEGEND_SIZE, frameon=True, framealpha=0.95)

plt.tight_layout()

out_png = outfile.replace(".csv", "_PEAK_DECAY_X.png")
plt.savefig(out_png, dpi=200)
plt.close()

print(f"✔ Saved peak decay plot: {out_png}")

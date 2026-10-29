"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import warnings
warnings.simplefilter("ignore")
from matplotlib.patches import Patch
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import gaussian_filter

# =========================================================
# SETTINGS
# =========================================================
THRESHOLD = 1.0
ZMAX = 50.0

FREQ_COLORS = {
    120.0: "blue",
    150.0: "green",
    200.0: "orange",
    250.0: "red",
    375.0: "purple",
}

NEL_VALUES = np.array([100, 200, 300, 400, 500, 600, 700, 800])
NTI_VALUES = np.array([
    100, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000,
    3200, 3500, 4000, 4500, 5000, 6000, 6400, 6500,
    7000, 7500, 8000, 8500, 9000, 10000, 11000, 12000, 12800
])

BASE_DIR = "frequency_tables_eta1.0_xi0.0"


SCATTER_POINTS = {
    120.0: (450, 3000),
    150.0: (500, 4200),
    200.0: (550, 5200),
    250.0: (630, 6800),
    375.0: (760, 9700),
}


# =========================================================
# LOAD VALID POINTS (robust parsing)
# =========================================================
valid_points = []

current_freq = None


# =========================================================
# LOAD DATA
# =========================================================
data_by_freq = {}


for nel in NEL_VALUES:
    for nti in NTI_VALUES:
        fname = f"{BASE_DIR}/FREQUENCY_TABLE_nx{nel}_nt{nti}_x100mm.csv"
        if not os.path.exists(fname):
            continue

        df = pd.read_csv(fname)

        for _, r in df.iterrows():
            f = float(r["f_kHz"])
            vp_err = float(r["vp_err_%"])
            A_err  = float(r["A_err_%"])

            rmse = np.sqrt((vp_err**2 + A_err**2) / 2.0)

            data_by_freq.setdefault(f, {"Nel": [], "Nti": [], "RMSE": []})
            data_by_freq[f]["Nel"].append(nel)
            data_by_freq[f]["Nti"].append(nti)
            data_by_freq[f]["RMSE"].append(rmse)


EXCLUDED_FREQS = {80.0, 750.0, 1000.0}

data_by_freq = {
    f: vals for f, vals in data_by_freq.items()
    if f not in EXCLUDED_FREQS
}

# =========================================================
# GRID FOR INTERPOLATION
# =========================================================
xi = np.linspace(NEL_VALUES.min(), NEL_VALUES.max(), 300)
yi = np.linspace(NTI_VALUES.min(), NTI_VALUES.max(), 300)
Xi, Yi = np.meshgrid(xi, yi)

SAFE_MASK = np.ones_like(Xi, dtype=bool)
DOMAIN_MASK = np.zeros_like(Xi, dtype=bool)

# =========================================================
# MAIN FIGURE (3D)
# =========================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

legend_handles_3d = [
    Patch(facecolor=FREQ_COLORS[f], edgecolor="k", label=f"{f:.0f} kHz")
    for f in data_by_freq.keys()
]

ax.legend(
    handles=legend_handles_3d,
    title="Frequency",
    loc="upper left",
    frameon=True
)

Z_thr = np.full_like(Xi, THRESHOLD)
Z_thr[~DOMAIN_MASK] = np.nan
THR_MASK_ALL = np.zeros_like(Xi, dtype=bool)



txt = open("Threshold_fits.txt", "w")
txt.write("RMSE = 1% threshold laws (hyperbola fit)\n\n")


Zi_1000_mask = None   # ← UNA variabile in più

for f, vals in data_by_freq.items():

    X = np.array(vals["Nel"], dtype=float)
    Y = np.array(vals["Nti"], dtype=float)
    Z = np.array(vals["RMSE"], dtype=float)

    if len(Z) < 6:
        continue

    Zi = griddata((X, Y), Z, (Xi, Yi), method="linear")
    if Zi is None or np.all(np.isnan(Zi)):
        continue

    # ---------- superficie RMSE sopra threshold ----------
    Zi_cut = Zi.copy()
    Zi_cut[Zi < THRESHOLD] = np.nan

    ax.plot_surface(
        Xi, Yi, Zi_cut,
        color=FREQ_COLORS.get(f, "gray"),
        alpha=0.45,
        edgecolor="none",
        shade=False
    )

    # ---------- curva di contorno ----------
    cs_3d = plt.contour(Xi, Yi, Zi, levels=[THRESHOLD])
    curves_3d = cs_3d.allsegs[0]
    plt.close()

    for seg in curves_3d:
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            np.full_like(seg[:, 0], THRESHOLD),
            color=FREQ_COLORS.get(f, "black"),
            linestyle="dotted",
            linewidth=2
        )

    # ---------- salva MASCHERA della 1000 kHz ----------
    if f == 375.0:
        Zi_1000_mask = Zi < THRESHOLD

    # ---------- piano threshold (vincolato alla 1000 kHz) ----------
    if Zi_1000_mask is not None:
        Z_thr_cut = np.full_like(Zi, THRESHOLD)
        Z_thr_cut[~Zi_1000_mask] = np.nan

        ax.plot_surface(
            Xi, Yi, Z_thr_cut,
            color="lightgray",
            alpha=0.9,
            edgecolor="none",
            shade=False
        )


# =========================================================
# SAFE MASK FINAL
# =========================================================
SAFE_MASK = binary_fill_holes(SAFE_MASK) & DOMAIN_MASK

ax.set_xlabel("Nel")
ax.set_ylabel("Nti")
ax.set_zlabel("RMSE error [%]")

ax.set_xlim(100, 1000)
ax.set_ylim(0, 12000)
ax.set_zlim(0, 60)

ax.view_init(elev=20, azim=25)
ax.set_title("RMSE = 1% threshold surfaces", pad=20)

fig.savefig("RMSE_ANGLE_view.jpg", dpi=300)
plt.close(fig)

HYPERBOLA_PARAMS = {}
ASYMPTOTES = {
    120.0:  (370.0, 2100.0),
    150.0:  (390.0, 3000.0),
    200.0:  (420.0, 4300.0),
    250.0:  (480.0, 5200.0),
    375.0: (650.0, 5800.0),
}


# =========================================================
# 2D PLOT (CURVES + IPERBOLI)
# =========================================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

HYPERBOLA_PARAMS = {}

# ---- curve di contorno RMSE = 1% ----
for f, vals in data_by_freq.items():

    Zi = griddata(
        (vals["Nel"], vals["Nti"]),
        vals["RMSE"],
        (Xi, Yi),
        method="linear"
    )

    if Zi is None or np.all(np.isnan(Zi)):
        continue

    ax2.contour(
        Xi, Yi, Zi,
        levels=[THRESHOLD],
        colors=FREQ_COLORS.get(f, "black"),
        linestyles="dotted",
        linewidths=2
    )



    # ---- iperbole solo se ho asintoti ----
    if f not in ASYMPTOTES:
        continue

    Nel0, Nti0 = ASYMPTOTES[f]

    # ---- invece di plt.contour direttamente su ax2 ----
    fig_tmp, ax_tmp = plt.subplots()
    cs_tmp = ax_tmp.contour(Xi, Yi, Zi, levels=[THRESHOLD])
    pts = np.vstack(cs_tmp.allsegs[0])
    plt.close(fig_tmp)


    A_vals = (pts[:,1] - Nti0) * (pts[:,0] - Nel0)
    A = np.nanmedian(A_vals[A_vals > 0])

    HYPERBOLA_PARAMS[f] = (Nel0, Nti0, A, 1.0)

    txt.write(
        f"{int(f)} kHz : "
        f"Nti = {Nti0:.1f} + {A:.3e} / (Nel - {Nel0:.1f})\n"
    )


# ---- plot iperboli ----
Nel_line = np.linspace(Xi.min(), Xi.max(), 1000)

for f, (Nel0, Nti0, A, _) in HYPERBOLA_PARAMS.items():

    Nel_valid = Nel_line[Nel_line > Nel0 + 1e-6]
    Nti_fit = Nti0 + A / (Nel_valid - Nel0)

    mask = (Nti_fit > 0) & (Nti_fit < 12000)

    ax2.plot(
        Nel_valid[mask],
        Nti_fit[mask],
        color=FREQ_COLORS.get(f, "black"),
        linewidth=2.5
    )



# =========================================================
# SCATTER POINTS (reference / design points)
# =========================================================
for f, (nel_s, nti_s) in SCATTER_POINTS.items():
    if f not in FREQ_COLORS:
        continue

    ax2.scatter(
        nel_s,
        nti_s,
        marker="o",      # oppure "^", "s", "D"
        s=110,
        color=FREQ_COLORS[f],
        edgecolor="black",
        linewidth=1.2,
        zorder=15
    )



ax2.set_xlabel("Nel")
ax2.set_ylabel("Nti")
ax2.set_title("RMSE error = 1% intersection curves + hyperbolic fit")




from matplotlib.lines import Line2D

# handle fittizio per i punti di simulazione
simulation_points_handle = Line2D(
    [], [], 
    marker='o',
    linestyle='None',
    markersize=9,
    markerfacecolor='lightgray',
    markeredgecolor='black',
    label='Points selected for 2D Simulation'
)

legend_handles_2d = [
    Patch(facecolor=FREQ_COLORS[f], edgecolor="k", label=f"{f:.0f} kHz")
    for f in HYPERBOLA_PARAMS.keys()
]

legend_handles_2d.append(simulation_points_handle)

ax2.legend(
    handles=legend_handles_2d,
    title="Legenda",
    loc="upper left",
    frameon=True
)

fig2.savefig("RMSE_THRESHOLD_2D.png", dpi=300)
plt.close(fig2)

txt.close()

# =========================================================
# NEW PLOT: Nel / lambda vs Nti (threshold curves)
# (BASE IPERBOLICO, COERENTE CON IL CODICE ORIGINALE)
# =========================================================
fig3, ax3 = plt.subplots(figsize=(8, 6))

# --- parametri fisici ---
rho = 2.3e-3
nu  = 0.0
E   = 30e3
eta = 1.0
xi  = 0.0

G   = E / (2.0 * (1.0 + nu))
wave_modulus = E / ((1 + nu) * (1 - 2 * nu))
lam = wave_modulus * nu
k1  = lam + 2 * G

# --- funzioni di dispersione ---
def kw_visc(w, rho, k1, c):
    wc = c / rho
    z = np.sqrt(rho / k1)
    re = w * z * (1 + (wc / w)**2)**0.25 * np.cos(-0.5*np.arctan(wc/w))
    im = w * z * (1 + (wc / w)**2)**0.25 * np.sin(-0.5*np.arctan(wc/w))
    return re + 1j*im

def kw_general(w, rho, k1, eta, xi_visc):
    if eta == 0 and xi_visc == 0:
        wave_speed_elastic = np.sqrt((wave_modulus * (1.0 - nu) / rho))
        return w / wave_speed_elastic
    elif eta != 0 and xi_visc == 0:
        return kw_visc(w, rho, k1, eta)
    else:
        raise ValueError("Invalid model")


# --- plot delle curve normalizzate ---
for f, (Nel0, Nti0, A, p) in HYPERBOLA_PARAMS.items():

    # lunghezza d'onda analitica
    w = 2.0 * np.pi * f
    kw0 = kw_general(w, rho, k1, eta, xi)
    vp = w / np.real(kw0)
    lambda_f = vp / f

    Nel_valid = Nel_line[Nel_line > Nel0 + 1e-6]

    Nti_fit = Nti0 + A / (Nel_valid - Nel0)
    mask = (Nti_fit > 0) & (Nti_fit < 13000)

    if np.any(mask):
        ax3.plot(
            Nel_valid[mask] / lambda_f,
            Nti_fit[mask]*f,
            color=FREQ_COLORS.get(f, "black"),
            lw=2.5,
            label=f"{f:.0f} kHz"
        )


ax3.set_xlabel(r"$Nel / \lambda$")
ax3.set_ylabel("Nti * frequency")
ax3.set_title("RMSE = 1% threshold — normalized spatial resolution")

ax3.grid(alpha=0.3)
ax3.legend(title="Frequency")

fig3.tight_layout()
fig3.savefig("RMSE_THRESHOLD_Nel_over_lambda.png", dpi=300)
plt.close(fig3)

import numpy as np
import pandas as pd


# =========================================================
# Nel grid (puoi cambiarla se vuoi)
# =========================================================
Nel_values = np.arange(100, 801, 10)

# =========================================================
# BUILD CSV TABLE: Nti vs Nel and Frequency (from FITS)
# =========================================================

NEL_TABLE = np.array([100, 200, 300, 400, 500, 600, 700, 800])

rows = []

for nel in NEL_TABLE:
    row = {"Nel": nel}

    for f in sorted(HYPERBOLA_PARAMS):
        Nel0, Nti0, A, p = HYPERBOLA_PARAMS[f]

        # cast di sicurezza
        Nel0 = float(Nel0)
        Nti0 = float(Nti0)
        A    = float(A)
        

        if nel <= Nel0:
            nti_val = np.nan
        else:
            nti_val = Nti0 + A * (nel - Nel0) 

        row[f"{int(f)}_kHz"] = nti_val

    rows.append(row)

df_table = pd.DataFrame(rows)

csv_name = "RMSE_fit_Nti_vs_Nel_and_Frequency.csv"
df_table.to_csv(csv_name, index=False)

print(f"[OK] CSV written: {csv_name}")

# =========================================================
# BUILD CSV TABLE: Nti vs (Nel / lambda) and Frequency
# =========================================================

rows_lambda = []

for nel in NEL_TABLE:
    row = {}

    for f in sorted(HYPERBOLA_PARAMS):

        # ---- parametri iperbola (TUPLA!) ----
        Nel0, Nti0, A, p = HYPERBOLA_PARAMS[f]

        # ---- lunghezza d'onda ----
        w = 2.0 * np.pi * f
        kw0 = kw_general(w, rho, k1, eta, xi)
        vp = w / np.real(kw0)
        lambda_f = vp / f

        nel_norm = nel / lambda_f

        if nel <= Nel0:
            nti_val = np.nan
        else:
            nti_val = Nti0 + A * (nel - Nel0) ** (-p)

        row["Nel_over_lambda"] = nel_norm
        row[f"{int(f)}_kHz"] = nti_val

    rows_lambda.append(row)

df_lambda = pd.DataFrame(rows_lambda)

csv_name = "RMSE_fit_Nti_vs_Nel_over_lambda_and_Frequency.csv"
df_lambda.to_csv(csv_name, index=False)

print(f"[OK] CSV written: {csv_name}")



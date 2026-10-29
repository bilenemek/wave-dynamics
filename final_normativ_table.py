"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np
import pandas as pd
import os, sys, re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# --------------------------
# constants / geometry
# --------------------------
xlength = 200.0
x_s = 0.5 * xlength   # 100 mm

# --------------------------
# MODE SELECTION
# --------------------------
RUN_SINGLE = len(sys.argv) > 1
AUTO_PEAK = False   # ← default: MANUALE
if "--auto" in sys.argv:
    AUTO_PEAK = True


# --------------------------
# ELASTIC REFERENCE SPEED
# --------------------------
rho = 1.78e-3
nu = 0.30
E_ref = 100e3  # MPa, valore di riferimento

G_ref = E_ref / (2.0 * (1.0 + nu))
lam_ref = (E_ref * nu) / ((1 + nu) * (1 - 2 * nu))
k1_ref = lam_ref + 2 * G_ref

def vp_elastic_from_E(E_mpa, rho=1.78e-3, nu=0.30):
    G = E_mpa/(2*(1+nu))
    lam = (E_mpa*nu)/((1+nu)*(1-2*nu))
    k1 = lam + 2*G
    return np.sqrt(k1/rho)  # mm/ms




# --------------------------
# FREQUENCY THRESHOLD
# --------------------------
FREQ_PHASE_THRESHOLD = 400.0  # kHz


# --------------------------
# ONSET
# --------------------------

def detect_onset_energy(t, u, t_pre_ms=0.002, win=25, k=6.0,
                        t_min_ms=None, t_max_ms=None, hold=10):
    """
    Onset via energia locale: E(t)=moving average(u^2)
    threshold = mean(E_pre) + k*std(E_pre)

    + finestra [t_min_ms, t_max_ms]
    + condizione di permanenza sopra soglia (hold) per evitare falsi trigger
    """
    u0 = u - np.mean(u)

    kernel = np.ones(win) / win
    E = np.convolve(u0**2, kernel, mode="same")

    # baseline
    pre = t <= t_pre_ms
    if np.sum(pre) < max(10, win):
        n0 = max(10, int(0.05 * len(t)))
        E_pre = E[:n0]
    else:
        E_pre = E[pre]

    mu = np.mean(E_pre)
    sig = np.std(E_pre)
    thr = mu + k * sig

    # finestra di ricerca
    i0 = 0 if t_min_ms is None else int(np.searchsorted(t, t_min_ms))
    i1 = len(t) if t_max_ms is None else int(np.searchsorted(t, t_max_ms))

    if i1 <= i0 + hold:
        raise RuntimeError("Onset window too small / invalid")

    # primo indice che supera thr e ci rimane per 'hold' campioni
    idx = None
    for i in range(i0, i1 - hold):
        if np.all(E[i:i+hold] > thr):
            idx = i
            break

    if idx is None:
        # fallback: max energia nella finestra (meglio che triggerare a caso)
        idx = i0 + int(np.argmax(E[i0:i1]))

    return {"t": t[idx], "idx": int(idx), "thr": thr}

def plot_E_c_correlation_all_freq(
    freq_tags,
    quantity="vp",
    c_value=0.8
):
    """
    Plot E–value correlation with one curve per frequency.
    Points + lines.
    
    quantity = "vp" or "alpha"
    c_value  = fixed c (float). If None → uses first available c.
    """

    if quantity == "vp":
        table_prefix = "TABLE_V_"
        ylabel = r"$v_p$ [mm/ms]"
        fname = "Ec_vp_all_freq.png"
    elif quantity == "alpha":
        table_prefix = "TABLE_ALPHA_"
        ylabel = r"$\alpha$ [1/mm]"
        fname = "Ec_alpha_all_freq.png"
    else:
        raise ValueError("quantity must be 'vp' or 'alpha'")

    outdir = "NORMATIVE_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for freq_tag in freq_tags:

        csv_file = f"{table_prefix}{freq_tag}.csv"
        if not os.path.isfile(csv_file):
            print(f"⚠ {csv_file} not found, skipping.")
            continue

        T = pd.read_csv(csv_file, index_col=0)
        T.index = T.index.astype(float)     # E [GPa]
        T.columns = T.columns.astype(float) # c

        # scegli c
        if c_value is None:
            c = T.columns[0]
        else:
            if c_value not in T.columns:
                print(f"⚠ c={c_value} not in {freq_tag}, skipping.")
                continue
            c = c_value

        E_vals = T.index.to_numpy()
        Y_vals = T[c].to_numpy()

        plt.plot(
            E_vals,
            Y_vals,
            marker="o",
            lw=2,
            label=f"{freq_tag}"
        )

    plt.xlabel("Young modulus E [GPa]", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    title = (
        f"{ylabel} vs E\n"
        f"Fixed c = {c}"
    )
    plt.title(title, fontsize=13)

    plt.grid(alpha=0.3)
    plt.legend(title="Frequency")
    plt.tight_layout()

    out = os.path.join(outdir, fname)
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"✔ E–c correlation (all frequencies) saved: {out}")


# ==========================================================
# SINGLE FILE MODE
# ==========================================================
if RUN_SINGLE:

    csv_path = sys.argv[1]
    mE = re.search(r"E_(\d+)", csv_path)
    if mE is None:
        raise RuntimeError("Young modulus E not found in path")
    E_mpa = float(mE.group(1))

    df = pd.read_csv(csv_path)

    t_np  = df["t_ms"].to_numpy()
    u_act = df["u_act_mm"].to_numpy()
    u_s   = df["u_s_mm"].to_numpy()

    frequency = float(df["frequency_kHz"].iloc[0])
    wave_tag  = str(df["wave_type"].iloc[0])
    Dim       = int(df["Dim"].iloc[0])

    fname = os.path.basename(csv_path)
    m_nx = re.search(r"_nx(\d+)", fname)
    m_nt = re.search(r"_nt(\d+)", fname)
    if m_nx is None or m_nt is None:
        raise ValueError("nx or nt not found in filename")

    nx = int(m_nx.group(1))
    nt = int(m_nt.group(1))


    # --------------------------
    # SENSOR peak selection
    # --------------------------
    def select_peak(t, u, frac, basepath, label, auto=False):

        peaks, _ = find_peaks(u)
        peaks = peaks[u[peaks] >= frac * np.max(u)]
        if len(peaks) == 0:
            raise RuntimeError(f"No {label} peaks found")

        amps = u[peaks]
        mu = np.median(amps)
        mad = np.median(np.abs(amps - mu))
        rel_amp_err = (1.4826 * mad / mu) if mu != 0 else 0.0

        idx_suggested = np.argmin(np.abs(amps - mu))

        print(f"\nCandidate {label} peaks:")
        for i, p in enumerate(peaks):
            tag = " <--- suggested" if i == idx_suggested else ""
            print(f"[{i:2d}] t={t[p]:.6e} ms | u={u[p]:+.6e} mm{tag}")

        if auto:
            idx = idx_suggested
        else:
            while True:
                choice = input(
                    f"\nChoose {label} peak index [0–{len(peaks)-1}] "
                    "(ENTER = suggested): "
                ).strip()

                if choice == "":
                    idx = idx_suggested
                    break

                if choice.isdigit():
                    idx = int(choice)
                    if 0 <= idx < len(peaks):
                        break

                print("❌ Invalid choice, try again.")

        p = peaks[idx]

        plt.figure(figsize=(10, 4))
        plt.plot(t, u, lw=2)
        plt.scatter(t[peaks], u[peaks], s=50)
        plt.scatter(t[p], u[p], s=140, color="red", edgecolor="black")
        plt.title(f"{label.capitalize()} – selected peak")
        plt.xlabel("Time [ms]")
        plt.ylabel("Displacement [mm]")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_png = basepath + f"_{label.upper()}_PEAK_SELECTION.jpg"
        plt.savefig(out_png, dpi=180)
        plt.close()

        print(f"✔ {label} peak plot saved: {out_png}")

        return {
            "t": t[p],
            "u": u[p],
            "rel_amp_err": rel_amp_err,
            "peak_index": idx
        }


    basepath = csv_path.replace("_TIME_HISTORY.csv", "")

    if not AUTO_PEAK and not sys.stdin.isatty():
        raise RuntimeError(
            "Manual peak selection requires an interactive terminal.\n"
            "Use --auto or run with < /dev/tty"
        )


    # ==================================================
    # PEAK AMPLITUDES (SEMPRE, per attenuazione)
    # ==================================================
    p_s_peak = select_peak(
        t_np, u_s,
        frac=0.40,
        basepath=basepath,
        label="sensor",
        auto=AUTO_PEAK
    )

    p_a_peak = select_peak(
        t_np, u_act,
        frac=0.30,
        basepath=basepath,
        label="actuator",
        auto=AUTO_PEAK
    )


    # ==================================================
    # TEMPI: peak (low f) / onset (high f)
    # ==================================================
    if frequency < FREQ_PHASE_THRESHOLD:

        p_s = p_s_peak
        p_a = p_a_peak
        k = p_s["peak_index"]

    else:
        # ==============================
        # FINESTRA FISICA PER ONSET
        # ==============================
        vp_ref = vp_elastic_from_E(E_mpa)
        tof_ref = x_s / vp_ref   # ms

        onset_s = detect_onset_energy(
            t_np, u_s,
            t_min_ms=0.6 * tof_ref,
            t_max_ms=1.6 * tof_ref
        )

        onset_a = detect_onset_energy(
            t_np, u_act,
            t_min_ms=0.0,
            t_max_ms=0.5 * tof_ref
        )

        p_s = {"t": onset_s["t"], "u": p_s_peak["u"], "peak_index": -1}
        p_a = {"t": onset_a["t"], "u": p_a_peak["u"]}
        k = -1



    # --------------------------
    # TOF, velocity, attenuation
    # --------------------------
    TOF = p_s["t"] - p_a["t"]
    if TOF <= 0:
        raise RuntimeError("Negative TOF")

    vp_FEM = x_s / TOF 
    vp_ref = vp_elastic_from_E(E_mpa)
    vp_FEM = min(vp_FEM, 1.02 * vp_ref)

    alpha_FEM = (1.0 / x_s) * np.log(abs(p_a["u"]) / abs(p_s["u"]))

    method_used = (
        "peaks" if frequency < FREQ_PHASE_THRESHOLD
        else "onset-velocity + peak-attenuation"
    )


    # --------------------------
    # save INFO.csv
    # --------------------------
    out_info = csv_path.replace("_TIME_HISTORY.csv", "_INFO.csv")

    pd.DataFrame([{
        "Dim": Dim,
        "frequency_kHz": frequency,
        "wave_type": wave_tag,
        "x_sensor_mm": x_s,
        "TOF_FEM_ms": TOF,
        "vp_FEM_mm/ms": vp_FEM,
        "alpha_FEM_1/mm": alpha_FEM,
        "postprocess_method": method_used,
        "t_act_ms": p_a["t"],
        "u_act_mm": p_a["u"],
        "t_s_ms": p_s["t"],
        "u_s_mm": p_s["u"],
        "peak_index": k
    }]).to_csv(out_info, index=False)

    print("✔ Saved:", out_info)



# ==========================================================
# ANALYTICAL VISCOUS MODEL (COMPLEX k)
# ==========================================================
rho = 1.78e-3
nu = 0.30

def lame_from_E(E_mpa):
    G = E_mpa / (2.0 * (1.0 + nu))
    lam = (E_mpa * nu) / ((1 + nu) * (1 - 2 * nu))
    return lam, G

def kw_visc(w, rho, k1, c):
    wc = c / rho
    z = np.sqrt(rho / k1)
    fac = (1 + (wc / w)**2)**0.25
    phi = -0.5 * np.arctan(wc / w)
    return w * z * fac * (np.cos(phi) + 1j*np.sin(phi))

def kw_general(w, E_mpa, c):
    lam, G = lame_from_E(E_mpa)
    k1 = lam + 2 * G
    return kw_visc(w, rho, k1, c)

def vp_alpha_analytic(f_kHz, E_mpa, c):
    w = 2.0 * np.pi * f_kHz
    k = kw_general(w, E_mpa, c)
    vp = w / np.real(k)          # [mm/ms]
    alpha = -np.imag(k)          # [1/mm]
    return vp, alpha





# ==========================================================
# TABLE GENERATION (unchanged)
# ==========================================================
def build_tables(root="SIM_DATA_PARALLEL", c_ref_min=None):

    from scipy.optimize import least_squares

    # --------------------------
    # valori nominali per tabelle normative
    # --------------------------
    E_nom_GPa = np.array([50, 75, 100, 125, 150], dtype=float)
    E_nom_MPa = E_nom_GPa * 1000.0

    for freq_tag in sorted(os.listdir(root)):

        if not freq_tag.endswith("kHz"):
            continue

        f_kHz = float(freq_tag.replace("kHz", ""))
        freq_dir = os.path.join(root, freq_tag)

        rows_v, rows_a = [], []

        # ==================================================
        # individua TUTTI i valori di c presenti
        # ==================================================
        c_values = set()

        for E_dir in os.listdir(freq_dir):
            if not E_dir.startswith("E_"):
                continue
            for c_dir in os.listdir(os.path.join(freq_dir, E_dir)):
                if c_dir.startswith("c_"):
                    c_values.add(float(c_dir.split("_")[1]))

        if not c_values:
            print(f"❌ Nessun valore di c trovato per {freq_tag}")
            continue

        c_values = sorted(c_values)

        # ==================================================
        # scegli c di riferimento (minimo)
        # ==================================================
        if c_ref_min is None:
            c_ref = min(c_values)
        else:
            c_ref = c_ref_min

        print(f"\n▶ {freq_tag}: fitting E using c_ref = {c_ref}")

        # ==================================================
        # RACCOLTA FEM per c_ref  → fit di E
        # ==================================================
        E_fem, vp_fem = [], []

        for E_dir in sorted(os.listdir(freq_dir)):
            if not E_dir.startswith("E_"):
                continue

            E_MPa = float(E_dir.split("_")[1])
            c_dir = f"c_{c_ref:.2f}"
            c_path = os.path.join(freq_dir, E_dir, c_dir)

            if not os.path.isdir(c_path):
                continue

            for sub in os.listdir(c_path):
                subp = os.path.join(c_path, sub)
                if not os.path.isdir(subp):
                    continue

                info_files = [
                    x for x in os.listdir(subp)
                    if x.endswith("_INFO.csv")
                ]
                if not info_files:
                    continue

                info = pd.read_csv(os.path.join(subp, info_files[0]))
                E_fem.append(E_MPa)
                vp_fem.append(info["vp_FEM_mm/ms"].iloc[0])

        E_fem = np.array(E_fem)
        vp_fem = np.array(vp_fem)

        if len(E_fem) < 2:
            print(f"⚠ fit E fallito per {freq_tag}")
            continue

        # ==================================================
        # FIT sE (UNA SOLA VOLTA)
        # ==================================================
        def res_vp(sE):
            vp_a, _ = vp_alpha_analytic(f_kHz, sE[0] * E_fem, c_ref)
            return vp_a - vp_fem

        resE = least_squares(res_vp, x0=[1.0])
        sE_fit = resE.x[0]

        print(f"   ✔ sE_fit = {sE_fit:.4f}")

        # ==================================================
        # LOOP SU TUTTI I c → fit SOLO qC
        # ==================================================
        for c in c_values:

            E_fem, alpha_fem = [], []

            for E_dir in sorted(os.listdir(freq_dir)):
                if not E_dir.startswith("E_"):
                    continue

                E_MPa = float(E_dir.split("_")[1])
                c_dir = f"c_{c:.2f}"
                c_path = os.path.join(freq_dir, E_dir, c_dir)

                if not os.path.isdir(c_path):
                    continue

                for sub in os.listdir(c_path):
                    subp = os.path.join(c_path, sub)
                    if not os.path.isdir(subp):
                        continue

                    info_files = [
                        x for x in os.listdir(subp)
                        if x.endswith("_INFO.csv")
                    ]
                    if not info_files:
                        continue

                    info = pd.read_csv(os.path.join(subp, info_files[0]))
                    E_fem.append(E_MPa)
                    alpha_fem.append(info["alpha_FEM_1/mm"].iloc[0])

            E_fem = np.array(E_fem)
            alpha_fem = np.array(alpha_fem)

            if len(E_fem) < 2:
                print(f"⚠ skip c={c} @ {freq_tag}")
                continue

            # --------------------------
            # fit SOLO qC
            # --------------------------
            def res_alpha(qC):
                _, alpha_a = vp_alpha_analytic(
                    f_kHz, sE_fit * E_fem, qC[0] * c
                )
                return alpha_a - alpha_fem

            resC = least_squares(
                res_alpha,
                x0=[1.0],
                bounds=([0.1], [10.0])
            )

            qC_fit = resC.x[0]

            # --------------------------
            # costruzione tabelle
            # --------------------------
            for E_GPa, E_MPa in zip(E_nom_GPa, E_nom_MPa):

                vp_val, alpha_val = vp_alpha_analytic(
                    f_kHz,
                    sE_fit * E_MPa,
                    qC_fit * c
                )

                rows_v.append({
                    "E_GPa": E_GPa,
                    "c": c,
                    "vp_mm_ms": vp_val
                })

                rows_a.append({
                    "E_GPa": E_GPa,
                    "c": c,
                    "alpha_1_mm": alpha_val
                })

        # ==================================================
        # SALVATAGGIO CSV
        # ==================================================
        pd.DataFrame(rows_v).pivot(
            index="E_GPa", columns="c", values="vp_mm_ms"
        ).sort_index().to_csv(f"TABLE_V_{freq_tag}.csv")

        pd.DataFrame(rows_a).pivot(
            index="E_GPa", columns="c", values="alpha_1_mm"
        ).sort_index().to_csv(f"TABLE_ALPHA_{freq_tag}.csv")

        print(f"✔ Tables generated (physically constrained) for {freq_tag}")




def plot_normative_tables(freq_tag):
    """
    Plot vp(E) and alpha(E) for fixed c, given a frequency table,
    with FEM points overlaid.
    """

    V_csv = f"TABLE_V_{freq_tag}.csv"
    A_csv = f"TABLE_ALPHA_{freq_tag}.csv"

    if not (os.path.isfile(V_csv) and os.path.isfile(A_csv)):
        print(f"Tables for {freq_tag} not found, skipping plots.")
        return

    # =========================
    # load normative tables
    # =========================
    V = pd.read_csv(V_csv, index_col=0)
    A = pd.read_csv(A_csv, index_col=0)

    V.index = V.index.astype(float)
    A.index = A.index.astype(float)

    outdir = "NORMATIVE_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    freq_dir = os.path.join("SIM_DATA_PARALLEL", freq_tag)
    E_vals = V.index.to_numpy()

    # ==========================================================
    # loop on c
    # ==========================================================
    for c in V.columns:

        E_fem = []
        vp_fem = []
        alpha_fem = []

        # =========================
        # load FEM data
        # =========================
        for E_dir in os.listdir(freq_dir):
            if not E_dir.startswith("E_"):
                continue

            E_MPa = float(E_dir.split("_")[1])
            E_GPa = E_MPa / 1000.0

            c_dir = f"c_{float(c):.2f}"
            c_path = os.path.join(freq_dir, E_dir, c_dir)

            if not os.path.isdir(c_path):
                continue

            for sub in os.listdir(c_path):
                subp = os.path.join(c_path, sub)
                if not os.path.isdir(subp):
                    continue

                info_files = [
                    x for x in os.listdir(subp)
                    if x.endswith("_INFO.csv")
                ]
                if not info_files:
                    continue

                info = pd.read_csv(os.path.join(subp, info_files[0]))

                E_fem.append(E_GPa)
                vp_fem.append(info["vp_FEM_mm/ms"].iloc[0])
                alpha_fem.append(info["alpha_FEM_1/mm"].iloc[0])

        # =========================
        # vp plot
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(
            E_vals,
            V[c],
            marker="o",
            lw=2,
            label="Normative"
        )

        plt.scatter(
            E_fem,
            vp_fem,
            s=60,
            facecolors="none",
            edgecolors="k",
            label="FEM"
        )

        plt.xlabel("E [GPa]")
        plt.ylabel("Phase velocity $v_p$ [mm/ms]")
        plt.title(f"Phase velocity – c={c}, {freq_tag}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_v = os.path.join(outdir, f"vp_vs_E_c{c}_{freq_tag}.png")
        plt.savefig(out_v, dpi=180)
        plt.close()

        # =========================
        # alpha plot
        # =========================
        plt.figure(figsize=(6, 4))

        plt.plot(
            E_vals,
            A[c],
            marker="o",
            lw=2,
            label="Normative"
        )

        plt.scatter(
            E_fem,
            alpha_fem,
            s=60,
            facecolors="none",
            edgecolors="k",
            label="FEM"
        )

        plt.xlabel("E [GPa]")
        plt.ylabel(r"Attenuation $\alpha$ [1/mm]")
        plt.title(f"Attenuation – c={c}, {freq_tag}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_a = os.path.join(outdir, f"alpha_vs_E_c{c}_{freq_tag}.png")
        plt.savefig(out_a, dpi=180)
        plt.close()

        print(f"✔ Plots saved for c={c}, {freq_tag}")



def plot_normative_all_c(freq_tag):
    """
    Single plot vp(E) and alpha(E) with all c on the same figure,
    with FEM points overlaid.
    """
    V_csv = f"TABLE_V_{freq_tag}.csv"
    A_csv = f"TABLE_ALPHA_{freq_tag}.csv"

    if not (os.path.isfile(V_csv) and os.path.isfile(A_csv)):
        print(f"Tables for {freq_tag} not found, skipping plots.")
        return

    V = pd.read_csv(V_csv, index_col=0)
    A = pd.read_csv(A_csv, index_col=0)

    outdir = "NORMATIVE_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    freq_dir = os.path.join("SIM_DATA_PARALLEL", freq_tag)

    E_vals = V.index.to_numpy()
    c_vals = V.columns.to_numpy()

    # ================= alpha =================
    plt.figure(figsize=(7, 5))

    for c in c_vals:
        plt.plot(E_vals, A[c], marker="o", label=f"Normative c={c}")

    plt.xlabel("E [GPa]")
    plt.ylabel(r"$\alpha$ [1/mm]")
    plt.title(f"Attenuation vs E – {freq_tag}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_a = os.path.join(outdir, f"alpha_vs_E_all_c_{freq_tag}.png")
    plt.savefig(out_a, dpi=180)
    plt.close()

    print(f"✔ Normative plots saved in {outdir}/ for {freq_tag}")

def plot_vp_delta_linear(freq_tag, c_ref="0.1"):
    """
    Plot vp(E,c) - vp(E,c_ref) with linear y-scale.
    """

    V_csv = f"TABLE_V_{freq_tag}.csv"
    if not os.path.isfile(V_csv):
        print(f"Table {V_csv} not found.")
        return

    V = pd.read_csv(V_csv, index_col=0)
    V.index = V.index.astype(float)
    V = V.sort_index()

    if c_ref not in V.columns:
        raise ValueError(f"Reference c={c_ref} not found in table")

    E_vals = V.index.to_numpy()
    vp_ref = V[c_ref].to_numpy()

    outdir = "NORMATIVE_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for c in V.columns:
        if c == c_ref:
            continue

        delta_vp = V[c].to_numpy() - vp_ref

        plt.plot(
            E_vals,
            delta_vp,
            marker="o",
            lw=2,
            label=f"c={c}"
        )

    plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.6)

    plt.xlabel("Young modulus E [GPa]", fontsize=13)
    plt.ylabel(r"$v_p(E,c) - v_p(E,c_{\mathrm{ref}})$ [mm/ms]", fontsize=13)
    plt.title(
        f"Viscous effect on phase velocity – {freq_tag}\n"
        f"(reference c={c_ref})",
        fontsize=14
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(outdir, f"vp_delta_linear_{freq_tag}.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"✔ Linear Δvp plot saved: {out}")





   

from matplotlib.backends.backend_pdf import PdfPages

def export_pretty_tables_pdf(freq_tag):
    """
    Export normative tables (vp and alpha) to a formatted PDF.
    """

    V_csv = f"TABLE_V_{freq_tag}.csv"
    A_csv = f"TABLE_ALPHA_{freq_tag}.csv"

    if not (os.path.isfile(V_csv) and os.path.isfile(A_csv)):
        print(f"⚠ Tables for {freq_tag} not found, skipping PDF.")
        return

    V = pd.read_csv(V_csv, index_col=0)
    A = pd.read_csv(A_csv, index_col=0)

    # numeric ordering
    V.index = V.index.astype(float)
    A.index = A.index.astype(float)
    V = V.sort_index()
    A = A.sort_index()

    pdf_name = f"NORMATIVE_TABLES_{freq_tag}.pdf"

    with PdfPages(pdf_name) as pdf:

        # ==================================================
        # PHASE VELOCITY TABLE
        # ==================================================
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis("off")

        title = (
            "Normative table – Phase velocity $v_p$\n"
            f"Frequency: {freq_tag}\n"
            r"$c$ [g/(mm$\cdot$ms)]"
            r"$\rho = 1.78\ \mathrm{g/cm^3},\ \nu = 0.30$\n"
            r"$c$ [g/(mm$\cdot$ms)]"
        )

        table_data = V.copy()
        table_data.index.name = "E [GPa]"

        # format numbers
        fmt = lambda x: f"{x:,.2f}"
        table_data = table_data.apply(lambda col: col.map(fmt))


        mpl_table = ax.table(
            cellText=table_data.values,
            colLabels=[f"c={c}" for c in table_data.columns],
            rowLabels=table_data.index,
            loc="center"
        )

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(9)
        mpl_table.scale(1.2, 1.4)

        ax.set_title(title, fontsize=12, pad=20)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ==================================================
        # ATTENUATION TABLE
        # ==================================================
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis("off")

        title = (
            "Normative table – Attenuation $\\alpha$\n"
            f"Frequency: {freq_tag}\n"
            r"$c$ [g/(mm$\cdot$ms)]"
            r"$\rho = 1.78\ \mathrm{g/cm^3},\ \nu = 0.30$\n"
            r"$c$ [g/(mm$\cdot$ms)]"
        )

        table_data = A.copy()
        table_data.index.name = "E [GPa]"

        fmt = lambda x: f"{x:.4e}"
        table_data = table_data.apply(lambda col: col.map(fmt))


        mpl_table = ax.table(
            cellText=table_data.values,
            colLabels=[f"c={c}" for c in table_data.columns],
            rowLabels=table_data.index,
            loc="center"
        )

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(9)
        mpl_table.scale(1.2, 1.4)

        ax.set_title(title, fontsize=12, pad=20)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"✔ Pretty tables PDF saved: {pdf_name}")


def load_fem_data(freq_tag, root="SIM_DATA_PARALLEL"):
    """
    Load pure FEM data for a given frequency.
    Returns dict: data[c] = DataFrame(E_GPa, vp, alpha)
    """
    freq_dir = os.path.join(root, freq_tag)
    data = {}

    for E_dir in os.listdir(freq_dir):
        if not E_dir.startswith("E_"):
            continue

        E_MPa = float(E_dir.split("_")[1])
        E_GPa = E_MPa / 1000.0
        E_path = os.path.join(freq_dir, E_dir)

        for c_dir in os.listdir(E_path):
            if not c_dir.startswith("c_"):
                continue

            c = float(c_dir.split("_")[1])
            c_path = os.path.join(E_path, c_dir)

            for sub in os.listdir(c_path):
                subp = os.path.join(c_path, sub)
                if not os.path.isdir(subp):
                    continue

                info_files = [
                    os.path.join(subp, x)
                    for x in os.listdir(subp)
                    if x.endswith("_INFO.csv")
                ]
                if not info_files:
                    continue

                info = pd.read_csv(info_files[0])

                data.setdefault(c, []).append({
                    "E_GPa": E_GPa,
                    "vp": info["vp_FEM_mm/ms"].iloc[0],
                    "alpha": info["alpha_FEM_1/mm"].iloc[0]
                })

    # convert to DataFrames
    for c in data:
        data[c] = pd.DataFrame(data[c]).sort_values("E_GPa")

    return data

def plot_alpha_vs_E_all_c_FEM(freq_tag):
    data = load_fem_data(freq_tag)

    outdir = "FEM_ONLY_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for c, df in data.items():
        plt.plot(
            df["E_GPa"],
            df["alpha"],
            marker="o",
            lw=2,
            label=f"c={c}"
        )

    plt.xlabel("E [GPa]")
    plt.ylabel(r"$\alpha$ [1/mm]")
    plt.title(f"FEM-only attenuation vs E – {freq_tag}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(outdir, f"alpha_vs_E_all_c_FEM_{freq_tag}.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"✔ FEM-only alpha plot saved: {out}")


def plot_vp_delta_linear_FEM(freq_tag, c_ref):
    data = load_fem_data(freq_tag)

    if c_ref not in data:
        raise ValueError(f"Reference c={c_ref} not found in FEM data")

    df_ref = data[c_ref]
    E_vals = df_ref["E_GPa"].to_numpy()
    vp_ref = df_ref["vp"].to_numpy()

    outdir = "FEM_ONLY_PLOTS"
    os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for c, df in data.items():
        if c == c_ref:
            continue

        vp_interp = np.interp(E_vals, df["E_GPa"], df["vp"])
        delta_vp = vp_interp - vp_ref

        plt.plot(
            E_vals,
            delta_vp,
            marker="o",
            lw=2,
            label=f"c={c}"
        )

    plt.axhline(0.0, color="k", lw=1, ls="--", alpha=0.6)

    plt.xlabel("Young modulus E [GPa]")
    plt.ylabel(r"$\Delta v_p$ [mm/ms]")
    plt.title(
        f"FEM-only phase velocity shift – {freq_tag}\n"
        f"(reference c={c_ref})"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(outdir, f"vp_delta_linear_FEM_{freq_tag}.png")
    plt.savefig(out, dpi=180)
    plt.close()

    print(f"✔ FEM-only Δvp plot saved: {out}")



if not RUN_SINGLE:
    build_tables()

    freqs = ["250kHz", "375kHz", "500kHz"]

    for freq in freqs:
        plot_normative_tables(freq)
        plot_normative_all_c(freq)
        plot_vp_delta_linear(freq)
        export_pretty_tables_pdf(freq)

        plot_alpha_vs_E_all_c_FEM(freq)
        plot_vp_delta_linear_FEM(freq, c_ref=0.10)

    # ===== NEW GLOBAL PLOTS =====
    plot_E_c_correlation_all_freq(
        freqs, quantity="vp", c_value=0.8
    )

    plot_E_c_correlation_all_freq(
        freqs, quantity="alpha", c_value=0.8
    )



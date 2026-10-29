"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

"""
Action to compute – scalar wave propagation (2D / 3D FEM)
"""

# =============================================================================
# IMPORTS
# =============================================================================
from firedrake import *
from irksome import Dt, StageDerivativeNystromTimeStepper
from irksome.ButcherTableaux import GaussLegendre, RadauIIA, LobattoIIIC, BackwardEuler
import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description="Scalar wave propagation (2D/3D, Firedrake + Irksome)"
)

parser.add_argument("--dim", type=int, choices=[2, 3], required=True,
                    help="Spatial dimension (2 or 3)")

parser.add_argument("--frequency", type=float, required=True)
parser.add_argument("--eta", type=float, required=True)
parser.add_argument("--xi", type=float, required=True)
parser.add_argument("--nt", type=int, required=True)
parser.add_argument(
    "--method",
    type=str,
    default="RadauIIA",
    choices=["RadauIIA", "GaussLegendre", "LobattoIIIC", "BackwardEuler"],
    help="IRK time integrator (default: BackwardEuler)"
)

parser.add_argument("--ns", type=int, default=4)

parser.add_argument("--nx", type=int, required=True)
parser.add_argument("--ny", type=int, required=True)
parser.add_argument("--nz", type=int, default=1)
parser.add_argument(
    "--source",
    type=str,
    default="plane",
    choices=["plane", "circular", "spherical"],
    help="Source type: plane (default), circular (2D), spherical (3D)"
)
parser.add_argument("--outdir", type=str, default="multidimensional_dissipative_first_gradient_model")

args = parser.parse_args()


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
Dim        = args.dim
frequency  = args.frequency
eta        = args.eta
xi         = args.xi
nt_factor  = args.nt
method_name = args.method
ns         = args.ns

nx = args.nx
ny = args.ny
nz = args.nz

xlength = 200.0
ylength = 20.0
zlength = 20.0

outroot = args.outdir


# =============================================================================
# IRK METHOD
# =============================================================================
method_dict = {
    "GaussLegendre": GaussLegendre,
    "RadauIIA": RadauIIA,
    "LobattoIIIC": LobattoIIIC,
    "BackwardEuler": BackwardEuler,
}
method_class = method_dict[method_name]
try:
    butcher_tableau = method_class(ns)
except TypeError:
    butcher_tableau = method_class()

print("\n================ TIME INTEGRATOR =================")
print(f"Method name       : {method_name}")
print(f"IRK stages (ns)   : {ns}")
print(f"Butcher tableau  : {butcher_tableau}")
print("=================================================\n")

# =============================================================================
# MATERIAL PROPERTIES
# =============================================================================
rho = 2.3e-3
nu  = 0.2
E   = 30e3

G   = E / (2.0 * (1.0 + nu))

c  = eta
c1 = xi

wave_modulus = E / ((1 + nu) * (1 - 2 * nu))
lam = wave_modulus * nu
k1  = lam + 2 * G
mu = G


wave_speed_elastic = (wave_modulus * (1.0 - nu) / rho) ** 0.5

# =============================================================================
# MODEL TAG FOR TITLE
# =============================================================================
if eta == 0 and xi == 0:
    model_str = r"Elastic medium"
elif eta != 0 and xi == 0:
    model_str = fr"c={eta}"
elif eta == 0 and xi != 0:
    model_str = fr"c_{1}={xi}"
else:
    model_str = fr"c={eta},\;c_{1}={xi}"  


# =============================================================================
# SOURCE CONFIGURATION
# =============================================================================

# Allowed values:
#   "plane"     -> plane wave (uniform excitation on x = 0)
#   "circular"  -> circular wave (localized patch, 2D only)
#   "spherical" -> spherical wave (localized patch, 3D only)
SOURCE_TYPE = args.source


# Source amplitude [mm]
SOURCE_AMPLITUDE = 1.0 / 100.0


# =============================================================================
# WAVE TYPE TAG (for folders and files)
# =============================================================================
if SOURCE_TYPE == "plane":
    wave_tag = "plane"
elif SOURCE_TYPE == "circular":
    wave_tag = "circular"
elif SOURCE_TYPE == "spherical":
    wave_tag = "spherical"
else:
    raise ValueError("Unknown SOURCE_TYPE")



# =============================================================================
# MESH (2D / 3D VIA IF)
# =============================================================================
if Dim == 2:
    mesh = RectangleMesh(nx, ny, xlength, ylength)
elif Dim == 3:
    mesh = BoxMesh(nx, ny, nz, xlength, ylength, zlength)


# =============================================================================
# FUNCTION SPACE
# =============================================================================
V = VectorFunctionSpace(mesh, "CG", 1)   # <-- vettoriale

u  = Function(V, name="u")
ut = Function(V, name="ut")
v  = TestFunction(V)

def eps(w):
    return sym(grad(w))




# =============================================================================
# TIME
# =============================================================================
t_pulse = 1.0 / frequency
dt      = Constant(t_pulse / nt_factor)
t       = Constant(0.0)
tend    = 2*xlength / wave_speed_elastic


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================
x = SpatialCoordinate(mesh)

time_factor = -SOURCE_AMPLITUDE * cos(2 * pi * frequency * t)

# supporto spaziale solo per circular/spherical, per plane uniforme:
if SOURCE_TYPE == "plane":
    spatial_factor = Constant(1.0)

elif SOURCE_TYPE == "circular" and Dim == 2:
    yc = 0.5 * ylength

    if frequency >= 750.0:
        # --- wavelength-scaled patch (ONLY 750 & 1000 kHz) ---
        v_ref = wave_speed_elastic        # mm/ms
        lam_mm = v_ref / frequency        # lambda = v / f  [mm]
        n_lambda = 0.5                    # tunable (≈ same size as old r0)
        r0 = min(0.5 * n_lambda * lam_mm, 0.49 * ylength)
    else:
        # --- original behaviour ---
        r0 = 0.2 * ylength

    spatial_factor = conditional(abs(x[1] - yc) <= r0, 1.0, 0.0)

elif SOURCE_TYPE == "spherical" and Dim == 3:
    yc, zc = 0.5 * ylength, 0.5 * zlength

    if frequency >= 750.0:
        v_ref = wave_speed_elastic
        lam_mm = v_ref / frequency
        n_lambda = 0.5
        r0 = min(0.5 * n_lambda * lam_mm, 0.49 * min(ylength, zlength))
    else:
        r0 = 0.2 * min(ylength, zlength)

    spatial_factor = conditional(
        (x[1] - yc)**2 + (x[2] - zc)**2 <= r0**2,
        1.0, 0.0
    )

else:
    raise ValueError("SOURCE_TYPE incompatible with dimension")

bcExpr = time_factor * spatial_factor
bcs = []


# ACTUATOR: imponi spostamento in x sul bordo x=0 (id=1)
bcs.append(DirichletBC(V.sub(0), bcExpr, 1))

# Uniaxial strain: blocca u_y sulle pareti (y=0 id=3, y=ylength id=4)
# + blocca u_y anche sul bordo sorgente (x=0) per evitare shear spurio
bcs.append(DirichletBC(V.sub(1), Constant(0.0), 1))
bcs.append(DirichletBC(V.sub(1), Constant(0.0), 3))
bcs.append(DirichletBC(V.sub(1), Constant(0.0), 4))

# Se 3D: blocca anche u_z sulle pareti z (se vuoi uniaxial strain 3D “puro”)
if Dim == 3:
    # attenzione: gli id per z=0 e z=zlength dipendono dalla numerazione del BoxMesh
    # qui NON li aggiungo per "modifica minima"
    pass





# =============================================================================
# WEAK FORM 
# =============================================================================
F = (
    rho * dot(Dt(u, 2), v)
  + c   * dot(ut, v)
  + 2*mu * inner(eps(u), eps(v))
  + lam  * div(u) * div(v)
  + c1  * inner(grad(ut), grad(v))   # dissipazione first-gradient sulla velocità
) * dx



# =============================================================================
# IRK STEPPER
# =============================================================================
luparams_2d_precise = {
    "mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu"
}

params_3d_simple = {
    "mat_type": "aij",
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,            
    "ksp_max_it": 200,
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
}

if Dim == 2:
    
    solver_parameters = params_3d_simple
    
elif Dim == 3:
    solver_parameters = params_3d_simple
else:
    solver_parameters = luparams_2d_precise 

stepper = StageDerivativeNystromTimeStepper(
    F, butcher_tableau, t, dt, u, ut,
    bcs=bcs,
    solver_parameters=solver_parameters
)



# =============================================================================
# OUTPUT
# =============================================================================
if eta == 0 and xi == 0:
    model_tag = "elastic"
else:
    model_tag = f"eta{eta}_xi{xi}"

file_title = f"{Dim}D_{frequency}kHz_{wave_tag}_{model_tag}_nt{nt_factor}_nx{nx}"

outdir = os.path.join(outroot, file_title)
os.makedirs(outdir, exist_ok=True)

pvd_root = "/data/nicola/multidimensional_dissipative_first_gradient_model"
pvd_dir = os.path.join(pvd_root, file_title)
os.makedirs(pvd_dir, exist_ok=True)


u_file = File(os.path.join(pvd_dir, f"{file_title}_u.pvd"), mode="w")




# =============================================================================
# SENSOR DEFINITION (SINGLE SENSOR)
# =============================================================================
x_act = 0.0
x_s   = 0.5 * xlength
y_mid = 0.5 * ylength
z_mid = 0.5 * zlength


# =============================================================================
# STORAGE
# =============================================================================
t_hist = []
u_act  = []
u_s    = []

# =============================================================================
# TIME HISTORY EMISSION TIMER (WALL-CLOCK)
# =============================================================================
time_history_start = time.perf_counter()


# =============================================================================
# TIME LOOP
# =============================================================================

rank0 = COMM_WORLD.rank == 0

step_counter = 0
nsteps = int(tend / float(dt)) + 1

t_hist = np.empty(nsteps)
u_act  = np.empty(nsteps)
u_s    = np.empty(nsteps)

PRINT_EVERY = max(1, nsteps // 10)
t_start_wall = time.perf_counter()

# --- punti di campionamento ---
if Dim == 2:
    sample_points = [
        (x_act, y_mid),
        (x_s,   y_mid),
    ]
else:
    sample_points = [
        (x_act, y_mid, z_mid),
        (x_s,   y_mid, z_mid),
    ]

pe = PointEvaluator(mesh, sample_points)


i = 0
while float(t) < tend:

    stepper.advance()
    t_now = float(t)

    # --- sampling FEM ---
    values = pe.evaluate(u)


    ua_vec = values[0]
    us_vec = values[1]

    ua = ua_vec[0]        # componente x
    us = us_vec[0]

    t_hist[i] = t_now
    u_act[i]  = ua
    u_s[i]    = us

    i += 1
    step_counter += 1

    if step_counter % PRINT_EVERY == 0 and rank0:
        elapsed = time.perf_counter() - t_start_wall
        perc = 100.0 * step_counter / nsteps
        print(
            f"[{perc:6.2f}%] "
            f"step {step_counter}/{nsteps} | "
            f"t = {t_now:8.4f} ms | "
            f"elapsed = {elapsed:8.1f} s",
            flush=True
        )

    t.assign(t_now + float(dt))


# =============================================================================
# TIME HISTORY EMISSION TIMER STOP
# =============================================================================
time_history_end = time.perf_counter()
time_history_duration = time_history_end - time_history_start
if i > 0:
    avg_time_per_step = time_history_duration / i
else:
    avg_time_per_step = float("nan")



# =============================================================================
# SAVE TIME HISTORY CSV (ACTUATOR + SENSOR)
# =============================================================================

time_history_df = pd.DataFrame({
    "t_ms": t_hist[:i],
    "u_act_mm": u_act[:i],
    "u_s_mm": u_s[:i],
    "frequency_kHz": frequency,
    "eta": eta,
    "xi": xi,
    "wave_type": wave_tag,
    "Dim": Dim
})


time_history_df.to_csv(
    os.path.join(outdir, f"{file_title}_TIME_HISTORY.csv"),
    index=False
)


print("✔ Time history CSV saved.")


# =============================================================================
# WRITE TIME HISTORY TIMING TO TXT
# =============================================================================

timing_txt = os.path.join(outdir, "TIME_HISTORY_TIMING.txt")

with open(timing_txt, "a") as f:
    f.write("\n========================================\n")
    f.write("MULTIDIMENSIONAL FEM TIME HISTORY\n")
    f.write("========================================\n")
    f.write(f"Dimension           = {Dim}D\n")
    f.write(f"Wave type           = {wave_tag}\n")
    f.write(f"Frequency [kHz]     = {frequency}\n")
    f.write(f"eta                 = {eta}\n")
    f.write(f"xi                  = {xi}\n")
    f.write("----------------------------------------\n")
    f.write(f"nx, ny, nz          = {nx}, {ny}, {nz}\n")
    f.write(f"nt_factor           = {nt_factor}\n")
    f.write(f"dt [ms]             = {float(dt):.6e}\n")
    f.write("----------------------------------------\n")
    f.write(f"IRK method          = {method_name}\n")
    f.write(f"IRK stages (ns)     = {ns}\n")
    f.write("----------------------------------------\n")
    f.write(f"Time steps executed = {i}\n")
    f.write(f"Wall time total [s] = {time_history_duration:.6f}\n")
    f.write(f"Time / step [s]     = {avg_time_per_step:.6e}\n")
    f.write("========================================\n")


def pretty_print_table(csv_file, quantity_name, value_unit):
    """
    Pretty print a normative table with physical units.
    """

    if not os.path.isfile(csv_file):
        print(f"❌ File {csv_file} not found")
        return

    df = pd.read_csv(csv_file, index_col=0)
    df.index.name = "E [GPa]"

    # rinomina colonne con unità fisica di c
    df.columns = [
        f"c = {float(c):.2f} g/(mm·ms)" for c in df.columns
    ]

    # format numerico
    if quantity_name == "vp":
        fmt = "{:.1f}"   # mm/ms
    else:
        fmt = "{:.4e}"  # alpha

    df_fmt = df.applymap(lambda x: fmt.format(x))

    print("\n" + "=" * 80)
    print(f"{quantity_name.upper()} TABLE  ({value_unit})")
    print("=" * 80)
    print(df_fmt.to_string())
    print("=" * 80 + "\n")

pretty_print_table(
    "TABLE_V_375kHz.csv",
    quantity_name="vp",
    value_unit="mm/ms"
)

pretty_print_table(
    "TABLE_ALPHA_375kHz.csv",
    quantity_name="alpha",
    value_unit="1/mm"
)

"""
Supply code of the publication:

De Fazio, N., Abali, B.E., Placidi, L., and Luciano, R. (2026). Redefining ultrasonic testing: determining elastic and viscous damping parameters via the digital twinning technology

"""
__author__ = "B. Emek Abali and Nicola De Fazio"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html


"""
1D scalar wave propagation – CLEAN SIMULATOR
Firedrake + Irksome

• FEM simulation only
• Saves time histories (actuator + sensors)
• NO peak detection
• NO postprocessing

"""

# ==========================================================
# IMPORTS
# ==========================================================
from firedrake import *
from irksome import Dt, StageDerivativeNystromTimeStepper
from irksome.ButcherTableaux import (
    GaussLegendre, RadauIIA, LobattoIIIC, BackwardEuler
)
import argparse
import numpy as np
import pandas as pd
import os


# ==========================================================
# ARGUMENT PARSING
# ==========================================================
parser = argparse.ArgumentParser(
    description="1D scalar wave propagation (FEM only)"
)

parser.add_argument("--frequency", type=float, required=True,
                    help="Frequency [kHz]")
parser.add_argument("--eta", type=float, required=True,
                    help="Viscosity parameter")
parser.add_argument("--xi", type=float, required=True,
                    help="Gradient parameter")
parser.add_argument("--nt", type=int, required=True,
                    help="Time subdivision factor")
parser.add_argument("--nx", type=int, required=True,
                    help="Number of elements")
parser.add_argument("--method", type=str, default="BackwardEuler",
                    choices=["GaussLegendre", "RadauIIA",
                             "LobattoIIIC", "BackwardEuler"])
parser.add_argument("--ns", type=int, default=1,
                    help="IRK stages")
parser.add_argument("--sensors", nargs="+", type=float,
                    default=[75.0, 100.0, 160.0],
                    help="Sensor positions [mm]")

args = parser.parse_args()


# ==========================================================
# PHYSICAL PARAMETERS
# ==========================================================
frequency = args.frequency
eta = args.eta
xi = args.xi

rho = 2.3e-3
nu  = 0.0
E   = 30e3

G   = E / (2.0 * (1.0 + nu))
wave_modulus = E / ((1 + nu) * (1 - 2 * nu))
lam = wave_modulus * nu
k1  = lam + 2 * G

c  = eta
c1 = xi

# geometry
xlength = 200.0
x_act = 0.0
sensor_positions = args.sensors


# ==========================================================
# TIME PARAMETERS
# ==========================================================
t_pulse = 1.0 / frequency
dt = Constant(t_pulse / args.nt)
t  = Constant(0.0)

wave_speed_elastic = (wave_modulus * (1.0 - nu) / rho) ** 0.5
tend = 2 * xlength / wave_speed_elastic


# ==========================================================
# MESH & FUNCTION SPACE
# ==========================================================
mesh = IntervalMesh(args.nx, xlength)
V = FunctionSpace(mesh, "CG", 1)

u  = Function(V, name="u")
ut = Function(V, name="ut")
v  = TestFunction(V)


# ==========================================================
# TIME INTEGRATOR
# ==========================================================
method_dict = {
    "GaussLegendre": GaussLegendre,
    "RadauIIA": RadauIIA,
    "LobattoIIIC": LobattoIIIC,
    "BackwardEuler": BackwardEuler,
}
method_class = method_dict[args.method]

try:
    tableau = method_class(args.ns)
except TypeError:
    tableau = method_class()


# ==========================================================
# BOUNDARY CONDITION (ACTUATOR)
# ==========================================================
x = SpatialCoordinate(mesh)
SOURCE_AMPLITUDE = 1.0 / 100.0

bc_expr = -SOURCE_AMPLITUDE * cos(2 * pi * frequency * t)
bcs = [DirichletBC(V, bc_expr, 1)]


# ==========================================================
# WEAK FORM
# ==========================================================
F = (
    rho * Dt(u, 2) * v
  + c  * ut * v
  + k1 * u.dx(0) * v.dx(0)
  + c1 * ut.dx(0) * v.dx(0)
) * dx


# ==========================================================
# SOLVER
# ==========================================================
solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu"
}

stepper = StageDerivativeNystromTimeStepper(
    F, tableau, t, dt, u, ut,
    bcs=bcs,
    solver_parameters=solver_parameters
)


# ==========================================================
# STORAGE
# ==========================================================
max_steps = int(tend / float(dt)) + 2

t_hist = np.empty(max_steps)
u_act  = np.empty(max_steps)

u_sensors = {
    xs: np.empty(max_steps)
    for xs in sensor_positions
}

# ==========================================================
# TIME LOOP
# ==========================================================
i = 0
run_status = "OK"
error_msg = ""
nsteps = int(tend / float(dt)) + 1
PRINT_EVERY = max(1, nsteps // 100)   # stampa ~100 volte in totale


try:
    while float(t) < tend:

        stepper.advance()
        t_now = float(t)

        # sample points
        pts = [x_act] + sensor_positions
        values = u.at(pts)

        t_hist[i] = t_now
        u_act[i] = values[0]

        for j, xs in enumerate(sensor_positions):
            u_sensors[xs][i] = values[j + 1]

        if i % PRINT_EVERY == 0:
            print(f"Step {i+1} / {nsteps}", flush=True)


        # -------- TERMINAL OUTPUT (EVERY STEP) --------
        #msg = (
            #f"step={i:6d}  "
            #f"t={t_now:12.6e} ms  "
            #f"u_act={u_act[i]: .6e}"
        #)
        #for xs in sensor_positions:
            #msg += f"  u_s{int(xs)}={u_sensors[xs][i]: .6e}"
        #print(msg)
        # ----------------------------------------------

        i += 1
        t.assign(t_now + float(dt))

except Exception as e:
    run_status = "CRASHED"
    error_msg = str(e)



    # ----------------------------------------------

    i += 1
    t.assign(t_now + float(dt))

    # ----------------------------------------------

    # ---------DEBUG DIVERGENCES FOR GRADIENT MODEL-----------
    #i += 1
    #t.assign(t_now + float(dt))

    #u_all = np.concatenate(list(u_sensors.values()))

    #u_max = np.max(np.abs(u_all))
    #u_rms = np.sqrt(np.mean(u_all**2))
    #n = len(u_all)
    #u_max_tail = np.max(np.abs(u_all[int(0.8*n):]))
    #m = xlength / args.nx
    #dt_val = float(dt)
    #CFL_grad = xi * dt_val / (rho * m**2)

    #diagnostic_row = {
        #"nx": args.nx,
        #"nt": args.nt,
        #"dt": dt_val,
        #"m": m,
        #"xi": xi,
        #"CFL_grad": CFL_grad,
        #"u_max": u_max,
        #"u_max_tail": u_max_tail,
        #"u_rms": u_rms
    #}   


# ==========================================================
# TRUNCATE ARRAYS
# ==========================================================
t_hist = t_hist[:i]
u_act  = u_act[:i]
for xs in sensor_positions:
    u_sensors[xs] = u_sensors[xs][:i]

# ==========================================================
# OUTPUT DIRECTORY (needed for diagnostics too)
# ==========================================================
if eta == 0 and xi == 0:
    model_tag = "elastic"
else:
    model_tag = f"eta{eta}_xi{xi}"

outdir = f"results_1D_{frequency}kHz_eta{eta}_xi{xi}"
os.makedirs(outdir, exist_ok=True)


# ==========================================================
# DIAGNOSTIC SUMMARY (ONE ROW PER SIMULATION)
# ==========================================================

if i > 0:

    # ---- reflection-aware time window ----
    c_el = wave_speed_elastic
    t_refl_min = min((2*xlength - xs) / c_el for xs in sensor_positions)
    t_cut = 0.9 * t_refl_min

    mask = t_hist[:i] <= t_cut

    # ---- collect sensor-only data ----
    u_list = []
    for xs in sensor_positions:
        u_list.append(u_sensors[xs][:i][mask])

    if any(len(u) == 0 for u in u_list):
        # fallback: no reflection cut possible
        u_all = np.concatenate([u_sensors[xs][:i] for xs in sensor_positions])
    else:
        u_all = np.concatenate(u_list)

    # ---- diagnostics ----
    u_max = np.max(np.abs(u_all))
    u_rms = np.sqrt(np.mean(u_all**2))

    n = len(u_all)
    u_max_tail = np.max(np.abs(u_all[int(0.8*n):]))

else:
    u_max = np.nan
    u_rms = np.nan
    u_max_tail = np.nan



diagnostic_row = {
    "nx": args.nx,
    "nt": args.nt,
    "dt": float(dt),
    "m": xlength / args.nx,
    "frequency_kHz": frequency,
    "eta": eta,
    "xi": xi,
    "CFL_grad": xi * float(dt) / (rho * (xlength / args.nx)**2),
    "u_max": u_max,
    "u_max_tail": u_max_tail,
    "u_rms": u_rms,
    "method": args.method,
    "ns": args.ns,
    "status": run_status,
    "error": error_msg
}

diag_df = pd.DataFrame([diagnostic_row])

diag_out = os.path.join(
    outdir,
    "GRADIENT_DIAGNOSTIC_TABLE.csv"
)

# append if exists
if os.path.exists(diag_out):
    diag_df.to_csv(diag_out, mode="a", header=False, index=False)
else:
    diag_df.to_csv(diag_out, index=False)


# ==========================================================
# SAVE CSV
# ==========================================================
if eta == 0 and xi == 0:
    model_tag = "elastic"
else:
    model_tag = f"eta{eta}_xi{xi}"

file_title = (
    f"1D_{frequency}kHz_{model_tag}_"
    f"nt{args.nt}_nx{args.nx}"
)

outdir = f"results_1D_{frequency}kHz_eta{eta}_xi{xi}"
os.makedirs(outdir, exist_ok=True)

data = {
    "t_ms": t_hist,
    "u_act_mm": u_act,
}

for xs in sensor_positions:
    data[f"u_s_{int(xs)}mm"] = u_sensors[xs]

data.update({
    "frequency_kHz": frequency,
    "eta": eta,
    "xi": xi,
    "nx": args.nx,
    "nt_factor": args.nt,
})

df = pd.DataFrame(data)
outfile = os.path.join(outdir, f"{file_title}_TIME_HISTORY.csv")
df.to_csv(outfile, index=False)

print("\n✔ 1D FEM simulation completed")
print(f"✔ Time history saved to:\n  → {outfile}\n")

import os
os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.gridspec as gridspec

color_vec = ['r','b','g','b','k','y','c','m','olive']


plot_isp = False
plot_thrust = False
plot_mass = False
plot_firstCx0 = False
plot_boosterCx0 = False
plot_alt = False
plot_partition = False
plot_propmass = True


import matplotlib.pyplot as plt

def plot_sens(x_vals, y_vals, fig_num=1, subplot=(1,1,1), 
                    xlabel=None, ylabel=None, title=None, legend=None, style='-o', rel_plot=False, **kwargs):
    """
    Plot a graph into a specific subplot of a figure.

    Parameters
    ----------
    x_vals : array-like
        X-axis values.
    y_vals : array-like
        Y-axis values.
    fig_num : int, optional
        Figure number. If not exists, it is created.
    subplot : tuple (i, j, n), optional
        Subplot definition: i = rows, j = cols, n = index.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str, optional
        Subplot title.
    style : str, optional
        Line/marker style (default: '-o').
    kwargs : dict
        Extra keyword arguments passed to plt.plot().
    """
    # Get or create figure
    fig = plt.figure(fig_num, figsize=(16,10), constrained_layout=True)
    
    # Activate subplot
    ax = plt.subplot(*subplot)

    i_0 = np.argmin(np.abs(x_vals))
    Slope = (y_vals[i_0+1]-y_vals[i_0-1])/(x_vals[i_0+1]-x_vals[i_0-1])
    if rel_plot:
        y_vals = (y_vals-y_vals[i_0])/y_vals[0]

    if np.abs(Slope) < 1e-7:
        return
    
    # Plot
    if legend:
        label = f"{legend}. Slope: {Slope:2.2}"
        ax.plot(x_vals, y_vals, style, label=label, **kwargs)
    else:
        ax.plot(x_vals, y_vals, style, **kwargs)
    
    # Labels and title
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)
    if title:
        ax.set_title(title, fontsize=20)
    if legend:
        ax.legend(ncol = 2, fontsize=14)    
    # Grid
    ax.grid(True)
    
    return ax

def _centered_slope(x, y):
    """Return slope dy/dx at x≈0 using centered (or one-sided) difference, and nominal y0."""
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    # ensure ascending x for a clean neighborhood around zero
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    i0 = int(np.argmin(np.abs(x)))
    # one-sided if at edges, else centered
    if i0 == 0:
        dx = x[1] - x[0]
        dy = y[1] - y[0]
    elif i0 == len(x) - 1:
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
    else:
        dx = x[i0 + 1] - x[i0 - 1]
        dy = y[i0 + 1] - y[i0 - 1]
    if abs(dx) < 1e-12:
        return 0.0, y[i0]
    return float(dy / dx), float(y[i0])

def make_sensitivity_table_latex(
    title, label,
    met_first, met_second,                # e.g., "FirstStageIsp", "SecondStageIsp"
    x_first_vals, x_second_vals,          # x arrays for those sweeps
    per_step_first, per_step_second,      # step reported in x-units (e.g., 1.0 %pt, 0.01 frac, 100 m, 1000 kg)
    step_desc_first, step_desc_second,    # text for the step in the column header
    vehicles=("Falcon9","Starship"),
    orbits=("LEO","MEO","TLI"),
    recovs=("EXP","ASDS","RTLS"),
    nominal_unit="t", rel_unit="\\%"
):
    def _g(name):
        if name not in globals():
            raise KeyError(f"Missing variable: {name}")
        return globals()[name]

    def _centered_slope(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        order = np.argsort(x); x = x[order]; y = y[order]
        i0 = int(np.argmin(np.abs(x)))
        if i0 == 0:
            dx, dy = x[1]-x[0], y[1]-y[0]
        elif i0 == len(x)-1:
            dx, dy = x[-1]-x[-2], y[-1]-y[-2]
        else:
            dx, dy = x[i0+1]-x[i0-1], y[i0+1]-y[i0-1]
        if abs(dx) < 1e-12: return 0.0, float(y[i0])
        return float(dy/dx), float(y[i0])

    rows_per_vehicle = len(orbits) * len(recovs)
    def fmt(v, nd=2): return f"{v:.{nd}f}"

    lines = []
    lines += [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{title}}}",
        f"  \\label{{{label}}}",
        "  \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{|c|c|c|c|cc|cc|}",
        "      \\toprule",
        "      Vehicle & Mission & Recovery & Nominal [t] & "
        f"\\makecell{{$\\Delta$ Payload \\\\ {step_desc_first} [{nominal_unit}]}} & "
        f"\\makecell{{Rel. Sens. \\\\ (First) [{rel_unit}]}} & "
        f"\\makecell{{$\\Delta$ Payload \\\\ {step_desc_second} [{nominal_unit}]}} & "
        f"\\makecell{{Rel. Sens. \\\\ (Second) [{rel_unit}]}} \\\\",
        "      \\midrule",
    ]

    for vi, veh in enumerate(vehicles):
        veh_label = "Falcon 9" if veh == "Falcon9" else "Starship"
        vehicle_first_row = True

        for oi, orb in enumerate(orbits):
            # build three recovery rows for this orbit
            for ri, rec in enumerate(recovs):
                # fetch y arrays
                y1 = _g(f"{veh}_{orb}_{rec}_{met_first}")
                y2 = _g(f"{veh}_{orb}_{rec}_{met_second}")
                # slopes at nominal and nominal payload
                m1, y0_1 = _centered_slope(x_first_vals, y1)
                m2, y0_2 = _centered_slope(x_second_vals, y2)
                y0 = y0_1  # use first-series nominal as reference

                d1 = m1 * per_step_first
                d2 = m2 * per_step_second
                rel1 = (d1 / y0 * 100.0) if abs(y0) > 1e-12 else 0.0
                rel2 = (d2 / y0 * 100.0) if abs(y0) > 1e-12 else 0.0

                # Cells for the leftmost columns
                if vehicle_first_row and ri == 0:
                    veh_cell = f"\\multirow{{{rows_per_vehicle}}}{{*}}{{{veh_label}}}"
                else:
                    veh_cell = "&"  # skip vehicle column (spanned)

                if ri == 0:
                    mission_cell = f"\\multirow{{3}}{{*}}{{{orb}}}"
                else:
                    mission_cell = ""  # skip mission column (spanned)

                # Compose line
                # When mission_cell is empty, we must emit an extra '&' to step over that column.
                if ri == 0:
                    line = (
                        f"      {veh_cell} & {mission_cell} & {rec} & "
                        f"{fmt(y0)} & {fmt(d1)} & {fmt(rel1)} & {fmt(d2)} & {fmt(rel2)} \\\\"
                    )
                else:
                    line = (
                        f"      &  & {rec} & "
                        f"{fmt(y0)} & {fmt(d1)} & {fmt(rel1)} & {fmt(d2)} & {fmt(rel2)} \\\\"
                    )
                lines += [line]
                vehicle_first_row = False

            # horizontal separation between orbit blocks (within same vehicle)
            if oi < len(orbits) - 1:
                lines += ["        \\cmidrule(lr){2-8}"]

        # midrule between vehicles
        if vi < len(vehicles) - 1:
            lines += ["      \\midrule"]

    lines += [
        "      \\bottomrule",
        "    \\end{tabular}",
        "  }",
        "\\end{table}"
    ]
    return "\n".join(lines)


# ==============================
# Example: ISP table (matches your example semantics)
# x arrays are in percent points → per_step = 1.0
# ==============================
def print_isp_sensitivity_table():
    latex = make_sensitivity_table_latex(
        title="Payload mass sensitivity of Falcon~9 and Starship to ISP variations. Values show change in payload per +1\\% ISP increase for first and second stages.",
        label="tab:isp_sensitivity",
        met_first="FirstStageIsp",
        met_second="SecondStageIsp",
        x_first_vals=FirstStageIsp,
        x_second_vals=SecondStageIsp,
        per_step_first=1.0,     # +1%pt
        per_step_second=1.0,    # +1%pt
        step_desc_first="+1\\% First-stage ISP",
        step_desc_second="+1\\% Second-stage ISP",
    )
    print("="*30 + "ISP Table"+"="*30)
    print(latex)

# ==============================
# Convenience wrappers for other metrics
# (choose the reporting step you want to see)
# ==============================

def print_thrust_sensitivity_table():
    # x arrays here are FRACTIONS (e.g., 0.01 = +1%), so per_step = 0.01 for “+1%”
    latex = make_sensitivity_table_latex(
        title="Payload mass sensitivity to engine thrust variations. Values show change in payload per +1\\% thrust increase.",
        label="tab:thrust_sensitivity",
        met_first="FirstStageThrust",
        met_second="SecondStageThrust",
        x_first_vals=FirstStageThrust,
        x_second_vals=SecondStageThrust,
        per_step_first=0.01,    # +1% thrust
        per_step_second=0.01,   # +1% thrust
        step_desc_first="+1\\% First-stage Thrust",
        step_desc_second="+1\\% Second-stage Thrust",
    )
    print(latex)

def print_firstCx0_sensitivity_table():
    # x arrays are FRACTIONS relative to nominal → per +1% means 0.01
    latex = make_sensitivity_table_latex(
        title="Payload mass sensitivity to baseline drag coefficient $C_{x0}$. Values show change in payload per +25\\% $C_{x0}$ increase.",
        label="tab:cx0_sensitivity",
        met_first="FirstStageCx0",
        met_second="BoosterStageCx0",
        x_first_vals=FirstStageCx0,
        x_second_vals=BoosterStageCx0,
        per_step_first=0.1,    # +1% Cx0
        per_step_second=0.1,   # +1% Cx0
        step_desc_first="+1\\% First-stage $C_{x0}$",
        step_desc_second="+1\\% Booster $C_{x0}$",
    )
    print(latex)

def print_boosterCx0_sensitivity_table():
    # x arrays are FRACTIONS relative to nominal → per +1% means 0.01
    latex = make_sensitivity_table_latex(
        title="Payload mass sensitivity to baseline Booster drag coefficient $C_{x0}$. Values show change in payload per +25\\% $C_{x0}$ increase.",
        label="tab:cx0_sensitivity",
        met_first="BoosterStageCx0",
        met_second="BoosterStageCx0",
        x_first_vals=BoosterStageCx0,
        x_second_vals=BoosterStageCx0,
        per_step_first=0.1,    # +1% Cx0
        per_step_second=0.1,   # +1% Cx0
        step_desc_first="+1\\% First-stage $C_{x0}$",
        step_desc_second="+1\\% Booster $C_{x0}$",
    )
    print(latex)

def print_partition_sensitivity_table():
    # x arrays are FRACTIONS relative to nominal → per +1% means 0.01
    latex = make_sensitivity_table_latex(
        title="Payload mass sensitivity to Stage Partition. Values show change in payload per +1\\% increase in Stage 1 Propellant Fraction",
        label="tab:partition_sensitivity",
        met_first="StagePartition",
        met_second="StagePartition",
        x_first_vals=PartitionDelta,
        x_second_vals=PartitionDelta,
        per_step_first=0.1,    # +1% Cx0
        per_step_second=0.1,   # +1% Cx0
        step_desc_first="+1\\% First Stage Propellant Fraction",
        step_desc_second="+1\\% Booster $C_{x0}$",
    )
    print(latex)

def print_launch_alt_sensitivity_table(step_m=100.0):
    # x is in meters; default to report per +100 m (adjust step_m if you prefer +1 m or +1000 m)
    latex = make_sensitivity_table_latex(
        title=f"Payload mass sensitivity to launch-site altitude bias. Values show change in payload per +{int(step_m)}\\,m increase.",
        label="tab:launch_alt_sensitivity",
        met_first="LaunchAltDelta",
        met_second="LaunchAltDelta",
        x_first_vals=LaunchAltDelta/1000,
        x_second_vals=LaunchAltDelta/1000,
        per_step_first=step_m,
        per_step_second=step_m,
        step_desc_first=f"+{int(step_m)}\\,m",
        step_desc_second=f"+{int(step_m)}\\,m",
    )
    print(latex)

def print_empty_mass_sensitivity_table(step_kg=1000.0):
    # x is in kilograms; default to report per +1000 kg
    latex = make_sensitivity_table_latex(
        title=f"Payload mass sensitivity to empty mass. Values show change in payload per +{int(step_kg)}\\,kg increase.",
        label="tab:empty_mass_sensitivity",
        met_first="FirstStage_EmptyMass",
        met_second="SecondStage_EmptyMass",
        x_first_vals=FirstStage_EmptyMass/1000,
        x_second_vals=SecondStage_EmptyMass/1000 if 'SecondStage_EmptyMass' in globals() else np.linspace(-3,3,7),
        per_step_first=step_kg,
        per_step_second=step_kg,
        step_desc_first=f"+{int(step_kg)}\\,kg (First)",
        step_desc_second=f"+{int(step_kg)}\\,kg (Second)",
    )
    print(latex)

def print_prop_mass_sensitivity_table(step_kg=1000.0):
    # x is in kilograms; default to report per +1000 kg
    latex = make_sensitivity_table_latex(
        title=f"Payload mass sensitivity to propellant mass. Values show change in payload per +{int(step_kg)}\\,kg increase.",
        label="tab:prop_mass_sensitivity",
        met_first="FirstStage_PropMass",
        met_second="SecondStage_PropMass",
        x_first_vals=FirstStagePropMass/1000,
        x_second_vals=SecondStagePropMass/1000 ,
        per_step_first=step_kg,
        per_step_second=step_kg,
        step_desc_first=f"+{int(step_kg)}\\,kg (First)",
        step_desc_second=f"+{int(step_kg)}\\,kg (Second)",
    )
    print(latex)

FirstStageCx0 = np.linspace(0.5,2.0, 7)-1
BoosterStageCx0 = np.linspace(0.5,2.0, 7)-1
FirstStageThrust = np.linspace(0.9,1.15, 11)-1
SecondStageThrust = np.linspace(0.85,1.15, 13)-1
LaunchAltDelta = np.linspace(0.0, 5000.0, 6)
FirstStage_EmptyMass = np.linspace(-4000,4000, 9)
SecondStage_EmptyMass = np.linspace(-3000,3000, 7)

#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_FirstStageCx0=0.001*np.array([24869.661, 24611.184, 24258.750, 23822.538, 23310.934, 22733.278, 22100.089])
Falcon9_LEO_ASDS_FirstStageCx0=0.001*np.array([21610.269, 21408.823, 21136.592, 20800.938, 20413.138, 19969.544, 19400.982])
Falcon9_LEO_RTLS_FirstStageCx0=0.001*np.array([17402.749, 17345.946, 17193.287, 17014.823, 16785.642, 16561.980, 16307.139])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_FirstStageCx0=0.001*np.array([9430.849, 9308.242, 9142.798, 8939.272, 8702.019, 8435.094, 8142.344])
Falcon9_MEO_ASDS_FirstStageCx0=0.001*np.array([7765.500, 7669.977, 7540.683, 7380.789, 7195.239, 6986.200, 6760.777])
Falcon9_MEO_RTLS_FirstStageCx0=0.001*np.array([5629.343, 5600.909, 5545.728, 5464.150, 5395.264, 5267.899, 5242.559])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_FirstStageCx0=0.001*np.array([7013.478, 6913.065, 6776.344, 6609.634, 6415.419, 6197.619, 5959.490])
Falcon9_TLI_ASDS_FirstStageCx0=0.001*np.array([5630.601, 5553.156, 5447.867, 5318.499, 5167.239, 4998.048, 4815.148])
Falcon9_TLI_RTLS_FirstStageCx0=0.001*np.array([3859.113, 3827.216, 3798.262, 3710.729, 3651.231, 3562.339, 3488.573])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_FirstStageCx0=0.001*np.array([263420.100, 262219.875, 260550.507, 258423.453, 255854.213, 252903.159, 249535.210])
Starship_LEO_ASDS_FirstStageCx0=0.001*np.array([232900.913, 231863.186, 230427.330, 228611.237, 226422.039, 223850.690, 220943.777])
Starship_LEO_RTLS_FirstStageCx0=0.001*np.array([204137.141, 204158.918, 202938.014, 201666.807, 200702.994, 198985.218, 196996.594])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_FirstStageCx0=0.001*np.array([87165.985, 86533.756, 85652.152, 84523.358, 83183.647, 81639.541, 79890.852])
Starship_MEO_ASDS_FirstStageCx0=0.001*np.array([70077.801, 69545.745, 68813.294, 67884.849, 66761.431, 65452.677, 63982.118])
Starship_MEO_RTLS_FirstStageCx0=0.001*np.array([53319.493, 53013.644, 52582.983, 52024.259, 51365.551, 50586.747, 49710.955])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_FirstStageCx0=0.001*np.array([58638.249, 58118.421, 57402.658, 56492.487, 55396.922, 54138.144, 52722.322])
Starship_TLI_ASDS_FirstStageCx0=0.001*np.array([44721.573, 44291.501, 43696.591, 42934.723, 42009.467, 40938.088, 39744.227])
Starship_TLI_RTLS_FirstStageCx0=0.001*np.array([30919.465, 30455.006, 30059.861, 29821.172, 29129.482, 28486.500, 27781.053])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_BoosterStageCx0=0.001*np.array([24258.750, 24258.750, 24258.750, 24258.750, 24258.750, 24258.750, 24258.750])
Falcon9_LEO_ASDS_BoosterStageCx0=0.001*np.array([19750.383, 20698.129, 21136.592, 21380.004, 21570.866, 21730.359, 21867.832])
Falcon9_LEO_RTLS_BoosterStageCx0=0.001*np.array([15050.693, 16246.890, 17190.069, 17857.520, 18052.986, 18151.236, 18215.672])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_BoosterStageCx0=0.001*np.array([9142.798, 9142.798, 9142.798, 9142.798, 9142.798, 9142.798, 9142.798])
Falcon9_MEO_ASDS_BoosterStageCx0=0.001*np.array([7144.798, 7342.485, 7540.683, 7654.132, 7744.752, 7821.178, 7887.592])
Falcon9_MEO_RTLS_BoosterStageCx0=0.001*np.array([4766.272, 5147.802, 5545.728, 5861.023, 5949.001, 5998.062, 6040.098])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_BoosterStageCx0=0.001*np.array([6776.344, 6776.344, 6776.344, 6776.344, 6776.344, 6776.344, 6776.344])
Falcon9_TLI_ASDS_BoosterStageCx0=0.001*np.array([4950.395, 5292.539, 5447.867, 5540.673, 5614.989, 5677.756, 5732.369])
Falcon9_TLI_RTLS_BoosterStageCx0=0.001*np.array([2869.216, 3460.220, 3798.262, 4067.740, 4130.719, 4157.941, 4183.516])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_BoosterStageCx0=0.001*np.array([260550.507, 260550.507, 260550.507, 260550.507, 260550.507, 260550.507, 260550.507])
Starship_LEO_ASDS_BoosterStageCx0=0.001*np.array([215269.872, 220310.440, 225645.063, 233951.190, 239757.091, 244348.076, 247687.142])
Starship_LEO_RTLS_BoosterStageCx0=0.001*np.array([189395.754, 195663.576, 201212.211, 206122.919, 211033.501, 213597.998, 217408.369])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_BoosterStageCx0=0.001*np.array([85652.152, 85652.152, 85652.152, 85652.152, 85652.152, 85652.152, 85652.152])
Starship_MEO_ASDS_BoosterStageCx0=0.001*np.array([60864.419, 63567.906, 65918.339, 68642.385, 73675.293, 75775.514, 76865.141])
Starship_MEO_RTLS_BoosterStageCx0=0.001*np.array([45757.166, 48539.027, 51430.578, 54253.993, 56648.968, 58585.026, 60504.241])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_BoosterStageCx0=0.001*np.array([57402.658, 57402.658, 57402.658, 57402.658, 57402.658, 57402.658, 57402.658])
Starship_TLI_ASDS_BoosterStageCx0=0.001*np.array([37918.583, 39343.814, 43389.644, 44964.936, 45220.812, 49592.037, 50366.953])
Starship_TLI_RTLS_BoosterStageCx0=0.001*np.array([24526.311, 27028.137, 29166.546, 30475.459, 33268.813, 34963.392, 36659.606])

#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_FirstStageThrust=0.001*np.array([18814.438, 20225.768, 21595.958, 22936.865, 24258.750, 25571.199, 26880.221, 28192.014, 29507.955, 30835.236, 32174.092])
Falcon9_LEO_ASDS_FirstStageThrust=0.001*np.array([16882.505, 17976.269, 18973.436, 20158.490, 21136.592, 22121.602, 23105.928, 24091.229, 25075.979, 26061.474, 27049.286])
Falcon9_LEO_RTLS_FirstStageThrust=0.001*np.array([14440.183, 15221.771, 15902.068, 16537.745, 17169.086, 17764.384, 18338.169, 18896.177, 19435.878, 19958.861, 20468.846])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_FirstStageThrust=0.001*np.array([6716.543, 7344.859, 7955.170, 8552.931, 9142.798, 9729.280, 10316.422, 10905.589, 11498.272, 12096.624, 12703.215])
Falcon9_MEO_ASDS_FirstStageThrust=0.001*np.array([5729.932, 6200.233, 6648.538, 7096.439, 7540.683, 7980.984, 8416.824, 8850.163, 9279.526, 9707.504, 10136.612])
Falcon9_MEO_RTLS_FirstStageThrust=0.001*np.array([4429.998, 4747.614, 4988.701, 5269.576, 5545.728, 5785.097, 6024.188, 6245.857, 6460.831, 6644.882, 6872.045])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_FirstStageThrust=0.001*np.array([4816.177, 5323.479, 5815.853, 6298.521, 6776.344, 7252.985, 7728.878, 8207.767, 8689.983, 9124.983, 9596.144])
Falcon9_TLI_ASDS_FirstStageThrust=0.001*np.array([3966.342, 4327.788, 4669.054, 5092.298, 5447.867, 5799.583, 6148.104, 6495.215, 6840.665, 7184.412, 7528.114])
Falcon9_TLI_RTLS_FirstStageThrust=0.001*np.array([2959.425, 3149.119, 3376.412, 3578.050, 3798.262, 3988.030, 4154.754, 4337.915, 4514.602, 4677.483, 4837.692])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_SecondStageThrust=0.001*np.array([23626.933, 23759.118, 23879.108, 23988.011, 24087.143, 24176.954, 24258.750, 24333.208, 24400.565, 24461.774, 24517.332, 24567.624, 24612.850])
Falcon9_LEO_ASDS_SecondStageThrust=0.001*np.array([20509.100, 20638.461, 20756.855, 20864.696, 20963.534, 21054.021, 21136.592, 21212.427, 21281.711, 21344.548, 21400.821, 21451.943, 21494.930])
Falcon9_LEO_RTLS_SecondStageThrust=0.001*np.array([16670.653, 16779.920, 16860.012, 16949.518, 17051.943, 17124.356, 17193.287, 17248.417, 17319.501, 17366.946, 17390.057, 17329.815, 17459.688])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_SecondStageThrust=0.001*np.array([8983.720, 9018.643, 9049.428, 9076.844, 9101.379, 9123.198, 9142.798, 9159.788, 9174.501, 9186.989, 9197.745, 9206.364, 9213.445])
Falcon9_MEO_ASDS_SecondStageThrust=0.001*np.array([7394.004, 7426.645, 7455.595, 7481.066, 7503.613, 7523.377, 7540.683, 7555.937, 7568.827, 7580.107, 7589.722, 7597.535, 7604.070])
Falcon9_MEO_RTLS_SecondStageThrust=0.001*np.array([5360.439, 5416.488, 5472.959, 5495.565, 5514.780, 5531.397, 5545.728, 5557.760, 5547.018, 5577.110, 5552.933, 5597.921, 5602.899])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_SecondStageThrust=0.001*np.array([6666.176, 6690.985, 6712.876, 6732.124, 6748.933, 6763.604, 6776.344, 6787.323, 6796.652, 6804.483, 6810.951, 6816.161, 6819.839])
Falcon9_TLI_ASDS_SecondStageThrust=0.001*np.array([5334.459, 5359.319, 5381.461, 5401.217, 5418.684, 5434.218, 5447.867, 5459.429, 5469.218, 5477.433, 5484.042, 5489.386, 5493.459])
Falcon9_TLI_RTLS_SecondStageThrust=0.001*np.array([3688.210, 3712.159, 3724.126, 3756.866, 3772.673, 3780.301, 3798.262, 3808.413, 3810.904, 3818.214, 3824.231, 3829.037, 3815.780])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_FirstStageThrust=0.001*np.array([198916.470, 215537.609, 231218.864, 246173.228, 260550.507, 274642.204, 288497.279, 302171.652, 315693.712, 329084.170, 342418.886])
Starship_LEO_ASDS_FirstStageThrust=0.001*np.array([177641.921, 189047.455, 201952.389, 215131.067, 225645.063, 238723.740, 249845.695, 260720.583, 271412.442, 281873.252, 292165.806])
Starship_LEO_RTLS_FirstStageThrust=0.001*np.array([161667.392, 173190.627, 183287.336, 193991.180, 203264.196, 212078.044, 220551.675, 228356.051, 236340.816, 244564.533, 251842.629])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_FirstStageThrust=0.001*np.array([55821.133, 63789.653, 71359.974, 78624.585, 85652.152, 92427.773, 99066.782, 105515.217, 111931.114, 118317.778, 124707.425])
Starship_MEO_ASDS_FirstStageThrust=0.001*np.array([46794.425, 51135.135, 55476.483, 60697.249, 65918.339, 73749.639, 78674.084, 83120.699, 88148.600, 93028.669, 97842.868])
Starship_MEO_RTLS_FirstStageThrust=0.001*np.array([33278.981, 38219.570, 43160.135, 47295.061, 51430.578, 55322.425, 59083.037, 62681.863, 66194.824, 69517.417, 72884.768])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_FirstStageThrust=0.001*np.array([33544.150, 39950.854, 46017.301, 51820.396, 57402.658, 62858.505, 68222.421, 73533.212, 78807.468, 84060.565, 89321.195])
Starship_TLI_ASDS_FirstStageThrust=0.001*np.array([22546.314, 27934.813, 34393.638, 38891.702, 43389.644, 46668.514, 50909.551, 54973.748, 58917.748, 62862.995, 66709.592])
Starship_TLI_RTLS_FirstStageThrust=0.001*np.array([15288.632, 19006.746, 22556.973, 26038.372, 29166.546, 32497.137, 35281.543, 38121.360, 41711.598, 43530.331, 45844.952])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_SecondStageThrust=0.001*np.array([242111.738, 245855.678, 249296.859, 252463.652, 255380.878, 258070.789, 260550.507, 262836.806, 264946.648, 266894.792, 268240.822, 270138.811, 271565.669])
Starship_LEO_ASDS_SecondStageThrust=0.001*np.array([208534.633, 212372.898, 215897.133, 219137.422, 221174.536, 223211.592, 225645.063, 229326.646, 231175.084, 233024.747, 234638.114, 236016.736, 235972.201])
Starship_LEO_RTLS_SecondStageThrust=0.001*np.array([184095.441, 188291.485, 191078.023, 194250.238, 196787.440, 199182.306, 201212.211, 203141.308, 204846.629, 206412.121, 207737.622, 208929.995, 209971.312])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_SecondStageThrust=0.001*np.array([79450.679, 80786.340, 82002.561, 83105.858, 84042.471, 84897.384, 85652.152, 86290.830, 86873.857, 87338.480, 87755.480, 88085.509, 88363.909])
Starship_MEO_ASDS_SecondStageThrust=0.001*np.array([61820.860, 63488.727, 63683.896, 64723.438, 66777.319, 66625.499, 65918.339, 67734.821, 68257.430, 68682.164, 68909.386, 68141.832, 69568.806])
Starship_MEO_RTLS_SecondStageThrust=0.001*np.array([43746.474, 47614.719, 48745.221, 49606.110, 50674.688, 50972.666, 51430.578, 51927.175, 52432.893, 53145.476, 53041.330, 53263.861, 53410.871])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_SecondStageThrust=0.001*np.array([52599.735, 53607.575, 54520.831, 55349.229, 56100.895, 56783.116, 57402.658, 57965.440, 58476.620, 58894.519, 59269.019, 59578.367, 59836.875])
Starship_TLI_ASDS_SecondStageThrust=0.001*np.array([38574.303, 39675.548, 40672.546, 41576.509, 42372.613, 43066.812, 43696.591, 44223.506, 44695.048, 45092.388, 45431.685, 45712.626, 45941.616])
Starship_TLI_RTLS_SecondStageThrust=0.001*np.array([25909.280, 26843.464, 27678.997, 28378.976, 29030.111, 29797.154, 30111.765, 30539.319, 30862.575, 31196.154, 31434.111, 31629.640, 31772.252])

#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_LaunchAltDelta=0.001*np.array([24258.750, 24753.286, 25219.441, 25659.614, 26061.153, 26400.054])
Falcon9_LEO_ASDS_LaunchAltDelta=0.001*np.array([21136.592, 21538.360, 21920.393, 22281.556, 22611.103, 22878.355])
Falcon9_LEO_RTLS_LaunchAltDelta=0.001*np.array([17193.287, 17402.887, 17644.424, 17816.207, 17976.413, 18090.633])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_LaunchAltDelta=0.001*np.array([9142.798, 9362.300, 9569.662, 9765.679, 9949.128, 10115.825])
Falcon9_MEO_ASDS_LaunchAltDelta=0.001*np.array([7540.683, 7723.221, 7895.496, 8058.427, 8210.094, 8343.987])
Falcon9_MEO_RTLS_LaunchAltDelta=0.001*np.array([5545.728, 5640.065, 5716.239, 5801.663, 5858.223, 5912.938])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_LaunchAltDelta=0.001*np.array([6776.344, 6955.131, 7123.934, 7283.426, 7432.854, 7568.979])
Falcon9_TLI_ASDS_LaunchAltDelta=0.001*np.array([5447.867, 5594.482, 5732.999, 5863.217, 5985.877, 6095.015])
Falcon9_TLI_RTLS_LaunchAltDelta=0.001*np.array([3798.262, 3867.297, 3943.196, 4003.586, 4035.104, 4077.830])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_LaunchAltDelta=0.001*np.array([260550.507, 266106.159, 271355.936, 276315.839, 280934.993, 285007.254])
Starship_LEO_ASDS_LaunchAltDelta=0.001*np.array([230427.330, 235325.093, 239926.877, 244179.249, 248155.925, 251659.120])
Starship_LEO_RTLS_LaunchAltDelta=0.001*np.array([196415.178, 206434.234, 209404.851, 212195.402, 214711.906, 216340.158])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_LaunchAltDelta=0.001*np.array([85652.152, 88414.842, 91004.974, 93454.021, 95779.240, 97926.635])
Starship_MEO_ASDS_LaunchAltDelta=0.001*np.array([68813.294, 71199.637, 73443.560, 75581.354, 77574.315, 79398.060])
Starship_MEO_RTLS_LaunchAltDelta=0.001*np.array([52582.983, 54069.281, 55779.426, 56733.715, 58409.755, 58922.797])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_LaunchAltDelta=0.001*np.array([57402.658, 59635.840, 61748.090, 63746.534, 65644.783, 67407.054])
Starship_TLI_ASDS_LaunchAltDelta=0.001*np.array([42514.774, 46241.229, 46900.929, 49420.110, 50994.535, 52569.793])
Starship_TLI_RTLS_LaunchAltDelta=0.001*np.array([30059.861, 31299.220, 32403.618, 33435.221, 34390.794, 35219.578])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_FirstStage_EmptyMass=0.001*np.array([24929.201, 24758.852, 24590.344, 24423.652, 24258.750, 24095.692, 23934.342, 23774.731, 23616.812])
Falcon9_LEO_ASDS_FirstStage_EmptyMass=0.001*np.array([22071.280, 21894.910, 21536.875, 21380.615, 21136.592, 20899.961, 20669.512, 20443.985, 20222.355])
Falcon9_LEO_RTLS_FirstStage_EmptyMass=0.001*np.array([18757.516, 18382.631, 18005.132, 17581.418, 17193.287, 16838.623, 16493.686, 16144.107, 15819.443])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_FirstStage_EmptyMass=0.001*np.array([9473.842, 9389.502, 9306.188, 9223.957, 9142.798, 9062.658, 8983.232, 8904.625, 8827.022])
Falcon9_MEO_ASDS_FirstStage_EmptyMass=0.001*np.array([7986.680, 7869.428, 7756.537, 7647.094, 7540.683, 7436.870, 7335.234, 7235.130, 7133.140])
Falcon9_MEO_RTLS_FirstStage_EmptyMass=0.001*np.array([6284.857, 6110.839, 5946.760, 5717.903, 5545.728, 5379.943, 5219.738, 5066.255, 4933.219])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_FirstStage_EmptyMass=0.001*np.array([7050.154, 6980.297, 6911.388, 6843.407, 6776.344, 6710.218, 6644.914, 6580.432, 6516.767])
Falcon9_TLI_ASDS_FirstStage_EmptyMass=0.001*np.array([5811.222, 5715.886, 5623.726, 5534.585, 5447.867, 5363.204, 5280.439, 5199.209, 5118.614])
Falcon9_TLI_RTLS_FirstStage_EmptyMass=0.001*np.array([4377.214, 4226.277, 4074.455, 3908.152, 3798.262, 3669.811, 3541.488, 3417.528, 3292.155])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_SecondStage_EmptyMass=0.001*np.array([27258.750, 26258.750, 25258.750, 24258.750, 23258.750, 22258.750, 21258.750])
Falcon9_LEO_ASDS_SecondStage_EmptyMass=0.001*np.array([24136.097, 23136.592, 22136.592, 21136.592, 20136.592, 19136.592, 18136.592])
Falcon9_LEO_RTLS_SecondStage_EmptyMass=0.001*np.array([20208.159, 19208.841, 18190.069, 17193.287, 16168.577, 15168.577, 14208.841])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_SecondStage_EmptyMass=0.001*np.array([12142.798, 11142.798, 10142.798, 9142.798, 8142.798, 7142.798, 6142.798])
Falcon9_MEO_ASDS_SecondStage_EmptyMass=0.001*np.array([10540.683, 9540.683, 8540.683, 7540.683, 6540.683, 5540.683, 4540.683])
Falcon9_MEO_RTLS_SecondStage_EmptyMass=0.001*np.array([8545.728, 7545.728, 6545.728, 5545.728, 4545.728, 3534.790, 2552.884])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_SecondStage_EmptyMass=0.001*np.array([9776.344, 8776.344, 7776.344, 6776.344, 5776.344, 4776.344, 3776.344])
Falcon9_TLI_ASDS_SecondStage_EmptyMass=0.001*np.array([8447.867, 7447.867, 6447.867, 5447.867, 4447.867, 3447.867, 2447.867])
Falcon9_TLI_RTLS_SecondStage_EmptyMass=0.001*np.array([6783.395, 5798.262, 4792.187, 3798.262, 2792.187, 1792.187, 792.187])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_FirstStage_EmptyMass=0.001*np.array([261111.813, 260971.324, 260830.943, 260690.671, 260550.507, 260410.452, 260270.504, 260130.665, 259990.931])
Starship_LEO_ASDS_FirstStage_EmptyMass=0.001*np.array([231848.834, 231492.100, 231137.371, 230782.583, 230427.330, 230069.669, 229714.816, 229350.210, 228977.947])
Starship_LEO_RTLS_FirstStage_EmptyMass=0.001*np.array([204664.658, 204322.791, 203977.067, 203619.690, 203264.196, 202910.571, 202553.029, 202187.821, 201813.091])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_FirstStage_EmptyMass=0.001*np.array([85953.386, 85878.155, 85802.873, 85727.539, 85652.152, 85576.715, 85501.307, 85425.969, 85350.702])
Starship_MEO_ASDS_FirstStage_EmptyMass=0.001*np.array([69524.729, 69355.424, 69178.846, 68995.576, 68813.294, 68631.704, 68450.145, 68267.085, 68084.604])
Starship_MEO_RTLS_FirstStage_EmptyMass=0.001*np.array([53343.075, 53162.814, 52975.428, 52782.322, 52582.983, 52376.551, 52178.609, 52019.611, 51832.528])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_FirstStage_EmptyMass=0.001*np.array([57645.718, 57584.868, 57524.074, 57463.337, 57402.658, 57342.035, 57281.469, 57220.960, 57160.508])
Starship_TLI_ASDS_FirstStage_EmptyMass=0.001*np.array([43102.947, 44802.855, 44672.220, 44540.794, 44410.965, 44282.647, 44152.394, 44020.190, 43887.095])
Starship_TLI_RTLS_FirstStage_EmptyMass=0.001*np.array([30714.944, 30526.135, 30421.432, 30269.113, 30111.765, 29948.956, 29780.203, 29623.309, 29423.093])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_SecondStage_EmptyMass=0.001*np.array([264550.507, 263550.507, 262550.507, 261550.507, 260550.507, 259550.507, 258550.507, 257550.507, 256550.507])
Starship_LEO_ASDS_SecondStage_EmptyMass=0.001*np.array([234427.330, 233427.330, 232427.330, 231427.330, 230427.330, 229427.330, 228427.330, 227427.330, 226427.330])
Starship_LEO_RTLS_SecondStage_EmptyMass=0.001*np.array([206938.014, 206264.196, 205264.196, 204264.196, 203264.196, 202264.196, 201264.196, 200264.196, 199264.196])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_SecondStage_EmptyMass =0.001*np.array([89652.152, 88652.152, 87652.152, 86652.152, 85652.152, 84652.152, 83652.152, 82652.152, 81652.152])
Starship_MEO_ASDS_SecondStage_EmptyMass=0.001*np.array([72348.228, 72748.437, 71748.437, 70748.437, 69748.437, 68748.437, 67748.437, 66748.437, 65748.437])
Starship_MEO_RTLS_SecondStage_EmptyMass=0.001*np.array([56582.983, 55582.983, 54582.983, 53582.983, 52582.983, 51582.983, 50582.983, 49582.983, 48582.983])

#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_SecondStage_EmptyMass=0.001*np.array([61402.658, 60402.658, 59402.658, 58402.658, 57402.658, 56402.658, 55402.658, 54402.658, 53402.658])
Starship_TLI_ASDS_SecondStage_EmptyMass=0.001*np.array([47696.591, 46696.591, 45696.591, 44696.591, 43696.591, 42696.591, 41696.591, 40696.591, 39696.591])
Starship_TLI_RTLS_SecondStage_EmptyMass=0.001*np.array([34111.765, 33111.765, 32111.765, 31111.765, 30111.765, 29111.765, 28111.765, 27111.765, 26111.765])

#=================================================================================================
#=================================================================================================
#============================================== ISP ==============================================
#=================================================================================================
#=================================================================================================
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_FirstStageIsp=0.001*np.array([18798.347, 19662.062, 20540.639, 21439.065, 22358.155, 23297.030, 24258.750, 25241.651, 26248.189, 27278.628, 28327.543, 29397.265, 30489.063])
Falcon9_LEO_ASDS_FirstStageIsp=0.001*np.array([16689.389, 17372.053, 18088.728, 18766.769, 19596.543, 20329.683, 21136.592, 21961.445, 22821.034, 23711.914, 24629.971, 25567.921, 26526.190])
Falcon9_LEO_RTLS_FirstStageIsp=0.001*np.array([13562.146, 14130.220, 14730.268, 15317.480, 15909.442, 16537.186, 17193.287, 17838.651, 18543.165, 19249.991, 19943.147, 20646.835, 21412.659])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_FirstStageIsp=0.001*np.array([6598.093, 6998.104, 7407.646, 7826.264, 8254.697, 8694.045, 9142.798, 9603.462, 10075.874, 10559.185, 11054.675, 11562.171, 12085.387])
Falcon9_MEO_ASDS_FirstStageIsp=0.001*np.array([5494.155, 5785.164, 6091.081, 6454.039, 6803.847, 7164.032, 7540.683, 7932.160, 8335.773, 8750.859, 9176.348, 9614.795, 10066.929])
Falcon9_MEO_RTLS_FirstStageIsp=0.001*np.array([3971.920, 4214.697, 4467.885, 4740.580, 4979.626, 5269.762, 5545.728, 5841.886, 6117.258, 6432.095, 6535.050, 7052.597, 7368.914])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_FirstStageIsp=0.001*np.array([4703.055, 5028.978, 5362.283, 5702.469, 6051.485, 6409.238, 6776.344, 7153.786, 7540.376, 7937.467, 8343.937, 8760.245, 9188.197])
Falcon9_TLI_ASDS_FirstStageIsp=0.001*np.array([3797.268, 4033.872, 4292.148, 4576.815, 4850.988, 5142.696, 5447.867, 5764.729, 6092.100, 6430.224, 6776.814, 7133.553, 7501.835])
Falcon9_TLI_RTLS_FirstStageIsp=0.001*np.array([2529.855, 2746.225, 2974.572, 3101.337, 3335.755, 3565.928, 3798.262, 4031.591, 4262.553, 4502.059, 4760.886, 5002.890, 5250.000])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_SecondStageIsp=0.001*np.array([17810.359, 18898.267, 19980.920, 21057.286, 22129.422, 23196.729, 24258.750, 25315.243, 26365.716, 27409.893, 28446.840, 29476.133, 30497.963])
Falcon9_LEO_ASDS_SecondStageIsp=0.001*np.array([14918.338, 15959.559, 17000.071, 18038.819, 19074.957, 20107.514, 21136.592, 22051.006, 23185.369, 24066.514, 25218.591, 26228.750, 27166.767])
Falcon9_LEO_RTLS_SecondStageIsp=0.001*np.array([11341.156, 12273.568, 13264.144, 14242.298, 15242.331, 16164.055, 17193.287, 18173.213, 19119.249, 20131.299, 21109.266, 22106.618, 23092.266])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_SecondStageIsp=0.001*np.array([5019.773, 5684.964, 6360.384, 7044.875, 7737.357, 8436.949, 9142.798, 9854.211, 10571.153, 11293.102, 12019.567, 12749.896, 13483.687])
Falcon9_MEO_ASDS_SecondStageIsp=0.001*np.array([3717.509, 4327.744, 4950.182, 5583.724, 6227.295, 6879.919, 7540.683, 8208.756, 8883.369, 9563.814, 10248.971, 10939.271, 11634.349])
Falcon9_MEO_RTLS_SecondStageIsp=0.001*np.array([2137.677, 2673.483, 3208.743, 3788.149, 4329.616, 4926.414, 5545.728, 6143.294, 6780.313, 7238.757, 8040.898, 8635.802, 9320.556])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_SecondStageIsp=0.001*np.array([3174.546, 3748.484, 4334.108, 4930.366, 5536.438, 6152.024, 6776.344, 7408.689, 8048.324, 8694.575, 9346.885, 10004.696, 10667.416])
Falcon9_TLI_ASDS_SecondStageIsp=0.001*np.array([2133.413, 2656.269, 3192.246, 3740.329, 4299.572, 4869.076, 5447.867, 6035.195, 6630.958, 7234.657, 7845.647, 8463.353, 9087.234])
Falcon9_TLI_RTLS_SecondStageIsp=0.001*np.array([867.664, 1322.151, 1772.391, 2274.505, 2742.089, 3272.940, 3798.262, 4295.544, 4861.558, 5370.852, 5926.664, 6521.487, 7072.020])

FirstStageIsp = 100*(np.linspace(0.85,1.15, 13)-1)
SecondStageIsp = 100*(np.linspace(0.85,1.15, 13)-1)

#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_FirstStageIsp=0.001*np.array([208385.165, 216709.517, 225179.527, 233814.117, 242895.577, 251643.242, 260550.507, 269580.615, 278771.442, 288136.964, 297635.5, 307297.481, 317086.937])
Starship_LEO_ASDS_FirstStageIsp=0.001*np.array([182938.298, 190630.434, 198465.472, 206311.349, 214281.189, 222346.022, 230427.330, 238675.909, 246890.691, 255226.089, 263698.709, 272320.940, 281066.807])
Starship_LEO_RTLS_FirstStageIsp=0.001*np.array([161678.974, 168656.508, 175622.780, 182701.472, 189584.363, 196416.898, 203264.196, 210137.670, 217284.424, 224489.969, 231701.695, 239055.760, 246430.341])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_FirstStageIsp=0.001*np.array([59322.872, 63468.370, 67694.863, 72006.541, 76406.347, 80901.017, 85499.046, 90182.507, 94961.183, 99984.644, 104871.905, 109871.837, 114967.118])
Starship_MEO_ASDS_FirstStageIsp=0.001*np.array([44738.675, 48615.924, 52537.841, 56526.314, 60562.462, 64680.330, 68813.294, 73005.811, 77283.303, 81591.150, 85979.379, 90410.464, 94786.607])
Starship_MEO_RTLS_FirstStageIsp=0.001*np.array([32819.613, 36107.923, 39454.985, 42732.542, 46007.079, 49255.147, 52582.983, 56059.029, 59930.985, 63089.105, 63518.045, 71257.683, 74078.994])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_FirstStageIsp=0.001*np.array([35522.241, 39045.021, 42632.961, 46244.680, 49924.476, 49924.476, 57402.658, 61242.838, 65170.649, 69182.755, 73275.409, 77453.040, 81710.045])
Starship_TLI_ASDS_FirstStageIsp=0.001*np.array([23781.762, 29033.922, 26161.668, 35007.959, 36087.284, 41225.905, 44410.965, 47650.909, 50530.361, 53994.082, 57516.771, 61148.455, 64827.467])
Starship_TLI_RTLS_FirstStageIsp=0.001*np.array([14129.700, 16806.008, 19494.304, 22167.699, 24769.216, 27395.002, 30111.765, 32907.127, 35761.574, 38571.148, 41459.642, 44394.668, 47361.967])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_SecondStageIsp=0.001*np.array([175370.808, 190121.233, 204674.749, 219040.843, 233102.126, 246933.905, 260550.507, 274060.678, 287446.373, 300702.600, 313828.916, 326817.799, 339663.247])
Starship_LEO_ASDS_SecondStageIsp=0.001*np.array([146638.383, 161079.198, 175318.800, 189400.552, 203306.349, 217013.631, 230427.330, 243664.014, 256630.887, 269494.993, 282226.704, 294831.078, 307308.732])
Starship_LEO_RTLS_SecondStageIsp=0.001*np.array([119417.483, 133697.475, 147847.580, 161664.683, 175865.825, 189642.444, 203264.196, 216727.940, 229383.718, 242892.198, 255601.319, 268213.196, 280722.016])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_SecondStageIsp=0.001*np.array([27892.494, 37302.194, 46827.763, 56435.658, 66123.520, 75875.562, 85652.152, 95420.509, 105210.589, 114967.245, 124777.188, 134631.096, 144520.660])
Starship_MEO_ASDS_SecondStageIsp=0.001*np.array([13910.606, 22753.613, 31760.923, 40893.986, 50116.235, 59427.828, 68813.294, 78243.864, 87674.756, 97137.326, 106591.639, 116008.022, 125440.881])
Starship_MEO_RTLS_SecondStageIsp=0.001*np.array([409.455, 8678.310, 17154.609, 25812.349, 34628.070, 43558.044, 52582.983, 61700.833, 70898.358, 80125.980, 89385.744, 98689.892, 107948.348])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_SecondStageIsp=0.001*np.array([7306.455, 15372.688, 23575.508, 31893.428, 40318.251, 48834.134, 57402.658, 66086.265, 74873.563, 83753.553, 92711.856, 101741.221, 110833.861])
Starship_TLI_ASDS_SecondStageIsp=0.001*np.array([-3856.204, 3715.995, 11458.001, 19339.885, 27351.837, 35476.387, 43696.596, 51965.664, 60296.335, 68696.481, 77190.064, 85767.426, 94419.902])
Starship_TLI_RTLS_SecondStageIsp=0.001*np.array([-14815.132, -7763.610, -505.018, 6931.900, 14626.075, 22252.891, 30111.765, 38083.486, 46114.065, 54224.823, 62410.159, 70605.741, 78893.588])

# ==============================
# Helpers to reduce repetition
# ==============================
def _g(name):
    """Safe globals getter (raises KeyError with a clear name if missing)."""
    if name not in globals():
        raise KeyError(f"Missing variable: {name}")
    return globals()[name]

def plot_metric_pair(fig_num, title_base, x_first, x_second, xlabels, ylabels, 
                     met_first, met_second, rel_plot=False, styles=('-s','-o','-^'), dash_styles=('--s','--o','--^')):
    """
    Generic 2x2 layout like ISP:
      (1,1) Falcon 9 absolute, (1,2) Starship absolute
      (2,1) Falcon 9 relative,  (2,2) Starship relative
    met_* are the suffix strings used in your variables, e.g. 'FirstStageThrust', 'SecondStageThrust'
    """
    orbits = ['LEO','MEO','TLI']
    missions = ['EXP','ASDS','RTLS']
    vehicles = ['Falcon9','Starship']
    colors = [color_vec[0], color_vec[1], color_vec[2]]

    if rel_plot:
        sub_j = 2
    else:
        sub_j = 1
    # Absolute panels
    # Falcon 9
    for oi, orbit in enumerate(orbits):
        for mi, mission in enumerate(missions):
            y1 = _g(f"Falcon9_{orbit}_{mission}_{met_first}")
            if met_second:
                y2 = _g(f"Falcon9_{orbit}_{mission}_{met_second}")
            plot_sens(x_first, y1, fig_num=fig_num, subplot=(sub_j,2,1),
                      xlabel=xlabels[0], ylabel=ylabels[0],
                      title=f"Falcon 9 Payload Capacity Sensitivity to {title_base}",
                      legend=f"{orbit}_{mission}", style=styles[oi], color=colors[mi])
            if met_second:
                plot_sens(x_second, y2, fig_num=fig_num, subplot=(sub_j,2,1),
                        style=dash_styles[oi], color=colors[mi])

    # Starship
    for oi, orbit in enumerate(orbits):
        for mi, mission in enumerate(missions):
            y1 = _g(f"Starship_{orbit}_{mission}_{met_first}")
            if met_second:
                if met_second == 'SecondStage_EmptyMass':
                    x_second = np.linspace(-4,4,9)
                y2 = _g(f"Starship_{orbit}_{mission}_{met_second}")
            if met_first == 'StagePartition':
                x_first = np.linspace(-0.05, 0.05, 11)*100+68.4
            plot_sens(x_first, y1, fig_num=fig_num, subplot=(sub_j,2,2),
                      xlabel=xlabels[0], ylabel=ylabels[0],
                      title=f"Starship Payload Capacity Sensitivity to {title_base}",
                      legend=f"{orbit}_{mission}", style=styles[oi], color=colors[mi])
            if met_second:
                plot_sens(x_second, y2, fig_num=fig_num, subplot=(sub_j,2,2),
                      style=dash_styles[oi], color=colors[mi])
    if not rel_plot:
        return
    # Relative panels
    # Falcon 9
    for oi, orbit in enumerate(orbits):
        for mi, mission in enumerate(missions):
            y1 = _g(f"Falcon9_{orbit}_{mission}_{met_first}")
            if met_second:
                y2 = _g(f"Falcon9_{orbit}_{mission}_{met_second}")
            plot_sens(x_first, y1, fig_num=fig_num, subplot=(2,2,3),
                      xlabel=xlabels[1], ylabel=ylabels[1],
                      title=f"Falcon 9 Relative Payload Capacity Sensitivity to {title_base}",
                      legend=f"{orbit}_{mission}", style=styles[oi], rel_plot=True, color=colors[mi])
            if met_second:
                plot_sens(x_second, y2, fig_num=fig_num, subplot=(2,2,3),
                      style=dash_styles[oi], rel_plot=True, color=colors[mi])

    # Starship
    for oi, orbit in enumerate(orbits):
        for mi, mission in enumerate(missions):
            y1 = _g(f"Starship_{orbit}_{mission}_{met_first}")
            if met_second:
                y2 = _g(f"Starship_{orbit}_{mission}_{met_second}")
            plot_sens(x_first, y1, fig_num=fig_num, subplot=(2,2,4),
                      xlabel=xlabels[1], ylabel=ylabels[1],
                      title=f"Starship Relative Payload Capacity Sensitivity to {title_base}",
                      legend=f"{orbit}_{mission}", style=styles[oi], rel_plot=True, color=colors[mi])
            if met_second:
                plot_sens(x_second, y2, fig_num=fig_num, subplot=(2,2,4),
                      style=dash_styles[oi], rel_plot=True, color=colors[mi])

# ==============================
# X-axes (match your data lengths)
# ==============================
# Thrust:  First stage has 11 points (0.90..1.15), second has 13 points (0.85..1.15)
FirstStageIspPct  = 100*(np.linspace(0.85, 1.15, 11) - 1.0)
SecondStageThrustPct = 100*(np.linspace(0.85, 1.15, 13) - 1.0)

# ==============================
# X-axes (match your data lengths)
# ==============================
# Thrust:  First stage has 11 points (0.90..1.15), second has 13 points (0.85..1.15)
FirstStageThrustPct  = 100*(np.linspace(0.90, 1.15, 11) - 1.0)
SecondStageThrustPct = 100*(np.linspace(0.85, 1.15, 13) - 1.0)

# Aerodynamic Cx0s: 7 points (0.5..2.0) → percent deltas
Cx0Pct = 100*(np.linspace(0.5, 2.0, 7) - 1.0)

# Launch altitude bias: 6 absolute points in meters (already defined in your file as 0..5000)
LaunchAltDelta_m = LaunchAltDelta/1000  # reuse existing 6-point array

# Empty mass: first stage has 9 points, second stage 7 points (kg deltas)
FirstStageEmptyMass_kg  = FirstStage_EmptyMass
SecondStageEmptyMass_kg = np.linspace(-3000, 3000, 7)  # matches your *_SecondStage_EmptyMass arrays

# ==============================
# 1) ENGINE ISP (Fig 1)
# ==============================
if plot_isp:
    plot_metric_pair(
        fig_num=1,
        title_base="Engine Isp",
        x_first=FirstStageIsp,
        x_second=SecondStageIsp,
        xlabels=("First/Second Stage ISP Change [%]", "First/Second Stage ISP Change [%]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="FirstStageIsp",
        met_second="SecondStageIsp",
    )

    plt.savefig("ISP_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_isp_sensitivity_table()

# ==============================
# 1) ENGINE THRUST (Fig 2)
# ==============================
if plot_thrust:
    plot_metric_pair(
        fig_num=2,
        title_base="Engine Thrust",
        x_first=FirstStageThrustPct,
        x_second=SecondStageThrustPct,
        xlabels=("First/Second Stage Thrust Change [%]", "First/Second Stage Thrust Change [%]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="FirstStageThrust",
        met_second="SecondStageThrust",
    )

    plt.savefig("Thrust_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_thrust_sensitivity_table()


# ==============================
# 2) LAUNCH ALTITUDE DELTA (Fig 3)
# Uses one metric per stage name? Here both stages respond to the same x,
# and your arrays are named “…_LaunchAltDelta”. We'll pass it as both "first" and "second".
# ==============================
if plot_alt:
    plot_metric_pair(
        fig_num=3,
        title_base="Launch Site Altitude",
        x_first=LaunchAltDelta_m,
        x_second=LaunchAltDelta_m,
        xlabels=("Launch Altitude [Km]", "Launch Altitude [Km]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="LaunchAltDelta",
        met_second=None,
    )
    plt.savefig("LaunchAlt_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_launch_alt_sensitivity_table(step_m=1)

# ==============================
# 3) AERODYNAMIC DRAG BASELINE Cx0 (Fig 4)
# First-stage Cx0 and Booster-stage Cx0 share the same 7-point x-axis.
# ==============================
if plot_firstCx0:
    plot_metric_pair(
        fig_num=4,
        title_base="First Stage Cx₀",
        x_first=Cx0Pct,
        x_second=Cx0Pct,
        xlabels=("First Stage Cx₀ Change [%]", "First Stage Cx₀ Change [%]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="FirstStageCx0",
        met_second=None,
    )
    plt.savefig("FirstCx0_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_firstCx0_sensitivity_table()

# ==============================
# 4) EMPTY MASS (Fig 5)
# First-stage (9 points) and second-stage (7 points) are separate x’s.
# ==============================
if plot_mass:
    plot_metric_pair(
        fig_num=5,
        title_base="Inert Mass",
        x_first=FirstStageEmptyMass_kg/1000,
        x_second=SecondStageEmptyMass_kg/1000,
        xlabels=("ΔEmpty Mass [Tons]", "ΔEmpty Mass [Tons]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="FirstStage_EmptyMass",
        met_second="SecondStage_EmptyMass",
    )
    plt.savefig("InertMass_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_empty_mass_sensitivity_table(step_kg=1)


# ==============================
# 3) AERODYNAMIC DRAG BASELINE Cx0 (Fig 4)
# First-stage Cx0 and Booster-stage Cx0 share the same 7-point x-axis.
# ==============================
if plot_boosterCx0:
    plot_metric_pair(
        fig_num=6,
        title_base="Booster Cx₀",
        x_first=Cx0Pct,
        x_second=Cx0Pct,
        xlabels=("Booster Cx₀ Change [%]", "Booster Cx₀ Change [%]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="BoosterStageCx0",
        met_second=None,
    )
    plt.savefig("BoosterCx0_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_boosterCx0_sensitivity_table()

# ==============================
# 3) AERODYNAMIC DRAG BASELINE Cx0 (Fig 4)
# First-stage Cx0 and Booster-stage Cx0 share the same 7-point x-axis.
# ==============================
PartitionDelta = np.linspace(-0.05, 0.05, 11)*100
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_StagePartition=0.001*np.array([23080.171, 23430.377, 23724.025, 23960.879, 24138.985, 24258.750, 24318.474, 24315.945, 24247.217, 24109.908, 23897.432])
Falcon9_LEO_ASDS_StagePartition=0.001*np.array([20938.188, 21094.001, 21203.756, 21217.413, 21157.157, 21136.592, 20995.695, 20807.902, 20566.349, 20264.219, 19894.569])
Falcon9_LEO_RTLS_StagePartition=0.001*np.array([18496.111, 18475.492, 18374.624, 18145.143, 17734.340, 17147.467, 16645.356, 16121.933, 15530.117, 14869.859, 14143.383])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_StagePartition=0.001*np.array([8396.271, 8600.630, 8776.902, 8926.552, 9048.937, 9142.798, 9208.266, 9244.308, 9248.440, 9217.217, 9150.629])
Falcon9_MEO_ASDS_StagePartition=0.001*np.array([7198.612, 7316.633, 7389.194, 7484.184, 7522.909, 7540.683, 7535.397, 7504.958, 7447.536, 7358.558, 7216.335])
Falcon9_MEO_RTLS_StagePartition=0.001*np.array([5873.945, 5881.320, 5844.143, 5756.961, 5654.746, 5534.789, 5387.616, 5223.455, 5023.699, 4800.247, 4554.397])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_StagePartition=0.001*np.array([6034.824, 6228.184, 6399.734, 6549.039, 6674.256, 6776.344, 6854.311, 6906.244, 6929.929, 6923.287, 6891.604])
Falcon9_TLI_ASDS_StagePartition=0.001*np.array([5035.012, 5157.461, 5264.671, 5346.347, 5405.034, 5447.867, 5472.130, 5475.978, 5456.826, 5409.972, 5314.114])
Falcon9_TLI_RTLS_StagePartition=0.001*np.array([3909.691, 3943.646, 3941.544, 3899.937, 3846.999, 3783.395, 3696.802, 3597.414, 3469.245, 3321.763, 3155.878])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_StagePartition=0.001*np.array([229669.375, 236689.960, 243281.488, 249450.533, 255217.818, 260550.507, 265452.883, 269964.610, 274072.192, 277773.840, 281015.563])
Starship_LEO_ASDS_StagePartition=0.001*np.array([208451.891, 213889.991, 218805.416, 223183.395, 227051.741, 230427.330, 233282.806, 235632.861, 237483.513, 238866.586, 239774.165])
Starship_LEO_RTLS_StagePartition=0.001*np.array([190987.92, 195158.43, 198441.26, 201140.06, 203104.21, 204447.55, 205183.74, 205215.94, 204446.32, 203018.61, 200649.65])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_StagePartition=0.001*np.array([66581.37, 70838.57, 74873.62, 78685.69, 82277.19, 85652.12, 88794.14, 91718.09, 94433.10, 96936.82, 99224.51])
Starship_MEO_ASDS_StagePartition=0.001*np.array([56147.87, 59481.35, 62487.44, 65302.85, 67835.91, 68987.34, 71103.38, 72959.54, 74581.67, 75981.52, 77132.29])
Starship_MEO_RTLS_StagePartition=0.001*np.array([44055.99, 46635.71, 48825.53, 50727.89, 52300.51, 53540.26, 54440.04, 55063.66, 55349.42, 55176.79, 54600.60])
                                                                                                #    53540.26, 54440.04
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_StagePartition=0.001*np.array([40123.30, 43931.75, 47564.93, 51021.83, 54305.25, 57402.62, 60321.83, 63066.45, 65640.78, 68039.11, 70247.94])
Starship_TLI_ASDS_StagePartition=0.001*np.array([30434.09, 33530.21, 36412.07, 39083.39, 41551.04, 43824.30, 45906.75, 47782.98, 49471.64, 50973.91, 52285.89])
Starship_TLI_RTLS_StagePartition=0.001*np.array([21455.48, 23860.67, 26003.10, 27906.43, 29522.55, 30910.76, 32039.21, 32911.35, 33550.81, 33782.04, 32275.50])


if plot_partition:
    plot_metric_pair(
        fig_num=7,
        title_base="Stage Partition",
        x_first=np.linspace(-0.05, 0.05, 11)*100+81,
        x_second=np.linspace(-0.05, 0.05, 11)*100+68.4,
        xlabels=("Stage 1 Propellent Fraction [%]", "Stage 1 Propellent Fraction [%]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="StagePartition",
        met_second=None,
    )
    plt.savefig("StagePartition_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_partition_sensitivity_table()

# ==============================
# 4) EMPTY MASS (Fig 8)
# First-stage (9 points) and second-stage (7 points) are separate x’s.
# ==============================

#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_FirstStage_PropMass=0.001*np.array([23814.22, 23904.46, 23993.55, 24082.33, 24170.66, 24258.61, 24346.20, 24433.35, 24520.13, 24606.28, 24691.19])
Falcon9_LEO_ASDS_FirstStage_PropMass=0.001*np.array([20797.03, 20868.55, 20940.08, 21011.67, 21083.20, 21154.69, 21225.70, 21296.26, 21366.71, 21437.15, 21507.44])
Falcon9_LEO_RTLS_FirstStage_PropMass=0.001*np.array([16995.65, 17011.02, 17001.58, 17032.16, 17066.78, 17151.41, 17193.90, 17236.18, 17278.22, 17320.06, 17361.59])
#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_FirstStage_PropMass=0.001*np.array([8935.38, 8977.19, 9018.77, 9060.19, 9101.45, 9142.60, 9183.43, 9223.77, 9263.95, 9303.99, 9343.89])
Falcon9_MEO_ASDS_FirstStage_PropMass=0.001*np.array([7376.62, 7411.15, 7445.61, 7479.43, 7514.26, 7548.51, 7582.72, 7616.85, 7650.63, 7684.15, 7717.55])
Falcon9_MEO_RTLS_FirstStage_PropMass=0.001*np.array([5444.38, 5462.93, 5481.37, 5509.83, 5528.12, 5546.08, 5574.26, 5571.36, 5588.92, 5617.00, 5634.23])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_FirstStage_PropMass=0.001*np.array([6607.21, 6641.23, 6675.13, 6708.90, 6742.57, 6776.16, 6809.63, 6842.98, 6876.17, 6908.96, 6941.47])
Falcon9_TLI_ASDS_FirstStage_PropMass=0.001*np.array([5314.80, 5343.03, 5371.05, 5398.81, 5426.52, 5454.20, 5481.79, 5509.31, 5536.68, 5563.99, 5591.19])
Falcon9_TLI_RTLS_FirstStage_PropMass=0.001*np.array([3727.93, 3742.69, 3757.36, 3771.95, 3769.98, 3784.37, 3807.49, 3821.69, 3843.31, 3849.76, 3877.85])
#======================================== Orbit Name: LEO ========================================
Falcon9_LEO_EXP_SecondStage_PropMass=0.001*np.array([23558.10, 23717.73, 23868.01, 24008.92, 24138.37, 24258.61, 24369.98, 24472.30, 24563.85, 24646.27, 24720.28])
Falcon9_LEO_ASDS_SecondStage_PropMass=0.001*np.array([20169.09, 20169.09, 20588.94, 20785.51, 20974.04, 21154.69, 21326.49, 21491.12, 21648.34, 21795.97, 21924.89])
Falcon9_LEO_RTLS_SecondStage_PropMass=0.001*np.array([15830.93, 16132.92, 16424.31, 16704.78, 16975.11, 17214.95, 17432.11, 17696.99, 17947.89, 18187.56, 18414.94])

Falcon9_LEO_EXP_SecondStage_PropMass=0.001*np.array([23558.10, 23717.73, 23868.01, 24008.92, 24138.37, 24258.61, 24369.98, 24472.30, 24563.85, 24646.27, 24720.28])
Falcon9_LEO_ASDS_SecondStage_PropMass=0.001*np.array([20169.09, 20169.09, 20588.94, 20785.51, 20974.04, 21154.69, 21326.49, 21491.12, 21648.34, 21795.97, 21924.89])
Falcon9_LEO_RTLS_SecondStage_PropMass=0.001*np.array([15830.93, 16132.92, 16424.31, 16704.78, 16975.11, 17214.95, 17432.11, 17696.99, 17947.89, 18187.56, 18414.94])

#======================================== Orbit Name: MEO ========================================
Falcon9_MEO_EXP_SecondStage_PropMass=0.001*np.array([8668.58, 8772.41, 8871.19, 8965.79, 9056.36, 9142.59, 9224.44, 9301.98, 9375.49, 9445.13, 9510.26])
Falcon9_MEO_ASDS_SecondStage_PropMass=0.001*np.array([6978.10, 7100.40, 7218.51, 7332.47, 7442.33, 7548.51, 7651.03, 7749.19, 7843.20, 7933.64, 8020.38])
Falcon9_MEO_RTLS_SecondStage_PropMass=0.001*np.array([4821.57, 4977.99, 5130.34, 5278.56, 5423.56, 5564.10, 5692.91, 5825.64, 5954.01, 6075.98, 6196.30])
#======================================== Orbit Name: TLI ========================================
Falcon9_TLI_EXP_SecondStage_PropMass=0.001*np.array([6359.00, 6450.21, 6537.70, 6620.79, 6700.20, 6776.16, 6848.74, 6917.72, 6982.93, 7044.77, 7103.31])
Falcon9_TLI_ASDS_SecondStage_PropMass=0.001*np.array([4968.55, 5072.33, 5172.89, 5270.06, 5363.60, 5454.20, 5541.67, 5626.16, 5707.73, 5785.86, 5861.13])
Falcon9_TLI_RTLS_SecondStage_PropMass=0.001*np.array([3191.71, 3314.29, 3433.79, 3556.71, 3676.42, 3793.15, 3897.92, 4001.97, 4108.91, 4212.37, 4312.68])
#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_FirstStage_PropMass=0.001*np.array([259975.69, 260090.82, 260205.87, 260320.83, 260435.70, 260550.49, 260665.19, 260779.80, 260894.33, 261008.77, 261123.12])
Starship_LEO_ASDS_FirstStage_PropMass=0.001*np.array([230290.65, 230391.51, 230492.47, 230593.41, 230694.32, 230793.92, 230894.50, 230995.06, 231095.59, 231196.10, 231296.58])
Starship_LEO_RTLS_FirstStage_PropMass=0.001*np.array([204041.69, 204114.84, 204187.91, 204260.93, 204333.90, 204406.85, 204479.78, 204541.13, 204625.52, 204698.33, 204758.37])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_FirstStage_PropMass=0.001*np.array([85347.94, 85409.02, 85470.01, 85530.86, 85591.56, 85652.11, 85712.53, 85772.87, 85833.18, 85893.45, 85953.69])
Starship_MEO_ASDS_FirstStage_PropMass=0.001*np.array([69803.09, 69872.10, 69920.31, 69968.51, 68906.98, 68987.34, 69010.47, 69091.29, 69143.23, 69195.14, 69247.04])
Starship_MEO_ASDS_FirstStage_PropMass=0.001*np.array([68671.36, 66966.12, 67018.16, 67070.29, 67122.36, 67174.42, 67222.93, 67278.46, 67326.67, 67378.50, 67434.37])

# Starship_MEO_RTLS_FirstStage_PropMass=0.001*np.array([53326.76, 53362.24, 53397.68, 53433.08, 53468.45, 53503.79, 53539.10, 53574.37, 42135.27, 40992.16, 42218.01])
Starship_MEO_RTLS_FirstStage_PropMass=0.001*np.array([51383.16, 51306.99, 51337.90, 51543.69, 52161.42, 51430.55, 51461.41, 52171.78, 51523.09, 51553.91, 52273.58])
Starship_MEO_RTLS_FirstStage_PropMass=0.001*np.array([52675.90, 52710.54, 52774.47, 52779.72, 52814.25, 52907.12, 52913.23, 52947.63, 52981.93, 53054.86, 53019.53])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_FirstStage_PropMass=0.001*np.array([57156.11, 57205.46, 57254.79, 57304.09, 57304.09, 57402.62, 57451.84, 57501.03, 57550.20, 57599.34, 57648.46])
Starship_TLI_ASDS_FirstStage_PropMass=0.001*np.array([43629.87, 43671.20, 43712.56, 43753.92, 43795.28, 43836.64, 43878.00, 43919.36, 43960.73, 44002.09, 44031.60])
Starship_TLI_RTLS_FirstStage_PropMass=0.001*np.array([29904.38, 30764.98, 30793.27, 30821.55, 30843.31, 30878.06, 30906.28, 30934.47, 30962.64, 30990.78, 31018.89])

Starship_TLI_ASDS_FirstStage_PropMass=0.001*np.array([43549.99, 43589.43, 43628.86, 43668.28, 43707.68, 43973.83, 44012.48, 44051.11, 44089.74, 44128.35, 43622.57])
Starship_TLI_RTLS_FirstStage_PropMass=0.001*np.array([30259.15, 30294.00, 30321.75, 30349.48, 30377.18, 30404.85, 30432.48, 30460.09, 30487.67, 30515.21, 30542.73])

#======================================== Orbit Name: LEO ========================================
Starship_LEO_EXP_SecondStage_PropMass=0.001*np.array([260596.69, 260588.83, 260580.29, 260571.05, 260561.11, 260550.49, 260539.17, 260527.15, 260514.44, 260501.03, 260486.92])
Starship_LEO_ASDS_SecondStage_PropMass=0.001*np.array([225073.80, 230095.22, 230131.39, 230166.82, 230610.45, 230645.15, 230679.10, 225573.89, 227186.87, 227207.79, 227227.98])
Starship_LEO_RTLS_SecondStage_PropMass=0.001*np.array([200712.66, 200814.30, 200915.07, 201014.98, 201114.02, 201212.20, 201309.50, 201405.95, 201501.52, 201596.23, 201690.07])
#======================================== Orbit Name: MEO ========================================
Starship_MEO_EXP_SecondStage_PropMass=0.001*np.array([85524.30, 85550.70, 85576.77, 85602.51, 85627.76, 85652.12, 85675.67, 85698.85, 85721.70, 85744.22, 85766.34])
Starship_MEO_ASDS_SecondStage_PropMass=0.001*np.array([66937.83, 66980.80, 67023.44, 67065.67, 67107.68, 67149.34, 67190.68, 67231.69, 67272.36, 67312.71, 67352.73])
Starship_MEO_RTLS_SecondStage_PropMass=0.001*np.array([51032.67, 51112.96, 51192.89, 51381.84, 51351.69, 51538.68, 51509.06, 51694.06, 51665.00, 52398.72, 51819.50])
#======================================== Orbit Name: TLI ========================================
Starship_TLI_EXP_SecondStage_PropMass=0.001*np.array([57271.46, 57298.22, 57324.71, 57350.94, 57376.91, 57402.62, 57428.06, 57453.23, 57478.14, 57502.78, 57527.16])
Starship_TLI_ASDS_SecondStage_PropMass=0.001*np.array([42096.70, 42137.95, 42178.93, 42219.66, 42260.12, 42300.33, 42340.27, 42379.95, 42419.36, 42458.51, 42497.40])
Starship_TLI_ASDS_SecondStage_PropMass=0.001*np.array([43697.01, 43795.44, 43520.23, 43885.40, 43611.01, 44051.21, 44095.30, 44108.91, 43787.81, 44146.76, 43874.23])

Starship_TLI_RTLS_SecondStage_PropMass=0.001*np.array([28922.26, 28890.57, 29553.86, 29029.15, 29098.01, 29750.89, 29661.63, 29400.20, 29367.99, 30080.97, 29503.23])
Starship_TLI_RTLS_SecondStage_PropMass=0.001*np.array([30074.72, 30141.37, 30203.96, 30270.80, 30336.50, 30401.91, 30467.02, 30531.83, 30596.35, 30660.57, 30724.49])

FirstStagePropMass = np.linspace(-10000,10000, 11)
SecondStagePropMass = np.linspace(-10000,10000, 11)
if plot_propmass:
    plot_metric_pair(
        fig_num=8,
        title_base="Propellant Mass",
        x_first=FirstStagePropMass/1000,
        x_second=SecondStagePropMass/1000,
        xlabels=("ΔPropellant Mass [Tons]", "ΔPropellant Mass [Tons]"),
        ylabels=("Payload Capacity [Tons]", "Payload Capacity Change [%]"),
        met_first="FirstStage_PropMass",
        met_second="SecondStage_PropMass",
    )
    plt.savefig("PropMass_Sensitivity.png", dpi=300, bbox_inches="tight")
    print_prop_mass_sensitivity_table(step_kg=1)

plt.show()
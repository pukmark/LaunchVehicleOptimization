Convergence diagnosis — 9 November 2026

The strongest demonstrated sources of numerical sensitivity are the final-orbit
maneuver reserve and the booster's velocity-directed landing thrust. The
following experiments used temporary instrumentation; production equations and
physical limits were preserved in the repository.

The final-orbit reserve is calculated in
`LV_Optimization_Type.py::_orbit_burn_quantities`. Its plane geometry contains
square roots, clipping and branch changes. In particular,
`final_mass = if_else(margin >= 0, state[6] * retained_fraction, 0)` makes the
reserve jump to the entire terminal upper-stage mass when the target plane
cannot contain the burn position. Geometric feasibility is thereby encoded as
a discontinuous mass penalty inside the final mass constraint.

A numerical boundary probe used a 150 × 200 km parking orbit, a 200 km circular
reference orbit, target inclination 28.6 degrees, and terminal mass 20,000 kg.
At burn latitude 28.599999999 degrees, the computed reserve was 87.3616 kg.
At 28.6 degrees, the computed feasibility margin was approximately -1.89e-15
and the reserve became 20,000 kg. Thus roundoff alone can select the wrong side
of the branch at an exactly reachable geometric boundary. This is additional
to the physically genuine feasibility boundary.

Recovery dynamics in `LV_Type_scaled.py::BoosterReturn_2D` obtain thrust
orientation from `atan2(vz, vx)`, then multiply thrust and drag by its sine and
cosine. The direction is undefined at zero velocity; its derivatives grow as
speed approaches zero. A scaled dynamics-Jacobian probe for Falcon 9 True,
27,100 kg mass, 10 m altitude and thrust input 0.2 gave maximum entries of
approximately 345 at 5 m/s, 17,258 at 0.1 m/s, and 1,725,803 at 0.001 m/s.
At zero speed the Jacobian contained nonfinite values. Newton iterates and RK4
intermediate states can approach this region even if the final speed constraint
allows a nonzero touchdown speed.

Controlled solves used nominal dispersion, payload maximization, initial scalar
seed 0.5, MA97, tolerance 1e-6, a 400-iteration limit, and a 60-second IPOPT CPU
limit. Instrumentation recorded constraint residuals every 25 iterations.

| Scenario | Temporary diagnostic formulation | Iterations | Solve wall time [s] | Payload [kg] | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Falcon 9 True, EXP, 200 × 200 km, 28.6° | Original | 117 | 8.30 | 19,911.3102 | Success |
| Same EXP scenario | Plane-change component omitted from reserve | 26 | 2.17 | 19,911.3102 | Success |
| Falcon 9, ASDS, 300 × 200 km, 28.6° | Original | 294 | 30.26 | 20,114.9054 | Success |
| Same ASDS scenario | Plane-change component omitted from reserve | 271 | 27.62 | 20,102.6516 | Success |
| Same ASDS scenario | Landing direction regularized only | 400 | 52.98 | — | Iteration limit |
| Same ASDS scenario | Plane-change component omitted and landing direction regularized | 89 | 12.32 | 20,114.8971 | Success |
| Falcon 9, EXP, 300 × 200 km, 28.6° | Plane-change component omitted from reserve | 32 | 2.71 | 23,640.1024 | Success |

For the reserve experiment, the diagnostic replaced the plane-change-dependent
reserve by the rocket-equation reserve for the parking-to-target speed change.
It deliberately omits a physical cost and must not be used as the general fix.
In the tested EXP scenario, the optimized payload and actual full-model final
maneuver Δv were essentially identical between the two formulations. That
comparison directly demonstrates a numerical cost of the general plane-change
expression even where no appreciable plane change is required at the optimum.

For the landing experiment, thrust/drag direction used
`(vx, vz) / sqrt(vx**2 + vz**2 + 0.1**2)` instead of the angle expression.
This is a small model regularization, not an exactly equivalent formulation.
The combined diagnostic's payload differed from the original by approximately
0.0084 kg. All pressure, acceleration, heating, thrust and time limits were kept.
The failure of the landing-only experiment shows why changing a single setting
cannot be assumed to resolve the coupled problem.

The original ASDS solve repeatedly revisited upper-stage parking-apogee and
final-mass constraints. After removing the plane-change reserve expression,
landing-dynamics residuals dominated many intermediate iterations. This
supports two interacting difficult blocks rather than one universally bad
mission parameter. These bounded runs characterize sensitivity; they do not
establish that every reported nonconvergence has the same cause.

A secondary conditioning issue is the EXP first-stage final-mass constraint:
it is expressed in kilograms while most other constraints are normalized.
Its maximum scaled-variable Jacobian entry was 100,000, compared with roughly
101 for fairing separation and 15.5 for ascent dynamic pressure. IPOPT applies
additional scaling, and this issue's contribution was not isolated in these
experiments. Normalize this constraint consistently when revising the model.

Recommended implementation order:

1. Replace the discontinuous orbit-reserve feasibility penalty with explicit,
   continuous final-maneuver geometry constraints and a Δv variable. Retain the
   full plane-change cost and handle circular orbits without a discontinuous
   choice of apsidal direction.
2. Use a smooth low-speed thrust-direction model, or explicitly optimize landing
   thrust direction. Check both dynamics and RK4 derivatives near touchdown.
3. Normalize the remaining dimensional constraints, then validate warm starts
   and small parameter perturbations across EXP, ASDS and RTLS.
4. Preserve the intended Max-Q, heating and acceleration limits during this
   numerical work. Their changes can change the active constraints and amplify
   the sensitivity, but relaxing them is not established as the remedy.

Temporary reproduction harness and detailed solver histories for this session
are in `/tmp/lv_convergence_probe.py` and `/tmp/lv_probe_*.json`; they are not
part of the production solver.

Implementation and validation

The production solver now uses `FinalManeuver.py` to optimize the post-burn
velocity vector and a coast angle in the cutoff orbital plane. Two projected
eccentricity equations put the burn at parking apogee without dividing by the
eccentricity norm. The parking-perigee bound is expressed through inverse
semimajor axis, avoiding an energy division in trial states. Unit magnitude,
tangency and inclination constraints enforce the requested deployment orbit.
The full vector velocity change determines the rocket-equation reserve; an
unreachable target produces constraint infeasibility rather than a mass jump.
The Δv norm uses a conservative 0.001 m/s floor to keep its Hessian finite at
zero impulse. Direct-injection cases use energy/angular-momentum constraints
and retain zero second-burn reserve.

Landing throttle remains in the existing `u4_landing` field. A new
`landing_thrust_angle` field stores radians from the current radial vertical.
The angle is bounded to the outward hemisphere (±90 degrees), with a smooth
thrust-dot-velocity constraint to retain braking. Thrust magnitude, mass flow,
load limits and timing limits retain their existing definitions. Drag and
reentry velocity direction use a 0.001 m/s speed floor; the landing thrust
magnitude itself is not attenuated at rest. Heat flux and terminal-speed
constraints have finite derivatives at zero velocity. The equivalent EXP
separation-mass inequality is normalized by its existing mass scale.

Saved throttle-only trajectories acquire initial landing angles from their
retrograde velocity direction. New results preserve the optimized maneuver
position, before/after velocities, coast angle, Δv and landing angles. GUI and
Main plots use these values, and warm-start cache fingerprints include the new
maneuver module. Physical pressure, heating, acceleration and thrust limits
were not relaxed. Explicit landing steering changes the control model, and
local payload optima can differ between initial estimates.

Cold-start checks on the final formulation, using the current production cost
settings, a 400-iteration limit and a 45-second IPOPT CPU limit:

| Scenario | Initial payload-estimate factor | Iterations | Wall time [s] | Payload [kg] |
| --- | ---: | ---: | ---: | ---: |
| Falcon 9 True, EXP, 200 × 200 km, 28.6° | 0.35 | 62 | 5.62 | 19,911.31 |
| Falcon 9 True, EXP, 200 × 200 km, 28.6° | 0.50 | 109 | 8.63 | 19,911.31 |
| Falcon 9, ASDS, 300 × 200 km, 28.6° | 0.35 | 168 | 19.36 | 20,105.41 |
| Falcon 9, ASDS, 300 × 200 km, 28.6° | 0.50 | 80 | 10.06 | 20,005.54 |
| Falcon 9, RTLS, 300 × 200 km, 28.6° | 0.35 | 100 | 13.83 | 15,832.05 |
| Falcon 9, RTLS, 300 × 200 km, 28.6° | 0.50 | 128 | 17.82 | 15,812.34 |

All six solves reported success. The old saved ASDS trajectory also converged
in 15 iterations after migration. A saved SPHEREx RTLS trajectory converged in
24 iterations; its cold-start solve still reached the 400-iteration limit.
This remains a nonlinear local optimization, so the changes do not guarantee
convergence or a unique maximum payload for every mission and initial estimate.

Standalone tests verify agreement with analytic plane-change Δv for prograde
and retrograde targets, finite constraint Jacobians/Hessians for circular and
zero-impulse cases, finite landing/RK4/heat-flux derivatives at zero speed, and
unit landing thrust direction with unchanged mass flow. The actual sensitivity
worker path has also been exercised with ±300 Pa ascent-pressure perturbations
around a 30,000 Pa baseline for EXP, ASDS and RTLS.

Final regression run: 131 tests, 109 passed and 22 display-dependent tests
skipped. Six reconstructed final orbits matched target apsidal radii within
0.022 m and passed inclination, retained-mass and touchdown-speed checks. The
interactive iteration-plot callback also completed a headless one-iteration
smoke check; its iteration-limit status was intentional.

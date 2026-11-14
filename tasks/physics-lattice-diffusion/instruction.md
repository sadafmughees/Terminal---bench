# 2D Nonlinear Steady-State Lattice Heat Conduction & Thermal Runaway Bifurcation

You are tasked with building a numerical solver in `/app/solve_lattice.py` to model steady-state nonlinear heat conduction, thermal runaway bifurcation (fold point), linear stability, and Method of Manufactured Solutions (MMS) convergence on a 2D square crystalline silicon lattice domain $\Omega = [0, L] \times [0, L]$ with $L = 0.01\text{ m}$ ($1\text{ cm} \times 1\text{ cm}$).

Use **JAX** (`jax`, installed in the environment with 64-bit precision enabled) alongside sparse linear algebra routines to formulate the discrete residual, compute exact automatic-differentiation derivatives, and solve the nonlinear system.

## 1. Governing Physics & Constitutive Relations

The steady-state energy balance per unit volume is:
$$R(T, Q_0) = \nabla \cdot \big(\kappa(T)\nabla T\big) + Q(x, y, T) = 0$$

- **Temperature-dependent thermal conductivity** (phononic $1/T$ lattice degradation model):
  $$\kappa(T) = \frac{\kappa_0}{1 + \alpha (T - T_0)}$$
- **Localized exponential (Frank-Kamenetskii) volumetric heat source**:
  $$Q(x, y, T) = Q_0 \exp\!\Big(-\eta \big[(x - L/2)^2 + (y - L/2)^2\big]\Big) \exp\!\left(\frac{\gamma (T - T_0)}{T_0}\right)$$

### Baseline Physical Parameters
| Parameter | Symbol | Value | Units |
| :--- | :--- | :--- | :--- |
| Mass density | $\rho$ | `2330.0` | $\text{kg/m}^3$ |
| Specific heat capacity | $c$ | `700.0` | $\text{J/(kg}\cdot\text{K)}$ |
| Reference thermal conductivity | $\kappa_0$ | `150.0` | $\text{W/(m}\cdot\text{K)}$ |
| Conductivity temperature coefficient | $\alpha$ | `0.003` | $\text{K}^{-1}$ |
| Reference / heat-sink temperature | $T_0$ | `300.0` | $\text{K}$ |
| Ambient radiation temperature | $T_{\text{amb}}$ | `300.0` | $\text{K}$ |
| Domain side length | $L$ | `0.01` | $\text{m}$ |
| Gaussian source localization parameter | $\eta$ | `5.0e5` | $\text{m}^{-2}$ |
| Exponential thermal feedback parameter | $\gamma$ | `8.0` | dimensionless |
| Baseline source amplitude | $Q_0$ | `5.0e8` | $\text{W/m}^3$ |
| Top surface emissivity | $\varepsilon$ | `0.85` | dimensionless |
| Stefan-Boltzmann constant | $\sigma$ | `5.670374e-8` | $\text{W/(m}^2\cdot\text{K}^4)$ |

## 2. Discretization & Boundary Conditions

Discretize $\Omega$ on a uniform $N \times N$ cell-centered finite-volume grid ($N = 100$ for the physical problem), where $\Delta x = \Delta y = L / N$ and cell centers are located at:
$$x_j = (j + 0.5)\Delta x, \quad y_i = (i + 0.5)\Delta y, \quad i, j \in \{0, 1, \dots, N-1\}$$
Use harmonic-mean face conductivities $\kappa_f = \frac{2\kappa_a \kappa_b}{\kappa_a + \kappa_b}$ across interior cell faces.

Apply the following boundary conditions for the physical problem:
- **Bottom boundary ($y = 0$, row $i = 0$):** Isothermal Dirichlet condition $T(x, 0) = T_0 = 300\text{ K}$, discretized with half-cell distance $\Delta y / 2$ and harmonic-mean face conductivity between $\kappa(T_{0,j})$ and $\kappa(T_0)$.
- **Top boundary ($y = L$, row $i = N-1$):** Nonlinear radiative cooling flux $q_{\text{rad}} = \varepsilon \sigma (T_{N-1,j}^4 - T_{\text{amb}}^4)$ leaving the top face.
- **Left and right boundaries ($x = 0$ and $x = L$, columns $j = 0$ and $j = N-1$):** Adiabatic (zero heat flux, $\partial T / \partial x = 0$).

## 3. Required Analysis & Deliverables

1. **Lower-Branch Steady-State Field ($T_{\text{field}}$):** Solve for the stable lower-branch temperature distribution $T(x, y)$ at the specified $Q_0$ (with $Q_0 < Q_{\text{crit}}$).
2. **Thermal Runaway Fold Bifurcation ($Q_{\text{crit}}, T_{\text{crit}}$):** Trace the steady-state solution branch as $Q_0$ increases from $0$ past the saddle-node (fold) bifurcation turning point where the lower stable branch meets the upper unstable branch. Extract the critical source amplitude $Q_{\text{crit}}$ ($\text{W/m}^3$) and corresponding peak domain temperature $T_{\text{crit}} = \max_{x,y} T(x,y)$ ($\text{K}$) at the fold.
3. **Linear Stability Index ($S$):** At the baseline lower-branch state $(T, Q_0)$, form the linearized transient operator $A = \frac{1}{\rho c} \frac{\partial R}{\partial T}$ and compute its rightmost eigenvalue (largest real part):
   $$S = \max_{\lambda \in \sigma(A)} \operatorname{Re}(\lambda) \quad (\text{s}^{-1})$$
   For a thermally stable lower-branch operating point ($Q_0 < Q_{\text{crit}}$), $S < 0$.
4. **Method of Manufactured Solutions (`mms_slope`):** Verify second-order spatial convergence using the manufactured temperature field:
   $$T_{\text{ex}}(x, y) = T_0 + 50.0 \sin\!\left(\frac{\pi x}{L}\right)\sin\!\left(\frac{\pi y}{L}\right)$$
   with exact Dirichlet boundary conditions ($T = T_0$ on all four boundaries, half-cell distance $h/2$ at boundary faces) and analytical source term $S_{\text{MMS}}(x, y) = -\big[\nabla \cdot (\kappa(T_{\text{ex}})\nabla T_{\text{ex}}) + Q(x, y, T_{\text{ex}})\big]$ at baseline parameters ($\alpha = 0.003, \gamma = 8.0, Q_0 = 5.0\times 10^8$). Solve on grids $N \in \{25, 50, 100\}$, compute the discrete $L_2$ error $\|T - T_{\text{ex}}\|_{L_2} = \sqrt{\frac{1}{N^2}\sum_{i,j} (T_{i,j} - T_{\text{ex}}(x_j, y_i))^2}$, and fit the convergence order slope $\log(\text{error})$ vs. $\log(h)$.
    python /app/solve_lattice.py --alpha <val> --gamma <val> --q0 <val> --out <path> [--skip-mms]

- Defaults: `--alpha 0.003`, `--gamma 8.0`, `--q0 5.0e8`, `--out /app/solution_metrics.json`.
- When `--skip-mms` is omitted, run the baseline problem and write `/app/solution_metrics.json` containing:
  - `"T_field"`: $100 \times 100$ nested list of floats (row index `i` corresponds to $y_i$, column index `j` corresponds to $x_j$, in Kelvin).
  - `"Q_crit"`: float, critical fold bifurcation source amplitude ($\text{W/m}^3$).
  - `"T_crit"`: float, peak temperature at the fold bifurcation ($\text{K}$).
  - `"S"`: float, largest real eigenvalue of $\frac{1}{\rho c}\frac{\partial R}{\partial T}$ ($\text{s}^{-1}$).
  - `"mms_slope"`: float, fitted spatial convergence order across $N \in \{25, 50, 100\}$ (omitted only if `--skip-mms` is passed).

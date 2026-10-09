# Instruction: Physics Lattice Diffusion (Heat Conduction)

i have used Python Numpy to make grid of 16x 16 which is 2D and I create the situation on diffusion of heat 
i have set meta data with the code verification test .Py file
2D steady state heat conduction 1x1cm domain temprature dependent conductivity,localised heat source it depend on exponentially on temperature
the physical values rho,c,kappa_0 ALPHA T_0,T_ Amb ,eta ,gamma,epsilon,sigma and baseline Q_0=5.0e8w/m square
boundry condition downside dirichilet 300k upside radiative both side adiabatic,grid 100x100 cell centered x_j = (j+0.5)dx, y_i = (i+0.5)dy.
Output file: /app/solution_metrics.json ke keys T_field (100x100, row index i = y, column j = x, Kelvin), Q_crit (W/m³), T_crit (K, fold on peak temperature), S (s⁻¹, baseline par (1/ρc) ∂R/∂T the bigest real eigenvalue), mms_slope.
MMS check: T = T0 + 50 sin(πx/L) sin(πy/L), Dirichlet-from-exact boundaries, grids 25, 50, 100, L2 error se fitted order.
CLI: /app/solve_lattice.py --alpha <val> --gamma <val> --q0 <val> --out <path> jo T_field aur S likhe, bina Q_0 se zyada Q_crit  values for this.

import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
from scipy.linalg import expm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# --- Configure Matplotlib for English display ---
plt.rcParams['font.sans-serif'] = ['Arial']  # Default to Arial or a suitable English font
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly
print("Matplotlib font configuration updated to use English fonts.")


# --- CSTR Model Function ---
def cstr_model(state, inputs, params):
    """
    CSTR model differential equations.
    state = [c_A, T]
    inputs = [c_A0, Q]
    params = dict of model parameters (V, F, T0, rho, Cp, k0, E, delta_h, R_gas)
    """
    c_A, T = state
    c_A0, Q = inputs

    # Unpack parameters
    V = params['V']
    F = params['F']
    T0 = params['T0']
    rho = params['rho']
    Cp = params['Cp']
    k0 = params['k0']
    E = params['E']
    delta_h = params['delta_h']
    R_gas = params['R_gas']

    # Reaction rate constant (Arrhenius equation)
    k = k0 * np.exp(-E / (R_gas * T))

    # Differential equations
    d_cA_dt = (F / V) * (c_A0 - c_A) - k * c_A ** 2
    d_T_dt = (F / V) * (T0 - T) - (delta_h / (rho * Cp)) * k * c_A ** 2 + (Q / (rho * Cp * V))

    return np.array([d_cA_dt, d_T_dt])


# --- Model Parameters (from Table 1) ---
nominal_params = {
    'V': 1.0,  # m^3
    'F': 5.0,  # m^3/h
    'T0': 300.0,  # K
    'rho': 1000.0,  # kg/m^3
    'Cp': 0.231,  # kJ/(kg*K)
    'k0': 10 ** 6,  # m^3/(kmol*h)
    'E': 10 ** 4,  # kJ/kmol
    'delta_h': -1.15 * 10 ** 4,  # kJ/kmol
    'R_gas': 8.3145  # kJ/(kmol*K)
}


# --- Find Steady State Operating Point ---
def find_steady_state(inputs_ss, params_ss, initial_guess):
    """
    Solves steady-state equations f(x, u) = 0 to find steady state x_ss.
    Uses fsolve for numerical solution.
    Accepts an initial_guess parameter.
    """

    def equations_to_solve(state_ss):
        return cstr_model(state_ss, inputs_ss, params_ss)

    steady_state_x = fsolve(equations_to_solve, initial_guess)
    return steady_state_x


# Define inputs for finding steady state
ss_inputs = np.array([5.0, 0.0])  # c_A0 = 5.0 kmol/m^3, Q = 0.0 kJ/h

# --- Symbolize Model for Jacobian Matrix Calculation ---
c_A_sym, T_sym = sympy.symbols('c_A T')
c_A0_sym, Q_sym = sympy.symbols('c_A0 Q')
delta_h_sym, k0_sym = sympy.symbols('delta_h k0')

# Other parameters as constant symbols
V_sym, F_sym, T0_sym, rho_sym, Cp_sym, E_sym, R_gas_sym = sympy.symbols(
    'V F T0 rho Cp E R_gas'
)

# Reaction rate constant (symbolic form)
k_rate_sym = k0_sym * sympy.exp(-E_sym / (R_gas_sym * T_sym))

# Differential equations (symbolic form)
dc_A_dt_sym = (F_sym / V_sym) * (c_A0_sym - c_A_sym) - k_rate_sym * c_A_sym ** 2
dT_dt_sym = (F_sym / V_sym) * (T0_sym - T_sym) - (delta_h_sym / (rho_sym * Cp_sym)) * k_rate_sym * c_A_sym ** 2 + (
            Q_sym / (rho_sym * Cp_sym * V_sym))

f_sym = sympy.Matrix([dc_A_dt_sym, dT_dt_sym])
x_sym_vec = sympy.Matrix([c_A_sym, T_sym])
u_sym_vec = sympy.Matrix([c_A0_sym, Q_sym])
p_sym_vec = sympy.Matrix([delta_h_sym, k0_sym])

# Calculate Jacobian matrices
A_sym_matrix = f_sym.jacobian(x_sym_vec)
B_sym_matrix = f_sym.jacobian(u_sym_vec)
G_sym_matrix = f_sym.jacobian(p_sym_vec)


# --- Function to Evaluate Jacobian Matrices Numerically ---
def get_jacobian_matrices(state_op, inputs_op, params_op):
    substitutions = {
        c_A_sym: state_op[0],
        T_sym: state_op[1],
        c_A0_sym: inputs_op[0],
        Q_sym: inputs_op[1],
        V_sym: params_op['V'],
        F_sym: params_op['F'],
        T0_sym: params_op['T0'],
        rho_sym: params_op['rho'],
        Cp_sym: params_op['Cp'],
        E_sym: params_op['E'],
        R_gas_sym: params_op['R_gas'],
        delta_h_sym: params_op['delta_h'],
        k0_sym: params_op['k0']
    }

    A_val = np.array(A_sym_matrix.subs(substitutions)).astype(np.float64)
    B_val = np.array(B_sym_matrix.subs(substitutions)).astype(np.float64)
    G_val = np.array(G_sym_matrix.subs(substitutions)).astype(np.float64)

    return A_val, B_val, G_val


# --- Explore Different Steady States ---
print("\n--- Exploring Different Steady States and Stability ---")

initial_T_guesses = np.linspace(300, 450, 10)
found_steady_states = []

for i, initial_T_guess in enumerate(initial_T_guesses):
    initial_guess_cA = 3.0
    initial_guess = np.array([initial_guess_cA, initial_T_guess])

    try:
        current_steady_state = find_steady_state(ss_inputs, nominal_params, initial_guess)
        is_new_state = True
        for s_state in found_steady_states:
            if np.allclose(s_state, current_steady_state, atol=1e-3):
                is_new_state = False
                break

        if is_new_state:
            found_steady_states.append(current_steady_state)
            print(f"\n--- Found a new steady state (from initial guess T={initial_T_guess:.2f}K) ---")
            print(
                f"Steady state operating point x_op (cA, T): {current_steady_state[0]:.4f} kmol/m^3, {current_steady_state[1]:.4f} K")
            print(
                f"Derivatives at steady state (should be near zero): {cstr_model(current_steady_state, ss_inputs, nominal_params)}")

            A_current, _, _ = get_jacobian_matrices(current_steady_state, ss_inputs, nominal_params)
            eigenvalues = np.linalg.eigvals(A_current)
            print(f"Eigenvalues of continuous-time A matrix: {eigenvalues}")

            if np.all(np.real(eigenvalues) < 0):
                print("This steady state is: **Stable**.")
            else:
                print("This steady state is: **Unstable** (eigenvalues with non-negative real part exist).")

    except Exception as e:
        print(f"\nFailed to solve for steady state with initial guess T={initial_T_guess:.2f}K: {e}")

if not found_steady_states:
    print("\nNo steady states found. Please check model parameters or adjust initial guess range.")
    exit()

# Choose a stable operating point for further analysis (default to the first one found, modify based on stability analysis if needed)
stable_operating_points = []
for s_state in found_steady_states:
    A_temp, _, _ = get_jacobian_matrices(s_state, ss_inputs, nominal_params)
    eigenvalues_temp = np.linalg.eigvals(A_temp)
    if np.all(np.real(eigenvalues_temp) < 0):
        stable_operating_points.append(s_state)

if stable_operating_points:
    nominal_state = stable_operating_points[0]  # Select the first stable point as nominal
    print(
        f"\nSelected the first stable steady state as nominal operating point: cA={nominal_state[0]:.4f}, T={nominal_state[1]:.4f}")
else:
    print(
        "\nWarning: No stable steady states found. Covariance propagation may still overflow. Using the first found steady state for further analysis.")
    nominal_state = found_steady_states[0]  # If no stable point, use the first found (potentially unstable) one

nominal_inputs = ss_inputs

# --- Error Propagation Example (using selected nominal operating point) ---
print("\n--- First-Order Taylor Expansion Static Error Propagation (using selected nominal operating point) ---")
A_op, B_op, G_op = get_jacobian_matrices(nominal_state, nominal_inputs, nominal_params)

print("A matrix at operating point (sensitivity of states to states):\n", A_op)

delta_x0 = np.array([0.01, 0.5])
delta_u = np.array([0.02, 10.0])
delta_p = np.array([nominal_params['delta_h'] * 0.01, nominal_params['k0'] * 0.01])

delta_x_dot = A_op @ delta_x0 + B_op @ delta_u + G_op @ delta_p

print(f"\nInitial state error delta_x0: {delta_x0}")
print(f"Input error delta_u: {delta_u}")
print(f"Parameter error delta_p (Delta_h, k0): {delta_p}")
print(f"\nEstimated error in state derivatives (dc_A/dt, dT/dt) at current time delta_x_dot:\n {delta_x_dot}")

# --- Covariance Propagation Analysis (using selected nominal operating point) ---

# --- Discretize A Matrix ---
A_continuous = A_op

dt = 0.01
A_discrete = expm(A_continuous * dt)

print("\n--- Covariance Propagation Analysis (using selected nominal operating point) ---")
print("Discrete A matrix (A_discrete):\n", A_discrete)

eigenvalues_discrete = np.linalg.eigvals(A_discrete)
print(f"Eigenvalues of discrete A matrix: {eigenvalues_discrete}")
if np.all(np.abs(eigenvalues_discrete) < 1):
    print("Discrete system is stable (all eigenvalue magnitudes are less than 1).")
else:
    print(
        "Warning: Discrete system might be unstable (eigenvalues with magnitude >= 1 exist). Ensure you've chosen a truly stable operating point if overflow occurs.")

# --- Define Initial State Covariance P0 ---
sigma_cA_initial = 0.05
sigma_T_initial = 2.0

P0 = np.array([
    [sigma_cA_initial ** 2, 0.0],
    [0.0, sigma_T_initial ** 2]
])
print("\nInitial state covariance P0:\n", P0)

# --- Define Process Noise Covariance Q ---
Q_process_noise = np.array([
    [5e-6, 0.0],
    [0.0, 5e-5]
])
print("\nProcess noise covariance Q_process_noise:\n", Q_process_noise)

# --- Simulation Time ---
time_horizon = 2.0  # hours
num_steps = int(time_horizon / dt)
time_points = np.linspace(0, time_horizon, num_steps + 1)

# --- Covariance Propagation Loop ---
P_history = [P0]

for k in range(num_steps):
    P_k = P_history[-1]
    P_k_plus_1 = A_discrete @ P_k @ A_discrete.T + Q_process_noise
    P_history.append(P_k_plus_1)


# --- Helper function to plot confidence ellipse ---
def plot_covariance_ellipse(ax, mean, cov, confidence_level=0.95, **kwargs):
    # Check if covariance matrix is valid (positive definite)
    if not (np.all(np.isfinite(cov)) and cov[0, 0] > 0 and cov[1, 1] > 0 and np.linalg.det(cov) > 0):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    chi2_val = chi2.ppf(confidence_level, df=2)

    width = np.sqrt(eigenvalues[0] * chi2_val) * 2
    height = np.sqrt(eigenvalues[1] * chi2_val) * 2

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# --- Covariance Propagation Ellipses in State Space (c_A vs T) ---
fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(nominal_state[0], nominal_state[1], 'ko', markersize=8, label='Nominal Operating Point')

times_to_plot = [0, num_steps // 4, num_steps // 2, num_steps * 3 // 4, num_steps]
colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for i, idx in enumerate(times_to_plot):
    current_P = P_history[idx]
    current_time = time_points[idx]

    ell = plot_covariance_ellipse(ax, nominal_state, current_P, color=colors[i], alpha=0.6)
    if ell:
        ell.set_label(f't={current_time:.2f}h 95% Confidence Ellipse')

ax.set_xlabel('$c_A$ (kmol/m$^3$)')
ax.set_ylabel('T (K)')
ax.set_title('Covariance Propagation in State Space: Evolution of Uncertainty Ellipses')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()

# --- Plot Standard Deviation of Each State Variable over Time ---
valid_P_history_for_plot = [P for P in P_history if
                            np.all(np.isfinite(P)) and P[0, 0] > 0 and P[1, 1] > 0 and np.linalg.det(P) > 0]
valid_time_points_for_plot = time_points[:len(valid_P_history_for_plot)]

if len(valid_P_history_for_plot) > 0:
    std_dev_cA_history = np.array([np.sqrt(P[0, 0]) for P in valid_P_history_for_plot])
    std_dev_T_history = np.array([np.sqrt(P[1, 1]) for P in valid_P_history_for_plot])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(valid_time_points_for_plot, nominal_state[0] + std_dev_cA_history * 3, 'r--',
             label='Nominal $\pm$ 3$\sigma$')
    plt.plot(valid_time_points_for_plot, nominal_state[0] - std_dev_cA_history * 3, 'r--')
    plt.plot(valid_time_points_for_plot, [nominal_state[0]] * len(valid_time_points_for_plot), 'k-',
             label='Nominal $c_A$')
    plt.fill_between(valid_time_points_for_plot, nominal_state[0] - std_dev_cA_history * 3,
                     nominal_state[0] + std_dev_cA_history * 3,
                     color='blue', alpha=0.1, label='$\pm$ 3$\sigma$ Range')
    plt.xlabel('Time (h)');
    plt.ylabel('Concentration $c_A$ (kmol/m$^3$)');
    plt.title('$c_A$ Mean and 3$\sigma$ Uncertainty Range')
    plt.legend();
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(valid_time_points_for_plot, nominal_state[1] + std_dev_T_history * 3, 'r--',
             label='Nominal $\pm$ 3$\sigma$')
    plt.plot(valid_time_points_for_plot, nominal_state[1] - std_dev_T_history * 3, 'r--')
    plt.plot(valid_time_points_for_plot, [nominal_state[1]] * len(valid_time_points_for_plot), 'k-', label='Nominal T')
    plt.fill_between(valid_time_points_for_plot, nominal_state[1] - std_dev_T_history * 3,
                     nominal_state[1] + std_dev_T_history * 3,
                     color='red', alpha=0.1, label='$\pm$ 3$\sigma$ Range')
    plt.xlabel('Time (h)');
    plt.ylabel('Temperature T (K)');
    plt.title('T Mean and 3$\sigma$ Uncertainty Range')
    plt.legend();
    plt.grid(True)

    # Adjust Y-axis range for T vs Time plot to make it clearer
    max_std_T = np.max(std_dev_T_history) if len(std_dev_T_history) > 0 else 0
    y_min_T = nominal_state[1] - max_std_T * 4  # More space
    y_max_T = nominal_state[1] + max_std_T * 4
    if y_max_T - y_min_T < 10:  # If range is too small, provide a minimum display range
        y_min_T = nominal_state[1] - 5
        y_max_T = nominal_state[1] + 5
    plt.ylim([y_min_T, y_max_T])

    plt.tight_layout();
    plt.show()
else:
    print(
        "\nCould not plot standard deviation of state variables over time as all covariance matrices are invalid. This typically means the system is unstable, causing uncertainty to diverge rapidly.")

# --- New Section: Plot Uncertainty Ellipse in (c_A0, T) Coordinates ---

print("\n--- Plotting Uncertainty Ellipse in (c_A0, T) Coordinates ---")

# 1. Define input (c_A0, Q) uncertainty covariance matrix
# Adjust sigma_cA0_input value to make it more prominent on the plot
sigma_cA0_input = 0.5  # Increased standard deviation for c_A0 for better visibility
sigma_Q_input = 50.0  # Q standard deviation remains unchanged unless adjusted

P_inputs_uncertainty = np.array([
    [sigma_cA0_input ** 2, 0.0],
    [0.0, sigma_Q_input ** 2]
])
print("\nInput (c_A0, Q) Uncertainty Covariance Matrix P_inputs_uncertainty:\n", P_inputs_uncertainty)

# 2. Calculate steady-state sensitivity matrix S_ux = - inv(A_op) @ B_op
# S_ux matrix propagates input uncertainty to steady-state uncertainty
try:
    inv_A_op = np.linalg.inv(A_op)
    S_ux = -inv_A_op @ B_op
    print("\nSteady-State Sensitivity Matrix S_ux (delta_x_ss / delta_u):\n", S_ux)
except np.linalg.LinAlgError:
    print(
        "\nWarning: Matrix A_op is non-invertible, cannot compute steady-state sensitivity matrix S_ux. System may be unstable or linearization problematic.")
    S_ux = np.zeros((2, 2))  # Provide a zero matrix to avoid subsequent errors, but results will be meaningless

# 3. Construct the joint covariance matrix P_cA0_Tss for (c_A0, T_ss)
if np.any(S_ux != 0):  # Only proceed if S_ux is valid
    # var(c_A0)
    var_cA0 = P_inputs_uncertainty[0, 0]

    # var(T_ss) (propagated from input uncertainty)
    var_T_ss = (S_ux[1, 0] ** 2 * P_inputs_uncertainty[0, 0] +
                S_ux[1, 1] ** 2 * P_inputs_uncertainty[1, 1] +
                2 * S_ux[1, 0] * S_ux[1, 1] * P_inputs_uncertainty[0, 1])

    # cov(c_A0, T_ss)
    cov_cA0_Tss = S_ux[1, 0] * P_inputs_uncertainty[0, 0] + S_ux[1, 1] * P_inputs_uncertainty[0, 1]

    P_cA0_Tss = np.array([
        [var_cA0, cov_cA0_Tss],
        [cov_cA0_Tss, var_T_ss]
    ])
    print("\nJoint Covariance Matrix P_cA0_Tss for (c_A0, T_ss):\n", P_cA0_Tss)

    # 4. Plot the ellipse
    fig_cA0_T, ax_cA0_T = plt.subplots(figsize=(9, 7))

    # Ellipse center is (nominal c_A0, nominal steady-state T)
    ellipse_center_cA0_T = np.array([nominal_inputs[0], nominal_state[1]])

    ell_cA0_T = plot_covariance_ellipse(ax_cA0_T, ellipse_center_cA0_T, P_cA0_Tss, color='blue', alpha=0.7,
                                        label='95% Confidence Ellipse')

    if ell_cA0_T:
        ax_cA0_T.plot(ellipse_center_cA0_T[0], ellipse_center_cA0_T[1], 'ro', markersize=8,
                      label='Nominal Operating Point ($c_{A0}$, T)')
        ax_cA0_T.set_xlabel('Feed Concentration $c_{A0}$ (kmol/m$^3$)')
        ax_cA0_T.set_ylabel('Reactor Temperature T (K)')
        ax_cA0_T.set_title('Steady-State Uncertainty Ellipse from Input Uncertainty in ($c_{A0}$, T) Coordinates')
        ax_cA0_T.legend()
        ax_cA0_T.grid(True)

        # --- Crucial Adjustment: Manually set axis limits for better visibility ---
        # Calculate approximate span of the ellipse
        # Note: This is a rough estimate as the ellipse is rotated
        chi2_val_for_span = chi2.ppf(0.95, df=2)  # Use 95% confidence for span calculation
        approx_cA0_span = np.sqrt(P_cA0_Tss[0, 0]) * np.sqrt(chi2_val_for_span) * 2
        approx_T_span = np.sqrt(P_cA0_Tss[1, 1]) * np.sqrt(chi2_val_for_span) * 2

        # Set limits slightly wider than the ellipse's approximate span
        margin_factor = 1.5  # 1.5 times the span for margin

        # Calculate limits based on nominal center and approximate span
        x_min_calculated = ellipse_center_cA0_T[0] - approx_cA0_span * margin_factor / 2
        x_max_calculated = ellipse_center_cA0_T[0] + approx_cA0_span * margin_factor / 2
        y_min_calculated = ellipse_center_cA0_T[1] - approx_T_span * margin_factor / 2
        y_max_calculated = ellipse_center_cA0_T[1] + approx_T_span * margin_factor / 2

        # Provide a fallback minimum range if the calculated range is too small
        # For c_A0, ensure at least 1.0 unit span if calculated is smaller
        if (x_max_calculated - x_min_calculated) < 1.0:
            x_min_calculated = ellipse_center_cA0_T[0] - 0.5
            x_max_calculated = ellipse_center_cA0_T[0] + 0.5
        # For T, ensure at least 10.0 K span if calculated is smaller
        if (y_max_calculated - y_min_calculated) < 10.0:
            y_min_calculated = ellipse_center_cA0_T[1] - 5.0
            y_max_calculated = ellipse_center_cA0_T[1] + 5.0

        ax_cA0_T.set_xlim([x_min_calculated, x_max_calculated])
        ax_cA0_T.set_ylim([y_min_calculated, y_max_calculated])

        ax_cA0_T.set_aspect('auto')  # Use auto aspect ratio as units are different
        plt.show()
    else:
        print("Could not plot ellipse in (c_A0, T) coordinates, covariance matrix might be invalid.")
else:
    print(
        "\nCould not plot ellipse in (c_A0, T) coordinates as steady-state sensitivity matrix S_ux is invalid. Please check system stability.")
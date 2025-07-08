import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.spatial.transform import Rotation as R


# 1. CSTR Model Equations
def cstr_model(t, y, params, c_A0_input, Q_input):
    """
    CSTR model differential equations.
    y = [c_A, T]
    params = [V, F, T0, rho, Cp, k0_val, E, delta_h_val, R_gas]
    c_A0_input: feed concentration (can be time-varying, but here constant for simplicity)
    Q_input: heating rate (can be time-varying, but here constant for simplicity)
    """
    c_A, T = y

    # Unpack parameters
    V, F, T0, rho, Cp, k0_val, E, delta_h_val, R_gas = params

    # Reaction rate constant (Arrhenius equation)
    k = k0_val * np.exp(-E / (R_gas * T))

    # Differential equations (from the document )
    d_cA_dt = (F / V) * (c_A0_input - c_A) - k * c_A ** 2
    d_T_dt = (F / V) * (T0 - T) - (delta_h_val / (rho * Cp)) * k * c_A ** 2 + (Q_input / (rho * Cp * V))

    return [d_cA_dt, d_T_dt]


# Explicit Euler Integration function
def explicit_euler(model_func, y0, t_span, dt, params, c_A0_input, Q_input):
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt)
    t_values = np.linspace(t_start, t_end, num_steps + 1)

    y_values = np.zeros((num_steps + 1, len(y0)))
    y_values[0] = y0

    for i in range(num_steps):
        dy_dt = model_func(t_values[i], y_values[i], params, c_A0_input, Q_input)
        y_values[i + 1] = y_values[i] + np.array(dy_dt) * dt
    return t_values, y_values


# 2. Define Nominal Parameters (from Table 1 )
V = 1.0  # m^3
F = 5.0  # m^3/h
T0 = 300.0  # K
rho = 1000.0  # kg/m^3
Cp = 0.231  # kJ/(kg*K)
k0_nominal = 10 ** 6  # m^3/(kmol*h)
E = 10 ** 4  # kJ/kmol
delta_h_nominal = -1.15 * 10 ** 4  # kJ/kmol
R_gas = 8.3145  # kJ/(kmol*K)

nominal_params_tuple = (V, F, T0, rho, Cp, k0_nominal, E, delta_h_nominal, R_gas)

# Initial conditions (from Table 2, within min/max range)
c_A_initial = 3.0  # kmol/m^3 (within 0-7.5)
T_initial = 350.0  # K (within 300-500)
initial_state = [c_A_initial, T_initial]

# Inputs (from Table 2, within min/max range)
c_A0_input = 5.0  # kmol/m^3 (within 0.5-7.5)
Q_input = 0.0  # kJ/h (can be positive or negative, within -5e5 to 5e5)

# Simulation time span and step
t_span = (0, 2.0)  # hours
dt = 0.01  # hours

# 3. Define the Ellipsoid for Uncertainty Analysis
# Center of the ellipsoid (nominal values)
mu = np.array([delta_h_nominal, k0_nominal])

# Define uncertainty ranges (e.g., +/- 10% for illustration, adjust as needed)
# The document does not provide uncertainty for parameters, this is an assumption.
percent_uncertainty_dh = 0.10  # +/- 10%
percent_uncertainty_k0 = 0.10  # +/- 10%

std_dev_dh = mu[0] * percent_uncertainty_dh
std_dev_k0 = mu[1] * percent_uncertainty_k0

# Covariance matrix (assuming no correlation for simplicity initially)
# This defines the shape and orientation. A diagonal matrix means axes align with parameter axes.
covariance_matrix = np.array([
    [std_dev_dh ** 2, 0],
    [0, std_dev_k0 ** 2]
])

# For a more general ellipsoid, you could introduce correlation:
# covariance_matrix = np.array([
#     [std_dev_dh**2, correlation * std_dev_dh * std_dev_k0],
#     [correlation * std_dev_dh * std_dev_k0, std_dev_k0**2]
# ])

# Choose a confidence level (e.g., 95%) and determine chi-squared value
# For 2 degrees of freedom (2 parameters), 95% confidence corresponds to chi2.ppf(0.95, 2)
confidence_level = 0.95
chi2_val = chi2.ppf(confidence_level, df=2)  # 2 degrees of freedom for 2 parameters

# 4. Sample Points within the Ellipsoid
num_samples = 500  # Number of parameter sets to sample

sampled_params = []
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
sqrt_eigenvalues = np.sqrt(eigenvalues)

# Generate random points in a unit circle and transform to the ellipsoid
for _ in range(num_samples):
    # Generate random points on a unit circle (polar coordinates)
    theta = 2 * np.pi * np.random.rand()
    r_rand = np.random.rand()  # Uniform random radius for points *inside* the ellipse

    # Scale by sqrt(chi2_val) for confidence level, and by random radius
    # For points uniformly *on* the boundary, r_rand would be 1.
    # For points uniformly *within* the ellipsoid, need to scale radius by sqrt of uniform random.
    # A standard way to sample uniformly within an ellipsoid is to sample from a unit ball
    # and then transform. Here, we're using a common approximation for illustration.

    # This method generates points within an ellipse defined by the covariance matrix, scaled by chi2_val.
    # Transform points from a unit circle to the ellipsoid
    # Generate random vector in a 2D unit sphere
    z = np.random.randn(2)
    # Normalize to get a point on the unit sphere, then scale by sqrt of a uniform random variate
    # for uniform sampling within the sphere.
    point_on_sphere = z / np.linalg.norm(z)
    u_rand = np.random.rand() ** (1 / 2)  # For 2D, power is 1/2
    scaled_point = u_rand * point_on_sphere

    # Transform to the ellipsoid using eigenvalues and eigenvectors
    # This creates a more accurate uniform sampling within the ellipsoid
    ellipsoid_point = mu + eigenvectors @ np.diag(sqrt_eigenvalues) @ scaled_point * np.sqrt(chi2_val)

    # Ensure sampled values are reasonable (e.g., k0 must be positive)
    if ellipsoid_point[1] > 0:  # k0 must be positive
        sampled_params.append(ellipsoid_point)

sampled_params = np.array(sampled_params)

# Visualize the sampled parameters (optional but good for debugging)
plt.figure(figsize=(8, 6))
plt.scatter(sampled_params[:, 0], sampled_params[:, 1], s=5, alpha=0.5, label='Sampled Parameters')
plt.scatter(mu[0], mu[1], color='red', marker='x', s=100, label='Nominal Values')
plt.xlabel('$\\Delta h$ (kJ/kmol)')
plt.ylabel('$k_0$ (m$^3$/(kmol$\\cdot$h))')
plt.title('Sampled Parameters within Ellipsoid')
plt.grid(True)
plt.legend()
plt.show()

# 5. Simulate the System for Each Sampled Parameter Set
all_cA_trajectories = []
all_T_trajectories = []

for i, (sampled_delta_h, sampled_k0) in enumerate(sampled_params):
    current_params = (V, F, T0, rho, Cp, sampled_k0, E, sampled_delta_h, R_gas)

    t_values, y_values = explicit_euler(cstr_model, initial_state, t_span, dt, current_params, c_A0_input, Q_input)

    all_cA_trajectories.append(y_values[:, 0])
    all_T_trajectories.append(y_values[:, 1])

all_cA_trajectories = np.array(all_cA_trajectories)
all_T_trajectories = np.array(all_T_trajectories)

# 6. Analyze and Plot the Results (Envelopes)
# Calculate min, max, and nominal trajectories
min_cA = np.min(all_cA_trajectories, axis=0)
max_cA = np.max(all_cA_trajectories, axis=0)
mean_cA = np.mean(all_cA_trajectories, axis=0)  # Often useful to plot mean as well

min_T = np.min(all_T_trajectories, axis=0)
max_T = np.max(all_T_trajectories, axis=0)
mean_T = np.mean(all_T_trajectories, axis=0)  # Often useful to plot mean as well

# Simulate with nominal parameters for comparison
_, nominal_y_values = explicit_euler(cstr_model, initial_state, t_span, dt, nominal_params_tuple, c_A0_input, Q_input)
nominal_cA = nominal_y_values[:, 0]
nominal_T = nominal_y_values[:, 1]

plt.figure(figsize=(12, 5))

# Plot c_A trajectories
plt.subplot(1, 2, 1)
plt.plot(t_values, nominal_cA, 'k--', linewidth=2, label='Nominal $c_A$')
plt.plot(t_values, min_cA, 'b-', label='Min $c_A$')
plt.plot(t_values, max_cA, 'r-', label='Max $c_A$')
plt.fill_between(t_values, min_cA, max_cA, color='gray', alpha=0.2, label='Uncertainty Envelope')
plt.xlabel('Time (h)')
plt.ylabel('Concentration $c_A$ (kmol/m$^3$)')
plt.title('Concentration $c_A$ with Parameter Uncertainty')
plt.legend()
plt.grid(True)

# Plot T trajectories
plt.subplot(1, 2, 2)
plt.plot(t_values, nominal_T, 'k--', linewidth=2, label='Nominal T')
plt.plot(t_values, min_T, 'b-', label='Min T')
plt.plot(t_values, max_T, 'r-', label='Max T')
plt.fill_between(t_values, min_T, max_T, color='gray', alpha=0.2, label='Uncertainty Envelope')
plt.xlabel('Time (h)')
plt.ylabel('Temperature T (K)')
plt.title('Temperature T with Parameter Uncertainty')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Additional analysis: Check if states stay within bounds (Table 2)
# You would need to add logic here to check if min_cA, max_cA, min_T, max_T
# go beyond the specified minimum/maximum values from Table 2.
# For example:
# cA_min_bound = 0.0 # kmol/m^3
# cA_max_bound = 7.5 # kmol/m^3
# T_min_bound = 300.0 # K
# T_max_bound = 500.0 # K
#
# if np.any(min_cA < cA_min_bound) or np.any(max_cA > cA_max_bound):
#     print("c_A uncertainty envelope exceeds defined bounds.")
# if np.any(min_T < T_min_bound) or np.any(max_T > T_max_bound):
#     print("T uncertainty envelope exceeds defined bounds.")
import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import fsolve
from scipy.linalg import expm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# --- 配置 Matplotlib 以支持中文显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 尝试SimHei，如果SimHei不可用则尝试Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
print("Matplotlib字体配置已更新，尝试使用SimHei/Arial Unicode MS支持中文。")


# --- CSTR 模型函数 ---
def cstr_model(state, inputs, params):
    """
    CSTR 模型微分方程。
    state = [c_A, T]
    inputs = [c_A0, Q]
    params = dict of model parameters (V, F, T0, rho, Cp, k0, E, delta_h, R_gas)
    """
    c_A, T = state
    c_A0, Q = inputs

    # 解包参数
    V = params['V']
    F = params['F']
    T0 = params['T0']
    rho = params['rho']
    Cp = params['Cp']
    k0 = params['k0']
    E = params['E']
    delta_h = params['delta_h']
    R_gas = params['R_gas']

    # 反应速率常数 (Arrhenius equation)
    k = k0 * np.exp(-E / (R_gas * T))

    # 微分方程
    d_cA_dt = (F / V) * (c_A0 - c_A) - k * c_A ** 2
    d_T_dt = (F / V) * (T0 - T) - (delta_h / (rho * Cp)) * k * c_A ** 2 + (Q / (rho * Cp * V))

    return np.array([d_cA_dt, d_T_dt])


# --- 模型参数 (来自 Table 1) ---
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


# --- 寻找稳态操作点 ---
def find_steady_state(inputs_ss, params_ss, initial_guess):
    """
    解稳态方程 f(x, u) = 0 来找到稳态点 x_ss。
    这里使用 fsolve 进行数值求解。
    接受一个 initial_guess 参数。
    """

    def equations_to_solve(state_ss):
        return cstr_model(state_ss, inputs_ss, params_ss)

    steady_state_x = fsolve(equations_to_solve, initial_guess)
    return steady_state_x


# 定义一个用于寻找稳态的输入
ss_inputs = np.array([5.0, 0.0])  # c_A0 = 5.0 kmol/m^3, Q = 0.0 kJ/h

# --- 符号化模型用于雅可比矩阵计算 ---
c_A_sym, T_sym = sympy.symbols('c_A T')
c_A0_sym, Q_sym = sympy.symbols('c_A0 Q')
delta_h_sym, k0_sym = sympy.symbols('delta_h k0')

# 其他参数视为常数符号
V_sym, F_sym, T0_sym, rho_sym, Cp_sym, E_sym, R_gas_sym = sympy.symbols(
    'V F T0 rho Cp E R_gas'
)

# 反应速率常数 (符号形式)
k_rate_sym = k0_sym * sympy.exp(-E_sym / (R_gas_sym * T_sym))

# 微分方程 (符号形式)
dc_A_dt_sym = (F_sym / V_sym) * (c_A0_sym - c_A_sym) - k_rate_sym * c_A_sym ** 2
dT_dt_sym = (F_sym / V_sym) * (T0_sym - T_sym) - (delta_h_sym / (rho_sym * Cp_sym)) * k_rate_sym * c_A_sym ** 2 + (
            Q_sym / (rho_sym * Cp_sym * V_sym))

f_sym = sympy.Matrix([dc_A_dt_sym, dT_dt_sym])
x_sym_vec = sympy.Matrix([c_A_sym, T_sym])
u_sym_vec = sympy.Matrix([c_A0_sym, Q_sym])
p_sym_vec = sympy.Matrix([delta_h_sym, k0_sym])

# 计算雅可比矩阵
A_sym_matrix = f_sym.jacobian(x_sym_vec)
B_sym_matrix = f_sym.jacobian(u_sym_vec)
G_sym_matrix = f_sym.jacobian(p_sym_vec)


# --- 数值化雅可比矩阵的函数 ---
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


# --- 探索不同的稳态点 ---
print("\n--- 探索不同的稳态点及稳定性 ---")

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
            print(f"\n--- 找到一个新的稳态点 (来自初始猜测 T={initial_T_guess:.2f}K) ---")
            print(f"稳态操作点 x_op (cA, T): {current_steady_state[0]:.4f} kmol/m^3, {current_steady_state[1]:.4f} K")
            print(f"在稳态点处的导数 (应接近零): {cstr_model(current_steady_state, ss_inputs, nominal_params)}")

            A_current, _, _ = get_jacobian_matrices(current_steady_state, ss_inputs, nominal_params)
            eigenvalues = np.linalg.eigvals(A_current)
            print(f"连续时间 A 矩阵的特征值: {eigenvalues}")

            if np.all(np.real(eigenvalues) < 0):
                print("此稳态点是：**稳定**的。")
            else:
                print("此稳态点是：**不稳定**的 (存在特征值实部非负)。")

    except Exception as e:
        print(f"\n尝试初始猜测 T={initial_T_guess:.2f}K 时求解稳态失败: {e}")

if not found_steady_states:
    print("\n未能找到任何稳态点，请检查模型参数或调整初始猜测范围。")
    exit()

# 从找到的稳定点中选择一个用于后续分析 (这里默认选择第一个，您可以根据稳定性分析结果手动修改)
stable_operating_points = []
for s_state in found_steady_states:
    A_temp, _, _ = get_jacobian_matrices(s_state, ss_inputs, nominal_params)
    eigenvalues_temp = np.linalg.eigvals(A_temp)
    if np.all(np.real(eigenvalues_temp) < 0):
        stable_operating_points.append(s_state)

if stable_operating_points:
    nominal_state = stable_operating_points[0]  # 选择第一个找到的稳定点作为名义操作点
    print(f"\n已选择第一个稳定的稳态点作为名义操作点: cA={nominal_state[0]:.4f}, T={nominal_state[1]:.4f}")
else:
    print("\n警告: 未找到任何稳定的稳态点。协方差传播可能仍会溢出。将使用第一个找到的稳态点进行后续分析。")
    nominal_state = found_steady_states[0]  # 如果没有稳定点，则使用第一个找到的（可能是）不稳定的点

nominal_inputs = ss_inputs

# --- 误差传播示例 (使用选定的名义操作点) ---
print("\n--- 一阶泰勒展开静态误差传播 (使用选定的名义操作点) ---")
A_op, B_op, G_op = get_jacobian_matrices(nominal_state, nominal_inputs, nominal_params)

print("在操作点处的 A 矩阵 (状态对状态的敏感度):\n", A_op)

delta_x0 = np.array([0.01, 0.5])
delta_u = np.array([0.02, 10.0])
delta_p = np.array([nominal_params['delta_h'] * 0.01, nominal_params['k0'] * 0.01])

delta_x_dot = A_op @ delta_x0 + B_op @ delta_u + G_op @ delta_p

print(f"\n初始状态误差 delta_x0: {delta_x0}")
print(f"输入误差 delta_u: {delta_u}")
print(f"参数误差 delta_p (Delta_h, k0): {delta_p}")
print(f"\n当前时刻状态导数 (dc_A/dt, dT/dt) 的估计误差 delta_x_dot:\n {delta_x_dot}")

# --- 协方差传播分析 (使用选定的名义操作点) ---

# --- 离散化 A 矩阵 ---
A_continuous = A_op

dt = 0.01
A_discrete = expm(A_continuous * dt)

print("\n--- 协方差传播分析 (使用选定的名义操作点) ---")
print("离散化后的 A 矩阵 (A_discrete):\n", A_discrete)

eigenvalues_discrete = np.linalg.eigvals(A_discrete)
print(f"离散化 A 矩阵的特征值: {eigenvalues_discrete}")
if np.all(np.abs(eigenvalues_discrete) < 1):
    print("离散化系统是稳定的 (所有特征值的模小于1)。")
else:
    print("警告: 离散化系统可能不稳定 (存在特征值模大于等于1)。如果出现溢出，请确保您选择了一个真正的稳定操作点。")

# --- 定义初始状态协方差 P0 ---
sigma_cA_initial = 0.05
sigma_T_initial = 2.0

P0 = np.array([
    [sigma_cA_initial ** 2, 0.0],
    [0.0, sigma_T_initial ** 2]
])
print("\n初始状态协方差 P0:\n", P0)

# --- 定义过程噪声协方差 Q ---
Q_process_noise = np.array([
    [5e-6, 0.0],
    [0.0, 5e-5]
])
print("\n过程噪声协方差 Q_process_noise:\n", Q_process_noise)

# --- 模拟时间 ---
time_horizon = 2.0  # 小时
num_steps = int(time_horizon / dt)
time_points = np.linspace(0, time_horizon, num_steps + 1)

# --- 协方差传播循环 ---
P_history = [P0]

for k in range(num_steps):
    P_k = P_history[-1]
    P_k_plus_1 = A_discrete @ P_k @ A_discrete.T + Q_process_noise
    P_history.append(P_k_plus_1)


# --- 绘制置信椭圆的辅助函数 ---
def plot_covariance_ellipse(ax, mean, cov, confidence_level=0.95, **kwargs):
    # 检查协方差矩阵是否有效 (正定)
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


# --- 状态空间中的协方差传播椭圆 (c_A vs T) ---
fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(nominal_state[0], nominal_state[1], 'ko', markersize=8, label='名义操作点')

times_to_plot = [0, num_steps // 4, num_steps // 2, num_steps * 3 // 4, num_steps]
colors = plt.cm.viridis(np.linspace(0, 1, len(times_to_plot)))

for i, idx in enumerate(times_to_plot):
    current_P = P_history[idx]
    current_time = time_points[idx]

    ell = plot_covariance_ellipse(ax, nominal_state, current_P, color=colors[i], alpha=0.6)
    if ell:
        ell.set_label(f't={current_time:.2f}h 95% 置信椭圆')

ax.set_xlabel('$c_A$ (kmol/m$^3$)')
ax.set_ylabel('T (K)')
ax.set_title('状态空间中的协方差传播：不确定性椭圆的演变')
ax.legend()
ax.grid(True)
ax.set_aspect('equal', adjustable='box')
plt.show()

# --- 绘制每个状态变量随时间的标准差 ---
valid_P_history_for_plot = [P for P in P_history if
                            np.all(np.isfinite(P)) and P[0, 0] > 0 and P[1, 1] > 0 and np.linalg.det(P) > 0]
valid_time_points_for_plot = time_points[:len(valid_P_history_for_plot)]

if len(valid_P_history_for_plot) > 0:
    std_dev_cA_history = np.array([np.sqrt(P[0, 0]) for P in valid_P_history_for_plot])
    std_dev_T_history = np.array([np.sqrt(P[1, 1]) for P in valid_P_history_for_plot])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(valid_time_points_for_plot, nominal_state[0] + std_dev_cA_history * 3, 'r--', label='名义 $\pm$ 3$\sigma$')
    plt.plot(valid_time_points_for_plot, nominal_state[0] - std_dev_cA_history * 3, 'r--')
    plt.plot(valid_time_points_for_plot, [nominal_state[0]] * len(valid_time_points_for_plot), 'k-', label='名义 $c_A$')
    plt.fill_between(valid_time_points_for_plot, nominal_state[0] - std_dev_cA_history * 3,
                     nominal_state[0] + std_dev_cA_history * 3,
                     color='blue', alpha=0.1, label='$\pm$ 3$\sigma$ 范围')
    plt.xlabel('时间 (h)');
    plt.ylabel('浓度 $c_A$ (kmol/m$^3$)');
    plt.title('$c_A$ 均值和 3$\sigma$ 不确定性范围')
    plt.legend();
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(valid_time_points_for_plot, nominal_state[1] + std_dev_T_history * 3, 'r--', label='名义 $\pm$ 3$\sigma$')
    plt.plot(valid_time_points_for_plot, nominal_state[1] - std_dev_T_history * 3, 'r--')
    plt.plot(valid_time_points_for_plot, [nominal_state[1]] * len(valid_time_points_for_plot), 'k-', label='名义 T')
    plt.fill_between(valid_time_points_for_plot, nominal_state[1] - std_dev_T_history * 3,
                     nominal_state[1] + std_dev_T_history * 3,
                     color='red', alpha=0.1, label='$\pm$ 3$\sigma$ 范围')
    plt.xlabel('时间 (h)');
    plt.ylabel('温度 T (K)');
    plt.title('T 均值和 3$\sigma$ 不确定性范围')
    plt.legend();
    plt.grid(True)
    plt.tight_layout();
    plt.show()
else:
    print("\n由于所有协方差矩阵都无效，无法绘制状态变量随时间的标准差图。这通常意味着系统不稳定，导致不确定性迅速发散。")

# --- 新增部分：在 (c_A0, T) 坐标系中绘制不确定性椭圆 ---

print("\n--- 在 (c_A0, T) 坐标系中绘制不确定性椭圆 ---")

# 1. 定义输入 (c_A0, Q) 的不确定性协方差矩阵
# 假设 c_A0 的标准差为 0.1 kmol/m^3，Q 的标准差为 50 kJ/h
# 假设它们之间不相关
sigma_cA0_input = 0.1
sigma_Q_input = 50.0

P_inputs_uncertainty = np.array([
    [sigma_cA0_input ** 2, 0.0],
    [0.0, sigma_Q_input ** 2]
])
print("\n输入 (c_A0, Q) 的不确定性协方差矩阵 P_inputs_uncertainty:\n", P_inputs_uncertainty)

# 2. 计算稳态敏感性矩阵 S_ux = - inv(A_op) @ B_op
# S_ux 矩阵将输入的不确定性传播到稳态状态的不确定性
try:
    inv_A_op = np.linalg.inv(A_op)
    S_ux = -inv_A_op @ B_op
    print("\n稳态敏感性矩阵 S_ux (delta_x_ss / delta_u):\n", S_ux)
except np.linalg.LinAlgError:
    print("\n警告: 矩阵 A_op 不可逆，无法计算稳态敏感性矩阵 S_ux。可能系统不稳定或线性化有问题。")
    S_ux = np.zeros((2, 2))  # 提供一个零矩阵避免后续错误，但结果将无意义

# 3. 构建 (c_A0, T_ss) 的联合协方差矩阵 P_cA0_Tss
if np.any(S_ux != 0):  # 仅当 S_ux 有效时进行计算
    # var(c_A0)
    var_cA0 = P_inputs_uncertainty[0, 0]

    # var(T_ss) (由 inputs 的不确定性传播而来)
    # P_x_ss = S_ux @ P_inputs_uncertainty @ S_ux.T
    # var_T_ss = P_x_ss[1,1]
    # 更直接地从 S_ux 和 P_inputs_uncertainty 计算 var_T_ss
    var_T_ss = (S_ux[1, 0] ** 2 * P_inputs_uncertainty[0, 0] +
                S_ux[1, 1] ** 2 * P_inputs_uncertainty[1, 1] +
                2 * S_ux[1, 0] * S_ux[1, 1] * P_inputs_uncertainty[0, 1])  # 考虑输入间的协方差，虽然这里是0

    # cov(c_A0, T_ss)
    # T_ss ≈ T_ss_nom + S_ux[1,0] * delta_c_A0 + S_ux[1,1] * delta_Q
    # cov(c_A0, T_ss) = E[delta_c_A0 * (S_ux[1,0] * delta_c_A0 + S_ux[1,1] * delta_Q)]
    # = S_ux[1,0] * E[delta_c_A0^2] + S_ux[1,1] * E[delta_c_A0 * delta_Q]
    # = S_ux[1,0] * var(c_A0) + S_ux[1,1] * cov(c_A0, Q)
    cov_cA0_Tss = S_ux[1, 0] * P_inputs_uncertainty[0, 0] + S_ux[1, 1] * P_inputs_uncertainty[0, 1]

    P_cA0_Tss = np.array([
        [var_cA0, cov_cA0_Tss],
        [cov_cA0_Tss, var_T_ss]
    ])
    print("\n(c_A0, T_ss) 的联合协方差矩阵 P_cA0_Tss:\n", P_cA0_Tss)

    # 4. 绘制椭圆
    fig_cA0_T, ax_cA0_T = plt.subplots(figsize=(9, 7))

    # 椭圆的中心点是 (名义 c_A0, 名义稳态 T)
    ellipse_center_cA0_T = np.array([nominal_inputs[0], nominal_state[1]])

    ell_cA0_T = plot_covariance_ellipse(ax_cA0_T, ellipse_center_cA0_T, P_cA0_Tss, color='blue', alpha=0.7,
                                        label='95% 置信椭圆')

    if ell_cA0_T:
        ax_cA0_T.plot(ellipse_center_cA0_T[0], ellipse_center_cA0_T[1], 'ro', markersize=8,
                      label='名义操作点 (c_A0, T)')
        ax_cA0_T.set_xlabel('进料浓度 $c_{A0}$ (kmol/m$^3$)')
        ax_cA0_T.set_ylabel('反应器温度 T (K)')
        ax_cA0_T.set_title('在 (c_A0, T) 坐标系中由输入不确定性引起的稳态不确定性椭圆')
        ax_cA0_T.legend()
        ax_cA0_T.grid(True)
        ax_cA0_T.set_aspect('auto')  # 使用auto而不是equal，因为坐标轴单位不同
        plt.show()
    else:
        print("无法在 (c_A0, T) 坐标系中绘制椭圆，可能协方差矩阵无效。")
else:
    print("\n由于稳态敏感性矩阵 S_ux 无效，无法在 (c_A0, T) 坐标系中绘制椭圆。请检查系统稳定性。")
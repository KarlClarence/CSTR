import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 从Table 1和Table 2定义的参数和范围
PARAMS = {
    'V': 1.0,  # m³
    'F': 5.0,  # m³/h
    'T0': 300.0,  # K
    'rho': 1000.0,  # kg/m³
    'cp': 0.231,  # kJ/(kg·K)
    'k0': 8.46e6,  # m³/(kmol·h)
    'Ea': 5.0e4,  # kJ/kmol
    'delta_h': -1.15e4,  # kJ/kmol
    'R': 8.3145  # kJ/(kmol·K)
}

VARIABLE_RANGES = {
    'cA': (0, 7.5),  # kmol/m³
    'T': (300, 500),  # K
    'cA0': (0.5, 7.5),  # kmol/m³
    'Q': (-5e5, 5e5)  # kJ/h
}


def cstr_model(t, y, cA0, Q):
    """CSTR动态模型"""
    cA, T = y
    k = PARAMS['k0'] * np.exp(-PARAMS['Ea'] / (PARAMS['R'] * T))
    dcA_dt = (PARAMS['F'] / PARAMS['V']) * (cA0 - cA) - k * cA ** 2
    dT_dt = (PARAMS['F'] / PARAMS['V']) * (PARAMS['T0'] - T) - \
            (PARAMS['delta_h'] / (PARAMS['rho'] * PARAMS['cp'])) * k * cA ** 2 + \
            Q / (PARAMS['rho'] * PARAMS['cp'] * PARAMS['V'])
    return [dcA_dt, dT_dt]


def run_simulation(cA_init, T_init, cA0, Q, t_end=2):
    """运行单次模拟"""
    t_eval = np.linspace(0, t_end, 100)
    sol = solve_ivp(cstr_model, (0, t_end), [cA_init, T_init],
                    args=(cA0, Q), t_eval=t_eval, method='LSODA')
    return sol.t, sol.y[0], sol.y[1]


def plot_comparison(simulations):
    """绘制多组模拟结果的比较图"""
    plt.figure(figsize=(14, 6))

    # 浓度曲线
    plt.subplot(1, 2, 1)
    for i, (t, cA, T, label) in enumerate(simulations):
        plt.plot(t, cA, label=label, linewidth=2,
                 color=plt.cm.tab10(i))
    plt.xlabel('Time (h)')
    plt.ylabel('Concentration (kmol/m³)')
    plt.title('Concentration Comparison')
    plt.grid(True)
    plt.legend()

    # 温度曲线
    plt.subplot(1, 2, 2)
    for i, (t, cA, T, label) in enumerate(simulations):
        plt.plot(t, T, label=label, linewidth=2,
                 color=plt.cm.tab10(i))
    plt.xlabel('Time (h)')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Comparison')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('cstr_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # 定义多组模拟条件（每组：cA_init, T_init, cA0, Q, label）
    simulation_cases = [
        # 基准案例（中等条件）
        (2.0, 350, 2.5, 1e5, "Baseline"),
        # 高浓度案例
        (5.0, 350, 5.0, 1e5, "High cA"),
        # 低温案例
        (2.0, 320, 2.5, -1e5, "Low T with cooling"),
        # 高反应活性案例
        (1.0, 400, 1.5, 3e5, "High reactivity"),
        # 极限操作案例
        (0.5, 450, 7.5, 5e5, "Extreme operation")
    ]

    # 运行所有模拟
    results = []
    for case in simulation_cases:
        cA_init, T_init, cA0, Q, label = case

        # 检查变量范围
        assert VARIABLE_RANGES['cA'][0] <= cA_init <= VARIABLE_RANGES['cA'][1]
        assert VARIABLE_RANGES['T'][0] <= T_init <= VARIABLE_RANGES['T'][1]
        assert VARIABLE_RANGES['cA0'][0] <= cA0 <= VARIABLE_RANGES['cA0'][1]
        assert VARIABLE_RANGES['Q'][0] <= Q <= VARIABLE_RANGES['Q'][1]

        t, cA, T = run_simulation(cA_init, T_init, cA0, Q)
        results.append((t, cA, T, label))

    # 绘制比较图
    plot_comparison(results)

    # 打印稳态值
    print("\n稳态结果比较:")
    print(f"{'Case':<20} {'Final cA':<10} {'Final T':<10}")
    for (_, cA, T, label) in results:
        print(f"{label:<20} {cA[-1]:<10.3f} {T[-1]:<10.1f}")
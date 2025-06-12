import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


def load_parameters(file_path):
    """从JSON文件加载参数"""
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params


def cstr_model(t, y, params):
    """修改后的CSTR动态模型，接受直接参数输入"""
    cA, T = y

    # 解包参数
    V = params['V']
    F = params['F']
    T0 = params['T0']
    rho = params['rho']
    cp = params['cp']
    k0 = params['k0']
    Ea = params['Ea']
    delta_h = params['delta_h']
    R = params['R']
    cA0 = params['cA0']
    Q = params['Q']

    k = k0 * np.exp(-Ea / (R * T))

    # 动态方程
    dcA_dt = (F / V) * (cA0 - cA) - k * cA ** 2
    dT_dt = (F / V) * (T0 - T) - (delta_h / (rho * cp)) * k * cA ** 2 + Q / (rho * cp * V)

    return [dcA_dt, dT_dt]


def run_single_simulation(params, t_end=2, num_points=100):
    """运行单次模拟"""
    y0 = [params['cA_init'], params['T_init']]
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, num_points)

    sol = solve_ivp(cstr_model, t_span, y0, args=(params,),
                    t_eval=t_eval, method='LSODA')

    return sol.t, sol.y[0], sol.y[1]


def sensitivity_analysis(base_params_file, output_file='sensitivity_results.csv'):
    """执行灵敏度分析"""
    # 加载基础参数
    base_params = load_parameters(base_params_file)

    # 创建参数范围
    k0_base = base_params['model_parameters']['k0']
    delta_h_base = base_params['model_parameters']['delta_h']

    # 设置参数变化范围 (±20%)
    variations = np.linspace(0.8, 1.2, 5)  # 80%到120%，5个点

    results = []

    # 准备进度条
    total_simulations = len(variations) ** 2
    progress_bar = tqdm(total=total_simulations, desc="进行灵敏度分析")

    # 参数扫描
    for k0_var in variations:
        for dh_var in variations:
            # 创建当前参数组合
            current_params = {
                **base_params['model_parameters'],
                **base_params['inputs'],
                'cA_init': base_params['initial_state']['cA'],
                'T_init': base_params['initial_state']['T'],
                'k0': k0_base * k0_var,
                'delta_h': delta_h_base * dh_var
            }

            # 运行模拟
            t, cA, T = run_single_simulation(current_params)

            # 记录稳态结果（取最后10%时间点的平均值）
            steady_state_idx = int(len(t) * 0.9)
            cA_steady = np.mean(cA[steady_state_idx:])
            T_steady = np.mean(T[steady_state_idx:])

            # 保存结果
            results.append({
                'k0_variation': k0_var,
                'delta_h_variation': dh_var,
                'k0_actual': current_params['k0'],
                'delta_h_actual': current_params['delta_h'],
                'cA_steady': cA_steady,
                'T_steady': T_steady
            })

            progress_bar.update(1)

    progress_bar.close()

    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    # 生成灵敏度图
    plot_sensitivity_results(df)

    return df


def plot_sensitivity_results(df):
    """绘制灵敏度分析结果"""
    plt.figure(figsize=(15, 6))

    # 浓度灵敏度
    plt.subplot(1, 2, 1)
    pivot_cA = df.pivot(index='k0_variation', columns='delta_h_variation', values='cA_steady')
    sns.heatmap(pivot_cA, annot=True, fmt=".3f", cmap="YlOrRd",
                cbar_kws={'label': 'Steady-state cA (kmol/m^3)'})
    plt.title('Steady-state Concentration Sensitivity')
    plt.xlabel('Δh Variation')
    plt.ylabel('k0 Variation')

    # 温度灵敏度
    plt.subplot(1, 2, 2)
    pivot_T = df.pivot(index='k0_variation', columns='delta_h_variation', values='T_steady')
    sns.heatmap(pivot_T, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'Steady-state T (K)'})
    plt.title('Steady-state Temperature Sensitivity')
    plt.xlabel('Δh Variation')
    plt.ylabel('k0 Variation')

    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png')
    plt.close()

    print("灵敏度分析图已保存为 sensitivity_analysis.png")


if __name__ == "__main__":
    # 执行灵敏度分析
    results_df = sensitivity_analysis('cstr_params.json')

    # 计算灵敏度指标
    k0_sensitivity_cA = results_df.groupby('k0_variation')['cA_steady'].mean().std()
    dh_sensitivity_cA = results_df.groupby('delta_h_variation')['cA_steady'].mean().std()

    k0_sensitivity_T = results_df.groupby('k0_variation')['T_steady'].mean().std()
    dh_sensitivity_T = results_df.groupby('delta_h_variation')['T_steady'].mean().std()

    print("\n灵敏度指标:")
    print(f"k0 对浓度的影响程度: {k0_sensitivity_cA:.4f}")
    print(f"Δh 对浓度的影响程度: {dh_sensitivity_cA:.4f}")
    print(f"k0 对温度的影响程度: {k0_sensitivity_T:.1f} K")
    print(f"Δh 对温度的影响程度: {dh_sensitivity_T:.1f} K")
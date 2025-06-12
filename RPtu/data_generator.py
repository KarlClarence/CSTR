import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm
import os

# 从Table 2中定义的变量范围
VARIABLE_RANGES = {
    'cA': (0, 7.5),  # kmol/m³ (反应器浓度)
    'T': (300, 500),  # K (反应器温度)
    'cA0': (0.5, 7.5),  # kmol/m³ (进料浓度)
    'Q': (-5e5, 5e5)  # kJ/h (加热速率)
}


def cstr_model(t, y, params):
    """CSTR动态模型"""
    cA, T = y
    k = params['k0'] * np.exp(-params['Ea'] / (params['R'] * T))
    dcA_dt = (params['F'] / params['V']) * (params['cA0'] - cA) - k * cA ** 2
    dT_dt = (params['F'] / params['V']) * (params['T0'] - T) - (
                params['delta_h'] / (params['rho'] * params['cp'])) * k * cA ** 2 + params['Q'] / (
                        params['rho'] * params['cp'] * params['V'])
    return [dcA_dt, dT_dt]


def generate_dataset(num_samples=5000, save_path='data/cstr_dataset.csv'):
    """生成训练数据集（严格遵循变量范围）"""
    # 加载基础参数
    with open('cstr_params.json') as f:
        base_params = json.load(f)

    # 准备数据容器
    data = []
    rng = np.random.default_rng()

    for _ in tqdm(range(num_samples), desc="Generating data"):
        # 在指定范围内随机生成关键变量
        cA_init = rng.uniform(*VARIABLE_RANGES['cA'])
        T_init = rng.uniform(*VARIABLE_RANGES['T'])
        cA0 = rng.uniform(*VARIABLE_RANGES['cA0'])
        Q = rng.uniform(*VARIABLE_RANGES['Q'])

        # 固定其他参数（或小幅扰动）
        current_params = {
            **base_params['model_parameters'],
            'cA0': cA0,
            'Q': Q,
            'F': base_params['model_parameters']['F'],
            'V': base_params['model_parameters']['V'],
            'T0': base_params['model_parameters']['T0'],
            'rho': base_params['model_parameters']['rho'],
            'cp': base_params['model_parameters']['cp'],
            'k0': base_params['model_parameters']['k0'] * rng.uniform(0.9, 1.1),  # ±10%扰动
            'Ea': base_params['model_parameters']['Ea'],
            'delta_h': base_params['model_parameters']['delta_h'] * rng.uniform(0.9, 1.1),
            'R': base_params['model_parameters']['R']
        }

        # 运行模拟（捕获可能的数值不稳定）
        try:
            sol = solve_ivp(cstr_model, (0, 2), [cA_init, T_init],
                            args=(current_params,), t_eval=np.linspace(0, 2, 100))

            # 检查最终值是否在合理范围内
            final_cA = sol.y[0][-1]
            final_T = sol.y[1][-1]

            if (VARIABLE_RANGES['cA'][0] <= final_cA <= VARIABLE_RANGES['cA'][1] and
                    VARIABLE_RANGES['T'][0] <= final_T <= VARIABLE_RANGES['T'][1]):
                data.append([
                    current_params['k0'],
                    current_params['delta_h'],
                    Q,  # 加热速率
                    cA0,  # 进料浓度
                    cA_init,
                    T_init,
                    final_cA,
                    final_T
                ])
        except:
            continue

    # 保存数据集
    cols = [
        'k0', 'delta_h', 'Q', 'cA0_input',
        'cA_init', 'T_init',
        'cA_final', 'T_final'
    ]
    df = pd.DataFrame(data, columns=cols)
    os.makedirs('data', exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"数据集已保存到 {save_path}，包含 {len(df)} 个有效样本")
    print("\n生成的变量范围验证:")
    print(df[['cA_init', 'T_init', 'cA0_input', 'Q']].describe().loc[['min', 'max']])


if __name__ == "__main__":
    generate_dataset(num_samples=10000)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# ====================== 1. CSTR 动态模型 ======================
def cstr_dynamics(y, t, F, V, cA0, T0, rho, cp, k0, Ea, delta_h, R, Q):
    cA, T = y
    rate = k0 * np.exp(-Ea / (R * T)) * cA ** 2
    dcA_dt = (F / V) * (cA0 - cA) - rate
    dT_dt = (F / V) * (T0 - T) - (delta_h / (rho * cp)) * rate + Q / (rho * cp * V)
    return [dcA_dt, dT_dt]


# ====================== 2. 神经网络控制器 ======================
class Q_Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),  # 输入: [cA, T, cA_set, T_set]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, state, setpoint):
        x = torch.cat([state, setpoint], dim=-1)
        return self.net(x) * 5e5  # 缩放到 [-5e5, 5e5]


# ====================== 3. 数据生成与训练 ======================
def generate_data():
    params = {
        'V': 1.0, 'F': 5.0, 'cA0': 2.0, 'T0': 300.0, 'rho': 1000.0,
        'cp': 0.231, 'k0': 8.46e6, 'Ea': 5.0e4, 'delta_h': -1.15e4, 'R': 8.3145
    }

    states, setpoints, Qs = [], [], []
    for _ in range(1000):
        cA_set = np.random.uniform(0.5, 7.5)
        T_set = np.random.uniform(350, 450)
        cA = np.random.uniform(0.1, 7.0)
        T = np.random.uniform(320, 480)

        # 固定Q模拟动态（简化数据生成）
        Q = np.random.uniform(-5e5, 5e5)
        t = np.linspace(0, 1, 10)
        sol = odeint(cstr_dynamics, [cA, T], t,
                     args=(params['F'], params['V'], params['cA0'], params['T0'],
                           params['rho'], params['cp'], params['k0'], params['Ea'],
                           params['delta_h'], params['R'], Q))

        # 目标Q：假设稳态时Q应与设定点相关（简化逻辑）
        Q_target = 1e5 * (cA_set - sol[-1, 0]) + 1e3 * (T_set - sol[-1, 1])
        Q_target = np.clip(Q_target, -5e5, 5e5)

        states.append([cA, T])
        setpoints.append([cA_set, T_set])
        Qs.append([Q_target])

    return torch.FloatTensor(states), torch.FloatTensor(setpoints), torch.FloatTensor(Qs)


def train_model():
    states, setpoints, Qs = generate_data()
    model = Q_Controller()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(500):
        Q_pred = model(states, setpoints)
        loss = criterion(Q_pred, Qs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.2f}')

    return model


# ====================== 4. 模拟控制测试 ======================
def simulate_control(model):
    params = {
        'F': 5.0, 'V': 1.0, 'cA0': 2.0, 'T0': 300.0, 'rho': 1000.0,
        'cp': 0.231, 'k0': 8.46e6, 'Ea': 5.0e4, 'delta_h': -1.15e4, 'R': 8.3145
    }
    cA_set, T_set = 1.0, 350.0  # 设定值
    y0 = [1.5, 400]  # 初始状态

    t = np.linspace(0, 2, 100)
    cA_history, T_history, Q_history = [], [], []

    for ti in t:
        state = torch.FloatTensor([y0])
        setpoint = torch.FloatTensor([[cA_set, T_set]])
        with torch.no_grad():
            Q = model(state, setpoint).item()

        # 使用odeint更新状态（显式传递所有参数）
        y = odeint(cstr_dynamics, y0, [0, ti],
                   args=(params['F'], params['V'], params['cA0'], params['T0'],
                         params['rho'], params['cp'], params['k0'], params['Ea'],
                         params['delta_h'], params['R'], Q))
        y0 = y[-1]

        cA_history.append(y0[0])
        T_history.append(y0[1])
        Q_history.append(Q)

    # 绘图
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(t, cA_history, label='cA')
    plt.axhline(cA_set, color='r', linestyle='--', label='Setpoint')
    plt.legend()

    plt.subplot(132)
    plt.plot(t, T_history, label='T')
    plt.axhline(T_set, color='r', linestyle='--')
    plt.legend()

    plt.subplot(133)
    plt.plot(t, Q_history, label='Q')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ====================== 主程序 ======================
if __name__ == "__main__":
    model = train_model()
    simulate_control(model)
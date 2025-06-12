import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def load_parameters(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params


def cstr_model(t, y, params):
    cA, T = y
    model_params = params['model_parameters']
    inputs = params['inputs']

    V = model_params['V']
    F = model_params['F']
    T0 = model_params['T0']
    rho = model_params['rho']
    cp = model_params['cp']
    k0 = model_params['k0']
    Ea = model_params['Ea']
    delta_h = model_params['delta_h']
    R = model_params['R']

    cA0 = inputs['cA0']
    Q = inputs['Q']

    k = k0 * np.exp(-Ea / (R * T))

    dcA_dt = (F / V) * (cA0 - cA) - k * cA ** 2

    dT_dt = (F / V) * (T0 - T) - (delta_h / (rho * cp)) * k * cA ** 2 + Q / (rho * cp * V)

    return [dcA_dt, dT_dt]


def run_simulation(params_file):

    params = load_parameters(params_file)

    y0 = [params['initial_state']['cA'], params['initial_state']['T']]

    t_span = (params['simulation_settings']['t_start'],
              params['simulation_settings']['t_end'])
    t_eval = np.linspace(t_span[0], t_span[1], params['simulation_settings']['num_points'])

    sol = solve_ivp(cstr_model, t_span, y0, args=(params,),
                    t_eval=t_eval, method='LSODA')

    results = pd.DataFrame({
        'Time (h)': sol.t,
        'Concentration (kmol/m^3)': sol.y[0],
        'Temperature (K)': sol.y[1]
    })
    results.to_csv('cstr_simulation_results.csv', index=False)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], 'b-')
    plt.xlabel('Time (h)')
    plt.ylabel('Concentration (kmol/m^3)')
    plt.title('Concentration of A vs Time')

    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[1], 'r-')
    plt.xlabel('Time (h)')
    plt.ylabel('Temperature (K)')
    plt.title('Reactor Temperature vs Time')

    plt.tight_layout()
    plt.savefig('cstr_simulation_plot.png')
    plt.close()

    print("Finished, results have been saved to cstr_simulation_results.csv 和 cstr_simulation_plot.png")


if __name__ == "__main__":
    run_simulation('cstr_params.json')
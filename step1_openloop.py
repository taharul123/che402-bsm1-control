"""
step1_openloop.py
Runs BSM1 open-loop (no control) under dry, rain, and storm weather conditions.
Plots SO, SNH, SNO across all 5 reactor tanks for each condition and saves
the figures as PNG files you can open and put in your report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # so it saves to file instead of trying to pop up a window
import matplotlib.pyplot as plt

from bsm2_python import BSM1OL
from common import load_influent, COMP

WEATHER_FILES = {
    'dry': 'dry_data.txt',
    'rain': 'rain_data.txt',
    'storm': 'storm_data.txt',
}

TANK_ATTRS = ['y_out1_all', 'y_out2_all', 'y_out3_all', 'y_out4_all', 'y_out5_all']
VARS_TO_PLOT = ['SO', 'SNH', 'SNO']


def run_weather(name, filepath):
    print(f'\n=== Running open-loop BSM1 under {name} weather ===')
    data_in = load_influent(filepath)

    bsm1 = BSM1OL(data_in=data_in)
    bsm1.simulate(plot=False)

    for var in VARS_TO_PLOT:
        idx = COMP[var]
        plt.figure(figsize=(10, 5))
        for tank_num, attr in enumerate(TANK_ATTRS, start=1):
            tank_data_all = getattr(bsm1, attr)
            plt.plot(bsm1.simtime, tank_data_all[:, idx], label=f'Tank {tank_num}')
        plt.xlabel('Time [days]')
        plt.ylabel(f'{var} [g/m3]')
        plt.title(f'{var} across all 5 tanks — {name} weather (open loop)')
        plt.legend()
        plt.tight_layout()
        outname = f'step1_{name}_{var}.png'
        plt.savefig(outname, dpi=150)
        plt.close()
        print(f'  saved {outname}')

    return bsm1


if __name__ == '__main__':
    results = {}
    for name, filepath in WEATHER_FILES.items():
        results[name] = run_weather(name, filepath)

    print('\nAll done. Check the .png files in the file explorer.')
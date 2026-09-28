"""
step1_openloop.py  -  Step 1: understand the plant (open loop, no control)

Cases: constant influent, dry, rain, storm.
Every case follows the BSM1 initialization procedure (see common.py).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (COMP, TIMESTEP, build_constant_data, build_weather_data,
                    make_plant, stabilize_constant, run_open_loop)

TANK_ARRAYS = ['y_out1_all', 'y_out2_all', 'y_out3_all', 'y_out4_all', 'y_out5_all']
VARS = ['SO', 'SNH', 'SNO']


def plot_tanks(plant, label, plot_from, mark_eval):
    n = len(plant.timesteps)                    # the last time point is never simulated
    t = plant.simtime[:n] - plot_from
    keep = t >= 0
    for var in VARS:
        plt.figure(figsize=(10, 5))
        for k, name in enumerate(TANK_ARRAYS, start=1):
            plt.plot(t[keep], getattr(plant, name)[:n][keep, COMP[var]], label=f'Tank {k}')
        if mark_eval:
            plt.axvline(7, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Time [days]')
        plt.ylabel(f'{var} [g/m3]')
        plt.title(f'{var} in all five tanks - {label} (open loop)')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'step1_{label}_{var}.png', dpi=150)
        plt.close()


def run_case(label, data_in, eval_start, plot_from, mark_eval):
    print(f'\n=== {label} ===')
    plant = make_plant(data_in, (eval_start, None))
    rec = stabilize_constant(plant)                 # 100 days on the constant influent
    print(f'  after 100-day stabilization: SO5={plant.y_out5[COMP["SO"]]:.3f}  '
          f'SNO5={plant.y_out5[COMP["SNO"]]:.3f}  SNH5={plant.y_out5[COMP["SNH"]]:.3f}')
    run_open_loop(plant)
    plot_tanks(plant, label, plot_from, mark_eval)
    iqi, eqi, me, pe, ae = plant.get_final_performance()
    viol = plant.get_violations()['SNH']
    print(f'  IQI={iqi:.1f}  EQI={eqi:.1f}  AE={ae:.2f}  PE={pe:.2f}  ME={me:.2f}  SNH violation={viol:.3f} d')
    return rec, dict(iqi=iqi, eqi=eqi, ae=ae, pe=pe, me=me, viol=viol)


if __name__ == '__main__':
    results = {}

    # constant influent: 100-day stabilization is plotted too (supervisor step 7)
    rec, results['constant'] = run_case('constant', build_constant_data(14), 7.0, 0.0, False)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for a, col, nm in zip(ax, (1, 2, 3), ('SO tank 5 [g/m3]', 'SNO tank 2 [g/m3]', 'SNH tank 5 [g/m3]')):
        a.plot(rec[:, 0], rec[:, col]); a.set_ylabel(nm)
    ax[0].set_title('100-day stabilization on constant influent')
    ax[2].set_xlabel('Time [days]')
    plt.tight_layout(); plt.savefig('step1_stabilization_100d.png', dpi=150); plt.close()

    for label, fname in (('dry', 'dry_data.txt'), ('rain', 'rain_data.txt'), ('storm', 'storm_data.txt')):
        _, results[label] = run_case(label, build_weather_data(fname), 21.0, 14.0, True)

    print('\n=== Summary (evaluation = days 7-14 of each weather file) ===')
    for k, r in results.items():
        print(f'{k:9s} EQI={r["eqi"]:8.1f}  AE={r["ae"]:7.2f}  PE={r["pe"]:6.2f}  ME={r["me"]:5.1f}  SNH violation={r["viol"]:.3f} d')
    print('BSM1 report, open-loop dry weather: EQI=6690.1  AE=3341.39  PE=388.17  ME=240  SNH violation=4.375 d')
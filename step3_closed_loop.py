"""
step3_closed_loop.py
Step 3 & 4 of the project: close the loop, simulate, and compare two controller tunings.

Two independent PI controllers:
  1. DO controller: measures SO in tank 5, manipulates kLa in tank 5, setpoint 2 g/m3.
  2. Nitrate controller: measures SNO in tank 2, manipulates internal recycle flow, setpoint 1 g/m3.

This script runs the closed-loop plant TWICE under the same weather file, once for each
named tuning set below, and prints/plots both so they can be compared directly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bsm2_python import BSM1OL
from common import load_influent, COMP
from pi_controller import PIController, simc_tuning

TIMESTEP = 1 / 1440  # 1 minute, in days

SO_IDX = COMP['SO']
SNO_IDX = COMP['SNO']

SP_SO = 2.0   # g/m3, setpoint for DO in tank 5
SP_SNO = 1.0  # g/m3, setpoint for nitrate in tank 2

# ---- FOPDT parameters from Step 2 ----
KLA_K, KLA_TAU, KLA_THETA = 0.03776, 0.2417, 0.0001
QINTR_K, QINTR_TAU, QINTR_THETA = 0.00019, 0.0625, 0.1000

# ---- actuator limits ----
KLA_MIN, KLA_MAX = 0.0, 240.0
QINTR_MIN, QINTR_MAX = 0.0, 200000.0

# ---- the two tuning sets we are comparing ----
kp_kla_aggr, ti_kla_aggr = simc_tuning(KLA_K, KLA_TAU, KLA_THETA, tc=KLA_TAU)        # Set 1: faster (Tc = tau)
kp_kla_soft, ti_kla_soft = simc_tuning(KLA_K, KLA_TAU, KLA_THETA, tc=3 * KLA_TAU)    # Set 2: gentler (Tc = 3*tau)

kp_qintr_aggr, ti_qintr_aggr = simc_tuning(QINTR_K, QINTR_TAU, QINTR_THETA)          # nitrate loop: keep as-is for both sets

TUNING_SETS = {
    'Set1 (initial SIMC)': dict(
        kp_kla=kp_kla_aggr, ti_kla=ti_kla_aggr,
        kp_qintr=kp_qintr_aggr, ti_qintr=ti_qintr_aggr,
    ),
    'Set2 (detuned DO loop)': dict(
        kp_kla=kp_kla_soft, ti_kla=ti_kla_soft,
        kp_qintr=kp_qintr_aggr, ti_qintr=ti_qintr_aggr,
    ),
}


def run_closed_loop(weather_file, weather_label, tuning_name, tuning):
    print(f'\n=== {tuning_name} | {weather_label} weather ===')
    print(f'  DO controller:      Kp={tuning["kp_kla"]:.4f}  Ti={tuning["ti_kla"]:.4f} d')
    print(f'  Nitrate controller: Kp={tuning["kp_qintr"]:.4f}  Ti={tuning["ti_qintr"]:.4f} d')

    data_in = load_influent(weather_file)
    bsm1 = BSM1OL(data_in=data_in, timestep=TIMESTEP)
    bsm1.stabilize(atol=1e-5)

    do_ctrl = PIController(tuning['kp_kla'], tuning['ti_kla'], TIMESTEP,
                            KLA_MIN, KLA_MAX, bias=bsm1.klas[4])
    no_ctrl = PIController(tuning['kp_qintr'], tuning['ti_qintr'], TIMESTEP,
                            QINTR_MIN, QINTR_MAX, bias=bsm1.qintr)

    n_steps = len(bsm1.timesteps)
    sim_t = bsm1.simtime[:n_steps]

    so_response = np.zeros(n_steps)
    sno_response = np.zeros(n_steps)
    kla_signal = np.zeros(n_steps)
    qintr_signal = np.zeros(n_steps)
    so_error = np.zeros(n_steps)
    sno_error = np.zeros(n_steps)

    so_current = bsm1.y_out5[SO_IDX]
    sno_current = bsm1.y_out2[SNO_IDX]

    for i, t in enumerate(sim_t):
        u_kla, err_so = do_ctrl.update(SP_SO, so_current)
        u_qintr, err_sno = no_ctrl.update(SP_SNO, sno_current)

        klas_arr = bsm1.klas.copy()
        klas_arr[4] = u_kla
        bsm1.qintr = u_qintr
        bsm1.step(i, klas_arr)
        if i in (0, 1, 100, 5000):
            print(f'    [i={i}] we set klas[4]={u_kla:.3f}, resulting SO={bsm1.y_out5_all[i, SO_IDX]:.4f}')

        so_current = bsm1.y_out5_all[i, SO_IDX]
        sno_current = bsm1.y_out2_all[i, SNO_IDX]

        so_response[i] = so_current
        sno_response[i] = sno_current
        kla_signal[i] = u_kla
        qintr_signal[i] = u_qintr
        so_error[i] = err_so
        sno_error[i] = err_sno

    print(f'  DEBUG checksum: so_response.sum()={so_response.sum():.6f}  kla_signal.sum()={kla_signal.sum():.6f}')
    
    bsm1.finish_evaluation(plot=False)
    iqi_eval, eqi_eval, mixingenergy, pumpingenergy, aerationenergy = bsm1.get_final_performance()
    violations = bsm1.get_violations()

    iae_so = np.sum(np.abs(so_error)) * TIMESTEP
    iae_so = np.sum(np.abs(so_error)) * TIMESTEP
    iae_sno = np.sum(np.abs(sno_error)) * TIMESTEP
    ise_so = np.sum(so_error ** 2) * TIMESTEP
    print(f'  IAE (DO tracking error)      = {iae_so:.4f}')
    print(f'  ISE (DO tracking error)      = {ise_so:.4f}')
    print(f'  IAE (nitrate tracking error) = {iae_sno:.4f}')

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(sim_t, so_response, label='SO tank 5 (actual)')
    axes[0].axhline(SP_SO, color='r', linestyle='--', label='setpoint')
    axes[0].set_ylabel('SO [g/m3]')
    axes[0].set_title(f'{tuning_name} — DO control — {weather_label} weather')
    axes[0].legend()

    axes[1].plot(sim_t, kla_signal)
    axes[1].set_ylabel('kLa tank 5 [1/d]')

    axes[2].plot(sim_t, sno_response, label='SNO tank 2 (actual)')
    axes[2].axhline(SP_SNO, color='r', linestyle='--', label='setpoint')
    axes[2].set_ylabel('SNO [g/m3]')
    axes[2].set_title(f'{tuning_name} — Nitrate control — {weather_label} weather')
    axes[2].legend()

    axes[3].plot(sim_t, qintr_signal)
    axes[3].set_ylabel('Qintr [m3/d]')
    axes[3].set_xlabel('Time [days]')

    plt.tight_layout()
    safe_name = tuning_name.split()[0]
    outname = f'step3_{safe_name}_{weather_label}.png'
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f'  saved {outname}')

    print(f'  IQI={iqi_eval:.2f}  EQI={eqi_eval:.2f}  '
          f'AerationEnergy={aerationenergy:.2f}  PumpingEnergy={pumpingenergy:.2f}  MixingEnergy={mixingenergy:.2f}')
    print(f'  Effluent violations (SNH > 4 g/m3): {violations}')

    return {'iae_so': iae_so, 'ise_so': ise_so, 'iae_sno': iae_sno,
            'eqi': eqi_eval, 'aeration_energy': aerationenergy,
            'pumping_energy': pumpingenergy, 'mixing_energy': mixingenergy,
            'violations': violations}


WEATHER_FILES = {
    'dry': 'dry_data.txt',
    'rain': 'rain_data.txt',
    'storm': 'storm_data.txt',
}

if __name__ == '__main__':
    all_results = {}
    for weather_label, weather_file in WEATHER_FILES.items():
        for tuning_name, tuning in TUNING_SETS.items():
            key = (weather_label, tuning_name)
            all_results[key] = run_closed_loop(weather_file, weather_label, tuning_name, tuning)

    print('\n=== Comparison summary (all weathers) ===')
    for (weather_label, tuning_name), res in all_results.items():
        print(f'{weather_label:6s} | {tuning_name:25s}: '
              f'IAE_DO={res["iae_so"]:.4f}  ISE_DO={res["ise_so"]:.4f}  IAE_NO3={res["iae_sno"]:.4f}')
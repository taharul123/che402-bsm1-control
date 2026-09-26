"""
step3_closed_loop.py
Step 3 of the project: close the loop and simulate.

Two independent PI controllers:
  1. DO controller: measures SO in tank 5, manipulates kLa in tank 5, setpoint 2 g/m3.
  2. Nitrate controller: measures SNO in tank 2, manipulates internal recycle flow, setpoint 1 g/m3.

Controllers are tuned with the SIMC method using the FOPDT parameters found in Step 2.
Run under dry weather first, as the project asks.
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

# ---- FOPDT parameters from Step 2 (update these if you rerun step2 and get different numbers) ----
KLA_K, KLA_TAU, KLA_THETA = 0.01150, 0.0198, 0.1906
QINTR_K, QINTR_TAU, QINTR_THETA = 0.00019, 0.0625, 0.1000

# ---- actuator limits ----
KLA_MIN, KLA_MAX = 0.0, 240.0
QINTR_MIN, QINTR_MAX = 0.0, 200000.0


def run_closed_loop(weather_file='dry_data.txt', label='dry'):
    print(f'\n=== Closed-loop simulation under {label} weather ===')
    data_in = load_influent(weather_file)
    bsm1 = BSM1OL(data_in=data_in, timestep=TIMESTEP)
    bsm1.stabilize(atol=1e-5)

    kla_bias = bsm1.klas[4]
    qintr_bias = bsm1.qintr

    kp_kla, ti_kla = simc_tuning(KLA_K, KLA_TAU, KLA_THETA)
    kp_qintr, ti_qintr = simc_tuning(QINTR_K, QINTR_TAU, QINTR_THETA)
    print(f'  DO controller:      Kp={kp_kla:.4f}  Ti={ti_kla:.4f} d')
    print(f'  Nitrate controller: Kp={kp_qintr:.4f}  Ti={ti_qintr:.4f} d')

    do_ctrl = PIController(kp_kla, ti_kla, TIMESTEP, KLA_MIN, KLA_MAX, bias=kla_bias)
    no_ctrl = PIController(kp_qintr, ti_qintr, TIMESTEP, QINTR_MIN, QINTR_MAX, bias=qintr_bias)

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

        bsm1.klas[4] = u_kla
        bsm1.qintr = u_qintr
        bsm1.step(i)

        so_current = bsm1.y_out5_all[i, SO_IDX]
        sno_current = bsm1.y_out2_all[i, SNO_IDX]

        so_response[i] = so_current
        sno_response[i] = sno_current
        kla_signal[i] = u_kla
        qintr_signal[i] = u_qintr
        so_error[i] = err_so
        sno_error[i] = err_sno

    iae_so = np.sum(np.abs(so_error)) * TIMESTEP
    iae_sno = np.sum(np.abs(sno_error)) * TIMESTEP
    ise_so = np.sum(so_error ** 2) * TIMESTEP
    print(f'  IAE (DO tracking error)      = {iae_so:.4f}')
    print(f'  ISE (DO tracking error)      = {ise_so:.4f}')
    print(f'  IAE (nitrate tracking error) = {iae_sno:.4f}')

    # --- plots ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(sim_t, so_response, label='SO tank 5 (actual)')
    axes[0].axhline(SP_SO, color='r', linestyle='--', label='setpoint')
    axes[0].set_ylabel('SO [g/m3]')
    axes[0].set_title(f'DO control — {label} weather')
    axes[0].legend()

    axes[1].plot(sim_t, kla_signal)
    axes[1].set_ylabel('kLa tank 5 [1/d]')

    axes[2].plot(sim_t, sno_response, label='SNO tank 2 (actual)')
    axes[2].axhline(SP_SNO, color='r', linestyle='--', label='setpoint')
    axes[2].set_ylabel('SNO [g/m3]')
    axes[2].set_title(f'Nitrate control — {label} weather')
    axes[2].legend()

    axes[3].plot(sim_t, qintr_signal)
    axes[3].set_ylabel('Qintr [m3/d]')
    axes[3].set_xlabel('Time [days]')

    plt.tight_layout()
    outname = f'step3_closedloop_{label}.png'
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f'  saved {outname}')

    return {'iae_so': iae_so, 'ise_so': ise_so, 'iae_sno': iae_sno}


if __name__ == '__main__':
    results = run_closed_loop('dry_data.txt', 'dry')
    print('\n=== Summary (dry weather) ===')
    print(results)
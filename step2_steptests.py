"""
step2_steptests.py
Step 2 of the project: characterize the plant with step tests.

Test A: step the 5th compartment's kLa, record how SO in tank 5 responds.
Test B: step the internal recycle flow (qintr), record how SNO in tank 2 responds.

For each, we estimate an approximate FOPDT (first-order-plus-dead-time) model:
    gain K, time constant tau, dead time theta
using the classic 28.3% / 63.2% two-point method on the step response.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bsm2_python import BSM1OL
from common import load_influent, COMP

TIMESTEP = 1 / 1440       # 1 minute, in days
STEP_TIME = 7.0           # day at which we apply the step (roughly mid-file)

SO_IDX = COMP['SO']
SNO_IDX = COMP['SNO']

def moving_average(x, window):
    """Simple moving average to smooth out the daily diurnal oscillation."""
    if window < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')

def daily_average(time, response):
    """
    Collapse a fine time series into one average value per calendar day.
    Returns (day_times, day_values) -- day_times are day-centers (0.5, 1.5, ...).
    """
    day_index = np.floor(time).astype(int)
    n_days = day_index.max() + 1
    day_values = np.array([response[day_index == d].mean() for d in range(n_days)])
    day_times = np.arange(n_days) + 0.5
    return day_times, day_values


def fit_fopdt(time, response, step_time, u_before, u_after,
              baseline_window=(-3.0, -1.0), steady_window=(4.0, 6.0)):
    """
    Very simple FOPDT fit using the 28.3%/63.2% two-point method.
    baseline_window: (start, end) days RELATIVE TO the step, both negative,
        averaged to get a clean "before" value.
    steady_window: (start, end) days AFTER the step, averaged for the "settled" value.
    """
    mask = time >= step_time
    t_after = time[mask] - step_time
    y_after = response[mask]

    base_mask = (time >= step_time + baseline_window[0]) & (time <= step_time + baseline_window[1])
    y0 = response[base_mask].mean() if base_mask.sum() > 0 else y_after[0]

    steady_mask = (t_after >= steady_window[0]) & (t_after <= steady_window[1])
    if steady_mask.sum() == 0:
        yss = y_after[-1]
    else:
        yss = y_after[steady_mask].mean()

    delta_y = yss - y0
    delta_u = u_after - u_before

    gain_k = delta_y / delta_u if delta_u != 0 else np.nan

    # target values for 28.3% and 63.2% of the total change
    y_283 = y0 + 0.283 * delta_y
    y_632 = y0 + 0.632 * delta_y

    def time_to_reach(target):
        # first index where response crosses the target value
        if delta_y >= 0:
            idx = np.argmax(y_after >= target)
        else:
            idx = np.argmax(y_after <= target)
        return t_after[idx]

    t1 = time_to_reach(y_283)
    t2 = time_to_reach(y_632)

    tau = 1.5 * (t2 - t1)
    theta = t2 - tau
    theta = max(theta, 0.0)  # dead time can't be negative

    return gain_k, tau, theta


def run_kla_step_test():
    print('\n=== Test A: step test on kLa (tank 5) -> SO response ===')
    data_in = load_influent('dry_data.txt')
    bsm1 = BSM1OL(data_in=data_in, timestep=TIMESTEP)
    bsm1.stabilize(atol=1e-5)

    kla_before = bsm1.klas[4]
    kla_after = kla_before + 100.0  # step size -- feel free to adjust

    n_steps = len(bsm1.timesteps)  # one less than len(bsm1.simtime) -- avoids IndexError
    sim_t = bsm1.simtime[:n_steps]

    so_response = np.zeros(n_steps)
    kla_signal = np.zeros(n_steps)

    for i, t in enumerate(sim_t):
        current_kla = kla_after if t >= STEP_TIME else kla_before
        klas_arr = bsm1.klas.copy()
        klas_arr[4] = current_kla
        kla_signal[i] = current_kla
        bsm1.step(i, klas_arr)
        so_response[i] = bsm1.y_out5_all[i, SO_IDX] if i < len(bsm1.y_out5_all) else bsm1.y_out5[SO_IDX]

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(sim_t, kla_signal)
    axes[0].set_ylabel('kLa tank 5 [1/d]')
    axes[0].set_title('Step test: kLa (tank 5) -> SO (tank 5)')
    axes[1].plot(sim_t, so_response, alpha=0.4, label='raw')
    window = int(round(1.0 / TIMESTEP))  # 1-day moving average
    so_smooth = moving_average(so_response, window)
    axes[1].plot(sim_t, so_smooth, label='1-day smoothed')
    axes[1].legend()
    axes[1].set_ylabel('SO tank 5 [g/m3]')
    axes[1].set_xlabel('Time [days]')
    plt.tight_layout()
    plt.savefig('step2_kla_step_test.png', dpi=150)
    plt.close()
    print('  saved step2_kla_step_test.png')

    k, tau, theta = fit_fopdt(sim_t, so_response, STEP_TIME, kla_before, kla_after,
                            baseline_window=(-0.10, -0.01), steady_window=(0.15, 0.30))
    print(f'  Estimated FOPDT for kLa -> SO:  K={k:.5f}  tau={tau:.4f} d  theta={theta:.4f} d')
    return k, tau, theta


def run_recycle_step_test():
    print('\n=== Test B: step test on internal recycle flow -> SNO (tank 2) response ===')
    data_in = load_influent('dry_data.txt')
    bsm1 = BSM1OL(data_in=data_in, timestep=TIMESTEP)
    bsm1.stabilize(atol=1e-5)

    qintr_before = bsm1.qintr
    qintr_after = qintr_before * 1.3  # step size -- 30% increase

    n_steps = len(bsm1.timesteps)  # one less than len(bsm1.simtime) -- avoids IndexError
    sim_t = bsm1.simtime[:n_steps]

    sno_response = np.zeros(n_steps)
    qintr_signal = np.zeros(n_steps)

    for i, t in enumerate(sim_t):
        if t >= STEP_TIME:
            bsm1.qintr = qintr_after
        qintr_signal[i] = bsm1.qintr
        bsm1.step(i)
        sno_response[i] = bsm1.y_out2_all[i, SNO_IDX] if i < len(bsm1.y_out2_all) else bsm1.y_out2[SNO_IDX]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(sim_t, qintr_signal)
    axes[0].set_ylabel('Internal recycle flow [m3/d]')
    axes[0].set_title('Step test: internal recycle -> SNO (tank 2)')
    axes[1].plot(sim_t, sno_response, alpha=0.4, label='raw')
    window = int(round(1.0 / TIMESTEP))  # 1-day moving average
    sno_smooth = moving_average(sno_response, window)
    axes[1].plot(sim_t, sno_smooth, label='1-day smoothed')
    axes[1].legend()
    axes[1].set_ylabel('SNO tank 2 [g/m3]')
    axes[1].set_xlabel('Time [days]')
    plt.tight_layout()
    plt.savefig('step2_recycle_step_test.png', dpi=150)
    plt.close()
    print('  saved step2_recycle_step_test.png')

    k, tau, theta = fit_fopdt(sim_t, sno_response, STEP_TIME, qintr_before, qintr_after,
                            baseline_window=(-0.10, -0.01), steady_window=(0.15, 0.30))
    print(f'  Estimated FOPDT for Qintr -> SNO:  K={k:.5f}  tau={tau:.4f} d  theta={theta:.4f} d')
    return k, tau, theta


if __name__ == '__main__':
    kla_params = run_kla_step_test()
    recycle_params = run_recycle_step_test()

    print('\n=== Summary ===')
    print(f'kLa -> SO      : K={kla_params[0]:.5f}, tau={kla_params[1]:.4f} d, theta={kla_params[2]:.4f} d')
    print(f'Qintr -> SNO   : K={recycle_params[0]:.5f}, tau={recycle_params[1]:.4f} d, theta={recycle_params[2]:.4f} d')
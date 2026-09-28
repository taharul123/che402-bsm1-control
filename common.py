"""
common.py  (new version, follows the IWA BSM1 report initialization procedure)

Procedure used everywhere in this project:
  1. Stabilize the plant for 100 days on a CONSTANT influent (BSM1 report, Table 5)
  2. Run the 14-day dry-weather file as a warm-up
  3. Run the weather file to be tested (14 days)
  4. Evaluate only days 7 to 14 of that last file
"""
import os
import numpy as np
from bsm2_python import BSM1OL

# ------------------------------------------------------------------ settings
TIMESTEP = 1 / 1440            # 1 minute [d] (needed for correct violation-time calculation)
STAB_DAYS = 100                # length of the initial stabilization [d]
STAB_DT = 1 / 96               # 15 min, only used during the 100-day stabilization (for speed)

# Which constant influent to use for the 100-day stabilization:
#   'bsm1_table5'  -> BSM1 report, Table 5 (Q = 18 446 m3/d)   <- default
#   'package_file' -> constinfluent_bsm2.csv shipped inside bsm2-python (Q = 20 648 m3/d)
CONSTANT_SOURCE = 'bsm1_table5'

COMP = {
    'SI': 0, 'SS': 1, 'XI': 2, 'XS': 3, 'XBH': 4, 'XBA': 5, 'XP': 6,
    'SO': 7, 'SNO': 8, 'SNH': 9, 'SND': 10, 'XND': 11, 'SALK': 12,
    'TSS': 13, 'Q': 14, 'TEMP': 15, 'SD1': 16, 'SD2': 17, 'SD3': 18,
    'XD4': 19, 'XD5': 20,
}


def _tss(xs, xi, xbh, xba, xp):
    """Influent TSS from the particulate COD fractions (BSM1 report, eq. 89)."""
    return 0.75 * (xs + xi + xbh + xba + xp)


def load_influent(filepath):
    """Read a course influent file (t Si Ss Xi Xs Xbh Xba Xp So Sno Snh Snd Xnd Salk Q)
    and return the (n, 22) array expected by bsm2-python."""
    with open(filepath) as f:
        first = f.readline().split()
    try:
        float(first[0]); skip = 0
    except ValueError:
        skip = 1                                   # file has a text header row
    raw = np.loadtxt(filepath, skiprows=skip)
    t, c, q = raw[:, 0], raw[:, 1:14], raw[:, 14]
    tss = _tss(c[:, 3], c[:, 2], c[:, 4], c[:, 5], c[:, 6])
    n = len(t)
    return np.column_stack([t, c, tss, q, np.full(n, 15.0), np.zeros((n, 5))])


def constant_row():
    """21-component constant influent used for the 100-day stabilization."""
    if CONSTANT_SOURCE == 'package_file':
        import bsm2_python
        p = os.path.join(os.path.dirname(bsm2_python.__file__), 'data', 'constinfluent_bsm2.csv')
        return np.loadtxt(p, delimiter=',', ndmin=2)[0, 1:]
    # BSM1 report, Table 5 (flow-weighted averages)
    si, ss, xi, xs, xbh, xba, xp = 30.0, 69.5, 51.2, 202.32, 28.17, 0.0, 0.0
    so, sno, snh, snd, xnd, salk, q = 0.0, 0.0, 31.56, 6.95, 10.59, 7.0, 18446.0
    return np.array([si, ss, xi, xs, xbh, xba, xp, so, sno, snh, snd, xnd, salk,
                     _tss(xs, xi, xbh, xba, xp), q, 15.0, 0, 0, 0, 0, 0])


def build_constant_data(days):
    """Constant influent lasting `days` days."""
    r = constant_row()
    return np.vstack([np.r_[0.0, r], np.r_[float(days), r]])


def build_weather_data(test_file, warmup_file='dry_data.txt'):
    """14 days of dry weather (warm-up) followed by the 14-day test file.
    The last 7 days of the test file (t = 21 ... 28 d) are the evaluation window."""
    warm = load_influent(warmup_file)[:-1].copy()   # drop last row (t=14) -> no duplicate time
    test = load_influent(test_file).copy()
    test[:, 0] += 14.0
    return np.vstack([warm, test])


def make_plant(data_in, eval_window):
    """Create an open-loop BSM1 plant (1-minute time step) with the given evaluation window [d]."""
    end = data_in[-1, 0]
    return BSM1OL(data_in=data_in, timestep=TIMESTEP,
                  evaltime=[float(eval_window[0]), float(end - TIMESTEP)])


def stabilize_constant(plant, days=STAB_DAYS, klas=None, record_every=96):
    """Run `days` days on the constant influent (no disturbance), starting from the plant's
    initial state. Returns recorded (time, SO5, SNO2, SNH5) so the run can be plotted/checked."""
    plant.y_in[0, :] = constant_row()               # first influent row = constant input
    old_dt = plant.timesteps[0]
    plant.timesteps[0] = STAB_DT
    n = int(round(days / STAB_DT))
    rec = []
    for k in range(n):
        plant.step(0, klas) if klas is not None else plant.step(0)
        if k % record_every == 0:
            rec.append((k * STAB_DT, plant.y_out5[COMP['SO']], plant.y_out2[COMP['SNO']], plant.y_out5[COMP['SNH']]))
    plant.timesteps[0] = old_dt
    return np.array(rec)


def run_open_loop(plant):
    """Run all time steps (open loop, default aeration KLa = [0, 0, 240, 240, 84], Qint = 55 338)."""
    for i in range(len(plant.timesteps)):
        plant.step(i)
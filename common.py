"""
common.py
Shared helper to load the dry/rain/storm influent .txt files and convert them
into the 22-column array (time + 21 components) that BSM1OL expects.
"""
import numpy as np


def load_influent(filepath, temp=15.0):
    """
    Load a BSM1 influent file with columns:
        t  Si  Ss  Xi  Xs  Xbh  Xba  Xp  So  Sno  Snh  Snd  Xnd  Salk  Q
    and return a (n, 22) array with columns:
        t  SI SS XI XS XBH XBA XP SO SNO SNH SND XND SALK TSS Q TEMP SD1 SD2 SD3 XD4 XD5

    Parameters
    ----------
    filepath : str
        Path to the influent .txt file.
    temp : float
        Constant temperature [deg C] to fill in (BSM1 default is 15).

    Returns
    -------
    data_in : np.ndarray, shape (n, 22)
    """
    # Some influent files have a text header row (e.g. "t Si Ss ..."); skip it if present.
    with open(filepath) as f:
        first_line = f.readline().strip()
    try:
        float(first_line.split()[0])
        skip = 0  # first line is already numeric data
    except ValueError:
        skip = 1  # first line is a header, skip it

    raw = np.loadtxt(filepath, skiprows=skip)  # shape (n, 15): t, Si..Salk, Q

    t = raw[:, 0]
    si, ss, xi, xs, xbh, xba, xp, so, sno, snh, snd, xnd, salk, q = (
        raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6],
        raw[:, 7], raw[:, 8], raw[:, 9], raw[:, 10], raw[:, 11], raw[:, 12],
        raw[:, 13], raw[:, 14],
    )

    n = raw.shape[0]
    tss = np.zeros(n)          # not provided in file -> assume 0 (standard BSM1 practice)
    temp_col = np.full(n, temp)
    sd1 = sd2 = sd3 = xd4 = xd5 = np.zeros(n)

    data_in = np.column_stack([
        t, si, ss, xi, xs, xbh, xba, xp, so, sno, snh, snd, xnd, salk,
        tss, q, temp_col, sd1, sd2, sd3, xd4, xd5
    ])
    return data_in


# Index of each ASM1 component within a 21-column row (y_out1, y_out2, ..., ys_eff, etc.)
COMP = {
    'SI': 0, 'SS': 1, 'XI': 2, 'XS': 3, 'XBH': 4, 'XBA': 5, 'XP': 6,
    'SO': 7, 'SNO': 8, 'SNH': 9, 'SND': 10, 'XND': 11, 'SALK': 12,
    'TSS': 13, 'Q': 14, 'TEMP': 15, 'SD1': 16, 'SD2': 17, 'SD3': 18,
    'XD4': 19, 'XD5': 20,
}

def build_constant_influent(reference_filepath, n_days=5, timestep_days=1 / 1440, temp=15.0):
    """
    Build a constant (unchanging) influent by averaging the dry-weather file,
    repeated for n_days. Used for clean step-response tests (Step 2).
    """
    data_in = load_influent(reference_filepath, temp=temp)
    avg_row = data_in[:, 1:].mean(axis=0)  # average of all 21 components over the whole file

    n_steps = int(n_days / timestep_days) + 1
    t = np.arange(n_steps) * timestep_days
    const_block = np.tile(avg_row, (n_steps, 1))
    return np.column_stack([t, const_block])
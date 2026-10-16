"""Remove numeric lab content while preserving all observation-pattern information."""
import numpy as np
from causal_control_features import WINDOWS, STATS
from transfer_sequence import collate

# Column order is train_rich_transfer.COLS. Keep HR/O2Sat/Resp/Temp, BP, EtCO2,
# and FiO2 numeric values. FiO2 is an oxygen-support setting, not a lab assay.
LAB_INDICES = tuple([4, 5, 6, 7, 8, 9, 14, 15] + list(range(17, 34)))
VALUE_STATS = tuple(range(7))  # last/mean/std/min/max/slope/change
BLIND_COLUMNS = np.asarray([w*34*len(STATS)+f*len(STATS)+k
                            for w in range(len(WINDOWS))
                            for f in LAB_INDICES for k in VALUE_STATS], dtype=np.int64)


def blind_table(x):
    result = np.array(x, copy=True)
    result[:, BLIND_COLUMNS] = 0
    return result


def blind_collate(items, median, std, device='cuda', active=None):
    batch = collate(items, median, std, device=device, active=active)
    # Keep observed, ever-observed and age channels byte-identical.
    batch[0][:, :, list(LAB_INDICES), 0] = 0
    return batch

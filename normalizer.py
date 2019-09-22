import numpy as np
import functools
import operator

def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(fp[0], interped), fp[1])

def norm(data_zip, scales):
    normalized = {i: [] for i in scales.keys()}
    for idx, i in enumerate(data_zip):
        for x in i[1]:  # iterate through array
            normalized[i[0]].append(interp_fast(x, scales[i[0]]))  # append to respective normalized input list
    
    return normalized
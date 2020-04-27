# coding: UTF8
import numpy as np


def remove_array_array(dad, son):
    index = 0
    for ida in dad:
        if(ida == son).all():
            break
        index += 1
    if index < len(dad):
        return np.delete(dad, index, axis=0)
    return dad

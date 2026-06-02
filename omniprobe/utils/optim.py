
import numpy as np


def cosine_decay_linear_warmup(current_step, max_step, warmup_step, min_factor=0.01):
    assert max_step > warmup_step

    range_factor = 1 - min_factor

    if warmup_step <= 0:
        rel_step = current_step / max_step
        return range_factor * np.cos(0.5 * rel_step * np.pi) + min_factor

    if current_step <= warmup_step:
        return range_factor * (current_step / warmup_step) + min_factor
    else:
        rel_step = (current_step - warmup_step) / (max_step - warmup_step)
        return range_factor * np.cos(0.5 * rel_step * np.pi) + min_factor

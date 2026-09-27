"""Central CPU budget, so the pipeline can be told to run cool.

On a thin gaming laptop every stage here will happily saturate all 12 threads
for hours and push the package past 90 C.  Two dials keep that in check:

``BER_THREADS``   worker/thread count used by every stage (default 6 of 12).
                  Power draw rises faster than throughput past about half the
                  cores, so halving threads sheds far more heat than speed.
``BER_THROTTLE``  duty-cycle pause.  After each unit of work a stage sleeps for
                  this fraction of the time that unit took, which caps the
                  *sustained* power the package has to dissipate.  0.25 means
                  "work 4 seconds, idle 1".

Set ``BER_THREADS=12 BER_THROTTLE=0`` to go back to running flat out.
"""
import os
import time

THREADS = max(1, int(os.environ.get("BER_THREADS", "6")))
THROTTLE = float(os.environ.get("BER_THROTTLE", "0.25"))

# BLAS/OpenMP pools are created on first import of numpy/torch, so these have
# to be set before those imports -- call apply() at the top of every entry
# point, before numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "RAYON_NUM_THREADS"):        # HuggingFace tokenizers use rayon
    os.environ.setdefault(_v, str(THREADS))


def apply():
    """Clamp torch too, if it is loaded."""
    try:
        import torch
        torch.set_num_threads(THREADS)
    except Exception:
        pass
    return THREADS


def breathe(t0, factor=None):
    """Sleep for a fraction of the work just done, to cap sustained power."""
    f = THROTTLE if factor is None else factor
    if f <= 0:
        return
    dt = time.time() - t0
    if dt > 0:
        time.sleep(dt * f)


class Pace:
    """Context manager form of :func:`breathe`."""

    def __init__(self, factor=None):
        self.factor = factor

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        breathe(self.t0, self.factor)
        return False

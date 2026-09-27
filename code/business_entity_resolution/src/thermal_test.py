"""Measure package temperature under the pipeline's actual workload.

    python thermal_test.py --threads 6 --throttle 0.25 --seconds 90

Runs the same mix of work the pipeline does (parallel term hashing, rapidfuzz
batches, LightGBM-style numpy) at a given CPU budget and reports the
temperature curve, so the budget can be chosen against a real target rather
than guessed.
"""
import argparse
import os
import subprocess
import sys
import time


def read_temp():
    """Package temperature in C, or None. Tries WMI then OpenHardwareMonitor."""
    ps = ("$t=Get-CimInstance -Namespace root/wmi -ClassName "
          "MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue; "
          "if($t){($t | Measure-Object -Property CurrentTemperature "
          "-Maximum).Maximum}")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=20)
        v = r.stdout.strip()
        if v:
            return float(v) / 10.0 - 273.15
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--throttle", type=float, default=0.25)
    ap.add_argument("--seconds", type=int, default=90)
    a = ap.parse_args()
    os.environ["BER_THREADS"] = str(a.threads)
    os.environ["BER_THROTTLE"] = str(a.throttle)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import hw
    hw.THREADS = a.threads
    hw.THROTTLE = a.throttle
    import numpy as np
    from rapidfuzz import fuzz, process

    base = read_temp()
    print(f"threads={a.threads} throttle={a.throttle} "
          f"start temp={'n/a' if base is None else f'{base:.0f} C'}",
          flush=True)
    rng = np.random.default_rng(0)
    words = ["alpha", "beta", "gamma", "delta", "road", "street", "private",
             "limited", "nagar", "colony"]
    A = [" ".join(rng.choice(words, 6)) for _ in range(60000)]
    B = [" ".join(rng.choice(words, 6)) for _ in range(60000)]
    t_end = time.time() + a.seconds
    temps = []
    last = 0
    while time.time() < t_end:
        t0 = time.time()
        process.cpdist(A, B, scorer=fuzz.token_sort_ratio,
                       workers=a.threads, dtype=np.float32)
        M = rng.random((1400, 1400), np.float32)
        M @ M
        hw.breathe(t0, a.throttle)
        if time.time() - last > 10:
            last = time.time()
            c = read_temp()
            if c:
                temps.append(c)
                print(f"  t={int(time.time()-(t_end-a.seconds)):>3}s  "
                      f"{c:.0f} C", flush=True)
    if temps:
        print(f"\nmax {max(temps):.0f} C   mean {sum(temps)/len(temps):.0f} C")
    else:
        print("\nno temperature sensor readable from WMI; "
              "watch Armoury Crate instead")


if __name__ == "__main__":
    main()

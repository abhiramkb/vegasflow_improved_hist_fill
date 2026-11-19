"""
    Example: basic integration

    Very basic example with a simple integrand

    USS_HISTFILL: Compare stock VegasFlow and VegasFlow with improved histogram fill for the simple integrand
"""

from vegasflow import VegasFlow, float_me, run_eager
import sys
sys.path.insert(0, "../../src/vegasflow")
from vflow_histfill import VegasFlowUSSHistFill
import time
import numpy as np
import tensorflow as tf


# MC integration setup
dim = 4
ncalls = int(1e6)
n_iter = 5


def symgauss(xarr):
    """symgauss test function"""
    n_dim = xarr.shape[-1]
    a = float_me(0.1)
    n100 = float_me(100 * n_dim)
    pref = tf.pow(1.0 / a / np.sqrt(np.pi), n_dim)
    coef = tf.reduce_sum(tf.range(n100 + 1))
    coef += tf.reduce_sum(tf.square((xarr - 1.0 / 2.0) / a), axis=1)
    coef -= (n100 + 1) * n100 / 2.0
    return pref * tf.exp(-coef)


if __name__ == "__main__":
    """Comparing stock and modified VegasFlow versions"""
    print(f"\nStock VegasFlow:")
    print(f"VEGAS MC, ncalls={ncalls}:")
    vegas_instance = VegasFlow(dim, ncalls, events_limit=int(7e4))
    vegas_instance.compile(symgauss)
    print("Warmup...")
    warmup = vegas_instance.run_integration(1)
    print("Warmup done. Now running the integration...")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
    print("Change the number of events and freeze the grid...")
    vegas_instance.freeze_grid()
    vegas_instance.run_integration(n_iter)
    del(vegas_instance)

    print(f"\n\nModified VegasFlow")
    start = time.time()
    vegas_instance = VegasFlowUSSHistFill(dim, ncalls, events_limit=int(7e4))
    vegas_instance.compile(symgauss)
    print("Warmup...")
    warmup = vegas_instance.run_integration(1)
    print("Warmup done. Now running the integration...")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
    print("Change the number of events and freeze the grid...")
    vegas_instance.freeze_grid()
    vegas_instance.run_integration(n_iter)
    del(vegas_instance)

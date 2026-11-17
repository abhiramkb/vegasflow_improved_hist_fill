from vegasflow import VegasFlow, float_me, run_eager
import time
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import sys
sys.path.insert(0, "../../src/vegasflow")
from vflow_histfill import VegasFlowUSSHistFill

PI = tf.constant(np.pi, dtype=tf.float64)

def watson_triple_integrand(x):
    """
    3D Watson's integrand.
    
    dim = 3, bounds: [0, pi]^3
    integral = 1/pi^3 * int 1/(1 - cos(x1) cos(x2) cos(x3)) dx
    Expected result: 1.393203929
    Integrable singularity at the corners where prod cos = 1.
    Reference: A. J. Guttmann, "Lattice Green functions in all dimensions",
               J. Phys. A 43 (2010) 305205, https://arxiv.org/abs/1004.1435
    """
    u,v,w = tf.unstack(x, axis=-1)
    return (1.0/PI**3)*1.0/(1.0 - tf.math.cos(u)*tf.math.cos(v)*tf.math.cos(w))
watson_triple_integrand.dimension = 3
watson_triple_integrand.expected_result = 1.393203929


# --------------------------------------------------------------------------------------
# Additional expensive integrands (can be swapped oout with the one in the main block).
# The bounds have to be specified in the constructor if they are not [0,1]^d.
# --------------------------------------------------------------------------------------

def watson_quadruple_integrand(x):
    """
    4D generalisation of Watson's integrand.

    dim = 4, bounds: [0, pi]^4
    integral = 1/pi^4 * int 1/(1 - cos(x1) cos(x2) cos(x3) cos(x4)) dx
    Expected result: 1.118636387
    """
    c = tf.reduce_prod(tf.math.cos(x), axis=-1)
    return (1.0/PI**4)/(1.0 - c)
watson_quadruple_integrand.dimension = 4
watson_quadruple_integrand.expected_result = 1.118636387
def sqrt_singular_integrand(x):
    """
    Separable integrable singularity along every dimension at x_i = 0.

    dim = d, bounds: [0, 1]^d
    integral = int prod_i 1/(2 sqrt(x_i)) dx
    Expected result: 1.0
    """
    return tf.reduce_prod(0.5*tf.math.rsqrt(x), axis=-1)
sqrt_singular_integrand.dimension = 3
sqrt_singular_integrand.expected_result = 1.0



def genz_corner_peak_integrand(x):
    """
    Genz corner-peak. Peak at the origin.

    dim = 3, bounds: [0, 1]^3
    integral = int (1 + c(x1 + x2 + x3))^-4 dx,  c = 5
    Expected result: 1/1056 = 0.000946969697
    """
    GENZ_C = 5.0
    return (1.0 + GENZ_C*tf.reduce_sum(x, axis=-1))**-4
genz_corner_peak_integrand.dimension = 3
genz_corner_peak_integrand.expected_result = 0.000946969697

if __name__ == "__main__":
    ncalls = int(4e6)
    n_iter = 5

    integrand = genz_corner_peak_integrand
    dim = integrand.dimension
    expected_result = integrand.expected_result

    print(f"***Running comparison between Stock VegasFlow and modified VegasFlow\n***with tf.unsorted_segment_sum histogram fill for integrand: {integrand.__name__}")
    print(f"***Expected result: {expected_result}")

    print(f"\nIntegrator: Stock VegasFlow")
    vegas_instance = VegasFlow(dim, ncalls, events_limit=int(7e4))
    vegas_instance.compile(integrand)
    print("Warmup...")
    warmup = vegas_instance.run_integration(1)
    print("Warmup done. Now running the integration...")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
    del(vegas_instance)

    print(f"\nModified VegasFlow")
    vegas_instance = VegasFlowUSSHistFill(dim, ncalls, events_limit=int(7e4))
    print("Warmup...")
    vegas_instance.compile(integrand)    
    warmup = vegas_instance.run_integration(1)
    print("Warmup done. Now running the integration...")
    start = time.time()
    result = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"Vegas took: time (s): {end-start}")
    del(vegas_instance)

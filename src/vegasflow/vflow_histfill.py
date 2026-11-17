import json

import numpy as np
import tensorflow as tf
from vegasflow import VegasFlow
from vegasflow.configflow import ALPHA, DTYPE, TECH_CUT
from vegasflow.vflow import refine_grid_per_dimension

# --- VEGAS grid training ---
class VegasFlowUSSHistFill(VegasFlow):
    def _importance_sampling_array_filling(self, results2, indices):
        if not self.train:
            return []

        n_bins = self.grid_bins - 1
        # indices: (N, n_dim) bin of each event in each dimension -> segment id dim*n_bins + bin
        segment_ids = indices + n_bins * tf.range(self.n_dim, dtype=indices.dtype)
        data = tf.broadcast_to(tf.expand_dims(results2, -1), tf.shape(indices))
        arr_res2 = tf.math.unsorted_segment_sum(data, segment_ids, self.n_dim * n_bins)

        return tf.reshape(arr_res2, (self.n_dim, n_bins))
from typing import Dict

import jax

from dgm.smoother.feature_extractors import get_features_to_features_extractor
from dgm.smoother.feature_extractors import get_time_and_states_to_features_extractor
from dgm.utils.helper_functions import squared_l2_norm
from dgm.utils.representatives import FeaturesToFeaturesType, TimeAndStatesToFeaturesType


def get_composite_gp_mean(core_type: TimeAndStatesToFeaturesType, core_kwargs: Dict,
                          mean_head_type: FeaturesToFeaturesType, mean_head_kwargs: Dict):
    core_init, core_apply, _ = get_time_and_states_to_features_extractor(core_type, core_kwargs)
    head_init, head_apply, _ = get_features_to_features_extractor(mean_head_type, mean_head_kwargs)

    def mean_init(rng, n_output_dimensions):
        # Why is in_shape n_output_dimensions + 1? Because we join time with state
        in_shape = (-1, n_output_dimensions + 1)
        core_rng, head_rng = jax.random.split(rng, 2)
        out_shape, _ = core_init(core_rng, in_shape)
        return head_init(head_rng, out_shape)

    @jax.jit
    def mean_apply(times, initial_conditions, parameters, core_parameters):
        features = core_apply(times, initial_conditions, core_parameters)
        return head_apply(features, parameters)

    @jax.jit
    def get_regularization(parameters, weights):
        return weights[mean_head_kwargs['weight_key']] * squared_l2_norm(parameters)

    return mean_init, mean_apply, get_regularization

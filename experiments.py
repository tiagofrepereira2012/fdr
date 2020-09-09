import numpy as np
from scipy.spatial.distance import pdist, cdist
import sampling
import scoring
import plots

def fair_biometric_experiment(FAR_THRESHOLDS=[0.1, 0.01, 0.001, 0.0001, 0.00001]):
    """
    Cannonical example of fair system
    """
    return generate_abstract_experiment(DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=500,FAR_THRESHOLDS=FAR_THRESHOLDS)

def unfair_biometric_experiment(FAR_THRESHOLDS=[0.1, 0.01, 0.001, 0.0001, 0.00001]):
    """
    Cannonical example of unfair system
    """
    
    """
    impostors_per_demographic_dev = [np.random.normal(loc=-6, scale=0.5, size=(100)),
                                    np.random.normal(loc=-5, scale=0.5, size=(100)),
                                    np.random.normal(loc=-4, scale=0.5, size=(100))]

    genuines_per_demographic_dev =  [np.random.normal(loc=-5, scale=0.5, size=(100)),
                                    np.random.normal(loc=-4, scale=0.5, size=(100)),
                                    np.random.normal(loc=-3, scale=0.5, size=(100))]

    impostors_per_demographic_eval = [np.random.normal(loc=-6, scale=0.5, size=(100)),
                                    np.random.normal(loc=-5, scale=0.5, size=(100)),
                                    np.random.normal(loc=-4, scale=0.5, size=(100))]

    genuines_per_demographic_eval =  [np.random.normal(loc=-5, scale=0.5, size=(100)),
                                    np.random.normal(loc=-4, scale=0.5, size=(100)),
                                    np.random.normal(loc=-3, scale=0.5, size=(100))]
    n_demographics = 3

    return impostors_per_demographic_dev, genuines_per_demographic_dev, impostors_per_demographic_eval,genuines_per_demographic_eval, n_demographics
    """

    
    #DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=100,
    return generate_abstract_experiment(DEMOGRAPHIC_DISPLACEMENTS=[0.8, 1.3, 1.5],
                                        DEMOGRAPHIC_STD=[0.1, 0.2, 0.3],
                                        DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=500,
                                        T_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
                                        Z_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
                                        FAR_THRESHOLDS=FAR_THRESHOLDS)


def generate_abstract_experiment(
    DEMOGRAPHIC_DISPLACEMENTS=[1, 1, 1],
    DEMOGRAPHIC_STD=[0.2, 0.2, 0.2],
    DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=100,
    T_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    Z_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    SEED=10,
    FAR_THRESHOLDS=[0.1, 0.01, 0.001, 0.0001, 0.00001],
):

    enroll_samples_demographics, probe_samples_demographics = sampling.sample_hypercube(
        demographics_displacements=DEMOGRAPHIC_DISPLACEMENTS,
        demographic_std=DEMOGRAPHIC_STD,
        n_identities=DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC
        + T_NORM_IDENTITIES_PER_DEMOGRAPHIC
        + Z_NORM_IDENTITIES_PER_DEMOGRAPHIC,
        seed=SEED,
    )

    (
        enroll_dev,
        probe_dev,
        enroll_eval,
        probe_eval,
        z_norm,
        t_norm,
    ) = sampling.split_sets(
        enroll_samples_demographics,
        probe_samples_demographics,
        DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC // 2,
        DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC // 2,
        T_NORM_IDENTITIES_PER_DEMOGRAPHIC,
        Z_NORM_IDENTITIES_PER_DEMOGRAPHIC,
    )

    (
        impostors_per_demographic_dev,
        genuines_per_demographic_dev,
    ) = scoring.compute_scores_per_demographic(enroll_dev, probe_dev, factor=-1)

    (
        impostors_per_demographic_eval,
        genuines_per_demographic_eval,
    ) = scoring.compute_scores_per_demographic(enroll_eval, probe_eval, factor=-1)

    n_demographics = len(DEMOGRAPHIC_DISPLACEMENTS)

    return (
        impostors_per_demographic_dev,
        genuines_per_demographic_dev,
        impostors_per_demographic_eval,
        genuines_per_demographic_eval,
        n_demographics
    )

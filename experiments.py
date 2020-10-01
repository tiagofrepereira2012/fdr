import numpy as np
from scipy.spatial.distance import pdist, cdist
import sampling
import scoring
import plots


def generate_abstract_samples(
    DEMOGRAPHIC_DISPLACEMENTS=[1, 1, 1],
    DEMOGRAPHIC_STD=[0.2, 0.2, 0.2],
    DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=100,
    T_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    Z_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    SEED=10,
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

    return enroll_dev, probe_dev, enroll_eval, probe_eval, z_norm, t_norm


def generate_raw_scores(enroll_dev, probe_dev, enroll_eval, probe_eval):

    (
        impostors_per_demographic_dev,
        genuines_per_demographic_dev,
    ) = scoring.compute_scores_per_demographic(enroll_dev, probe_dev, factor=-1)

    (
        impostors_per_demographic_eval,
        genuines_per_demographic_eval,
    ) = scoring.compute_scores_per_demographic(enroll_eval, probe_eval, factor=-1)

    return (
        impostors_per_demographic_dev,
        genuines_per_demographic_dev,
        impostors_per_demographic_eval,
        genuines_per_demographic_eval,
    )


def generate_uncalibrated_experiment(
    DEMOGRAPHIC_DISPLACEMENTS=[1, 1, 1],
    DEMOGRAPHIC_STD=[0.2, 0.2, 0.2],
    DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=100,
    T_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    Z_NORM_IDENTITIES_PER_DEMOGRAPHIC=25,
    SEED=10,
):

    (
        enroll_dev,
        probe_dev,
        enroll_eval,
        probe_eval,
        z_norm,
        t_norm,
    ) = generate_abstract_samples(
        DEMOGRAPHIC_DISPLACEMENTS=DEMOGRAPHIC_DISPLACEMENTS,
        DEMOGRAPHIC_STD=DEMOGRAPHIC_STD,
        DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC=DEV_EVAL_IDENTITIES_PER_DEMOGRAPHIC,
        T_NORM_IDENTITIES_PER_DEMOGRAPHIC=T_NORM_IDENTITIES_PER_DEMOGRAPHIC,
        Z_NORM_IDENTITIES_PER_DEMOGRAPHIC=Z_NORM_IDENTITIES_PER_DEMOGRAPHIC,
        SEED=SEED,
    )

    return generate_raw_scores(enroll_dev, probe_dev, enroll_eval, probe_eval)




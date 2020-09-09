import numpy as np
import bob.measure
from scipy.spatial.distance import pdist, cdist


def compute_scores_per_demographic(enroll_samples_demographics, probe_samples_demographics, factor=-1):
    

    def compute_scores_one_demographic(enroll_samples_demographics, probe_samples_demographics, factor=-1):
        impostors = []
        genuines = []
        for i in enroll_samples_demographics:
            for j in probe_samples_demographics:

                scores = compute_scores(enroll_samples_demographics[i].reshape(1,enroll_samples_demographics[i].shape[0]),\
                                       probe_samples_demographics[j],
                                       factor=factor)
                if i==j:
                    genuines.append(scores)
                else:
                    impostors.append(scores)

        return np.hstack(impostors).flatten(), np.hstack(genuines).flatten()
    
    genuines_per_demographic = []
    impostors_per_demographic = []
    for i,j in zip(enroll_samples_demographics, probe_samples_demographics): 
        impostors, genuines = compute_scores_one_demographic(enroll_samples_demographics[i], probe_samples_demographics[j])

        impostors_per_demographic.append(impostors)
        genuines_per_demographic.append(genuines)    
        
    return impostors_per_demographic, genuines_per_demographic


def compute_scores(e,p, factor=-1, metric="euclidean"):
    return factor*cdist(e, p, metric=metric)


def compute_fmr_fnmr_multiple_threshold(impostors_per_demographic, genuines_per_demographic, thresholds):

    FMR = {}
    FNMR = {}

    for i, (imp,gen) in enumerate(zip(impostors_per_demographic, genuines_per_demographic)):

        FMR[i] = []
        FNMR[i] = []

        for t in thresholds:
            fmr, fnmr = bob.measure.farfrr(imp,gen,t)

            FMR[i].append(fmr)
            FNMR[i].append(fnmr)

    return FMR, FNMR


def compute_thresholds(impostors_per_demographic, far_threshs):
    # PLOTTING THE THRESHOLD
    return [bob.measure.far_threshold(np.hstack(impostors_per_demographic), np.array([]), far_value=f) for f in far_threshs]

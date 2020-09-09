import numpy as np
import itertools as it

def sample_hypercube(demographics_displacements=[2], demographic_std=[0.2], n_identities=4, samples_per_identity=50, d=None, seed=10):
    """
    Generate samples by sampling the corners of an hypercube.
    Each corner of an hypercube of size `s` corresponds to an identity (e.g. In a 2d square, it's possible to fit four identities )
    
    
    Parameters
    ----------
    
      demographics_displacements: list
        Each element in this list corresponds to the average distance of samples within a demographic (cube side)
        
      demographic_std: list
        Each element corresponds to the standard deviation of samples from same identity in a demographic.
        
      n_identities: int
         Number of identities per demographic
        
      samples_per_identity: int
         Number of samples per identity
      
      d: int
         Dimensionality of the cube
         
      seed: int
         Seed of the random number generator
    """
    
    np.random.seed(seed)
    
    n = int(np.ceil(np.sqrt(n_identities))) if d is None else d
    
    enroll_samples_demographics = {}
    probe_samples_demographics = {}
    
    for i,(displacement,std) in enumerate(zip(demographics_displacements,demographic_std)):
        possible_identities = list(it.product([displacement,-displacement], repeat=n))
        
        enroll_samples_demographics[i] = {}
        probe_samples_demographics[i] = {}
        for j in range(n_identities):
            #h = np.sqrt(2*(i**2))-i
            loc = np.array(possible_identities[j])
            samples = np.random.normal(loc=loc, scale=std, size=(samples_per_identity, n))            
            enroll_samples_demographics[i][j] = (samples[0,:])
            probe_samples_demographics[i][j] = samples[1:,:]
    
    return enroll_samples_demographics, probe_samples_demographics


def split_sets(enroll_samples_demographics, probe_samples_demographics, dev_set_samples, eval_set_samples, t_norm_samples, z_norm_samples):
    
    enroll_dev = {}    
    probe_dev = {}
    
    enroll_eval = {}
    probe_eval = {}
    
    z_norm = {}
    t_norm = {}
    
    for k in enroll_samples_demographics:
        indexes = [w for w in enroll_samples_demographics[k]]
        enroll_dev[k] = {}
        probe_dev[k] = {}

        enroll_eval[k] = {}
        probe_eval[k] = {}
        
        z_norm[k] = {}
        t_norm[k] = {}
        
        # DEV SET
        offset = 0
        for i in indexes[offset:dev_set_samples]:
            enroll_dev[k][i] = enroll_samples_demographics[k][i]
            probe_dev[k][i] = probe_samples_demographics[k][i]

        # EVAL SET
        offset+= eval_set_samples
        for i in indexes[offset:dev_set_samples+eval_set_samples]:
            enroll_eval[k][i] = enroll_samples_demographics[k][i]
            probe_eval[k][i] = probe_samples_demographics[k][i]
        
        # Z-NORM
        offset+= z_norm_samples
        for i in indexes[offset:dev_set_samples+eval_set_samples+z_norm_samples]:
            z_norm[k][i] = probe_samples_demographics[k][i]
        
        offset+= t_norm_samples
        for i in indexes[offset:]:
            t_norm[k][i] = probe_samples_demographics[k][i]
        
        
    return enroll_dev, probe_dev, enroll_eval, probe_eval, z_norm, t_norm
import matplotlib.pyplot as plt
import scoring
import numpy as np


def compute_fdr(A_tau, B_tau, alpha=0.5):
    
    max_A_tau = max([abs(x-y) for x in A_tau for y in A_tau])
    max_B_tau = max([abs(x-y) for x in B_tau for y in B_tau])
    
    return 1-((alpha * max_A_tau+ (1-alpha)*max_B_tau))


def compute_fairness_biometric_verification(impostors_per_demographic, genuines_per_demographic, thresholds, n_tau, n_demographics, alpha=0.5):
    
    FMR_raw, FNMR_raw = scoring.compute_fmr_fnmr_multiple_threshold(impostors_per_demographic, genuines_per_demographic, thresholds)
    
    e_tau = []
    for i in range(n_tau):
        fmr_tau_per_demographic = []
        fnmr_tau_per_demographic = []
        for j in range(n_demographics):
            fmr_tau_per_demographic.append(FMR_raw[j][i])
            fnmr_tau_per_demographic.append(FNMR_raw[j][i])
        
        e_tau.append(compute_fdr(fmr_tau_per_demographic, fnmr_tau_per_demographic, alpha=alpha))

    return e_tau


def fpr_auc_score(f_tau, x):
    transform = lambda x : -1*np.log10(x)
    
    x_scaled = np.array(x)
    x_scaled = (x_scaled-min(x_scaled)) / (max(x_scaled)-min(x_scaled))
    
    return -1*np.trapz(f_tau, x_scaled)

    
def fairness_plot_clean(f_taus, labels, far_thresholds, epsilon=0.99, title=""):
    
    transform = lambda x : -1*np.log10(x)
    #transform = lambda x : x
    
    fig, ax = plt.subplots()
    [ax.plot(transform(far_thresholds),f_tau) for f_tau, label in zip(f_taus, labels)]
    [ax.scatter(transform(far_thresholds),f_tau, label=label) for f_tau, label in zip(f_taus, labels)]

    #ax.axhline(epsilon, color="red", label="$\epsilon$", linestyle="--")    
    plt.title(title)
    
    #plt.xlabel('$\\log(FMR^{dev})$')
    plt.ylabel('$FDR(\\tau)$', fontsize=18)
    plt.xlabel('x', fontsize=18)
    #plt.ylim((0, max([max(transform(f)) for f in f_taus])))
    plt.ylim((0.0, 1.1))
    plt.xticks(np.arange(len(far_thresholds)+1), [""]+[f"$10^{i}$" for i in range(1, len(far_thresholds)+1)])
    plt.xlim(0.95,len(far_thresholds)-1+0.05)
    plt.grid(True)
    
    plt.legend()
    #plt.savefig("XUXA.pdf")
    plt.show()

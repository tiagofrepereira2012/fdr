import matplotlib.pyplot as plt
import bob.measure
import numpy as np
import bob.measure
import tabulate
import scoring
import fairness_metric

def plot_simple_scatter(features):

    fig, ax = plt.subplots()

    plt.scatter(features[:, 0], features[:, 1], marker=".")
    plt.show()


def plot_simple_box_plot(scores, labels, title):

    fig, ax = plt.subplots()

    bp = ax.boxplot(
        scores,
        patch_artist=True,
        showfliers=False,
        labels=labels,
        vert=False,
    )

    # Shrink the plot and set the legend outside
    box = ax.get_position()
    plt.legend()
    plt.title(title)
    plt.legend(loc="upper left", bbox_to_anchor=(0.75, 2.55), fontsize=6)
    ax.set_xlabel("Scores")
    plt.show()


def plot_cohort_box_plot(impostors_per_demographic, genuines_per_demographic, labels, thresholds, far_thresholds):
    import matplotlib.ticker as mticker
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    
    def _color_boxplot(bp, color):
        for patch in bp["boxes"]:
            patch.set_facecolor(color)

    fig, ax = plt.subplots()
    red_square = dict(markerfacecolor="tab:red", marker="s")
    
    bp = ax.boxplot(
            impostors_per_demographic,
            patch_artist=True,            
            showfliers=False,
            labels=labels,
            vert=False,
        )
    _color_boxplot(bp, "tab:red")

    bp1 = ax.boxplot(
            genuines_per_demographic,
            patch_artist=True,            
            showfliers=False,
            labels=labels,
            vert=False,
        )
    _color_boxplot(bp1, "tab:blue")
    
    #[ax.axvline(T, linestyle="--", label=f"FMR {far_thresh} in the dev set") for T in thresholds]
    line_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive']
    [ax.axvline(T, linestyle="--", label=f"$x={f._formatSciNotation('%1.10e' % x)}$", c=c) for T,x,c in zip(thresholds,far_thresholds,line_colors)]

    # Shrink the plot and set the legend outside
    box = ax.get_position()        
    plt.legend(loc="upper left")
    #plt.legend(loc="upper right", bbox_to_anchor=(0.75, 2.55), fontsize=6)    
    ax.set_xlabel("Scores", fontsize=20)
    ax.set_ylabel("Demographics", fontsize=20)
    plt.show()

    
def plot_scatter(enroll_features, probe_features):
    
    fig, ax = plt.subplots()
    colors = ['r','g','b','y']
    markers = ['.','^','o']
    for i, (demographic_features_e,demographic_features_p) in enumerate(zip(enroll_features, probe_features)):
        impostors = []
        genuines = []
        
        for j,(e,p) in enumerate(zip(enroll_features[demographic_features_e], probe_features[demographic_features_p])):
            
            plt.scatter(probe_features[demographic_features_p][p][:,0],
                        probe_features[demographic_features_p][p][:,1],
                        marker='.',
                        c=colors[i])
    plt.show()
    
    
    

def plot_histograms(scores, title):
    fig, ax = plt.subplots()

    for s,c in zip(scores,['g','b']):
    # the histogram of the data
        n, bins, patches = plt.hist(s, 50, density=True, alpha=0.5, facecolor=c)

    plt.grid(True)
    plt.title(title)
    plt.show()
    
    

def compute_FMR_FNMR_FDR(impostors_per_demographic_eval, genuines_per_demographic_eval, thresholds, far_thresholds,tablefmt="plain"):
    
    def generate_raw_table(performance_dict):
        """
        Generates a simple table where axis X and Y are respectivelly the values of demographics        
        and \tau respectivelly
        """

        demographics_vs_fmerit = np.array([np.round(performance_dict[d],4) for d in performance_dict])        
        return demographics_vs_fmerit

    def generate_tabulate_table(raw_table, header, demographics):
        """
        """
        extended_table = np.concatenate((np.array([demographics]).T, raw_table), axis=1)
        return tabulate.tabulate(extended_table, headers=header, tablefmt=tablefmt)

    def generate_fdr_table(fmrs_taus, fnmrs_taus):
        fdrs = []
        for i in range(fmrs_taus.shape[1]):
            fdr_tau = fairness_metric.compute_fdr(fmrs_taus[:,i],fnmrs_taus[:,i])
            fdrs.append(fdr_tau)

        return fdrs

    def generate_fdr_table_tabulate(fdr_raw_table, header):
        return tabulate.tabulate([fdr_raw_table], headers=header, tablefmt=tablefmt)

    FMR_raw, FNMR_raw = scoring.compute_fmr_fnmr_multiple_threshold(impostors_per_demographic_eval, genuines_per_demographic_eval, thresholds)
    header = [""] + [k for k in far_thresholds]
    demographics = [k for k in FMR_raw]
    
    print("##############################################")
    print("################## FMR #######################")
    print("##############################################")
    raw_table_fmr = generate_raw_table(FMR_raw)    
    print(generate_tabulate_table(raw_table_fmr, header, demographics))
    
    print("##############################################")
    print("################## FNMR #######################")
    print("##############################################")

    raw_table_fnmr = generate_raw_table(FNMR_raw)
    print(generate_tabulate_table(raw_table_fnmr, header, demographics))
    
    
    print("##############################################")
    print("################## FDR #######################")
    print("##############################################")
    fdr_raw_table = generate_fdr_table(raw_table_fmr, raw_table_fnmr)
    print(generate_fdr_table_tabulate(fdr_raw_table, far_thresholds))
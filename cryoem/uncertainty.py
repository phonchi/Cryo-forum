import numpy as np
import matplotlib.pyplot as plt

def sum_bingham_dispersion_coeff(A):
    if len(A.shape) == 2:
      A = A.reshape(1,4,4)
    els = np.linalg.eigvalsh(A)
    min_el = els[:,0]
    I = np.repeat(np.eye(4).reshape(1,4,4), A.shape[0], axis=0)
    return np.trace(-A + I*min_el[:,None,None], axis1=1, axis2=2)

def max_dispersion_coeff(A):
    if len(A.shape) == 2:
      A = A.reshape(1,4,4)
    els = np.linalg.eigvalsh(A)
    
    lamb1A, lamb2A, lamb3A, lamb4A = els[:,0], els[:,1], els[:,2], els[:,3]
    
    lamb1 = -lamb4A + lamb1A
    lamb2 = -lamb3A + lamb1A
    lamb3 = -lamb2A + lamb1A
    return np.max([lamb1, lamb2, lamb3], axis=0)

def first_eig_gap(A):
    el = np.linalg.eigvalsh(A)
    spacings = np.diff(el, axis=1)
    return spacings[:, 0]

def wigner_log_likelihood_measure(A, reduce=False):
    el, _ = np.linalg.eig(A)
    el.sort(axis=1)
    spacings = np.diff(el, axis=1)
    lls = np.log(spacings) - 0.25*np.pi*(spacings**2)
    if reduce:
        return np.sum(lls, axis=1).mean()
    else:
        return np.sum(lls, axis=1)
        
def compute_threshold(A, uncertainty_metric_fn=first_eig_gap, quantile=0.75):
    stats = uncertainty_metric_fn(A)
    return np.quantile(stats, quantile)
  
def _scatter(ax, x, y, title, color='tab:red', marker=".", size =4, rasterized=False):
    ax.scatter(x, y, color=color, s=size, marker=marker, label=title, rasterized=rasterized)
    return

def _create_scatter_plot(thresh, thresh_label, lls, errors, labels, xlabel, ylim=None, legend=True, ylabel=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(10,6)
    ax.axvline(thresh, c='k', ls='--', lw=0.75, label=thresh_label)
    colors = ['tab:orange','grey']
    markers = ['.', '+']
    for i, (ll, error, label) in enumerate(zip(lls, errors, labels)):
        _scatter(ax, ll, error, label, color=colors[i], size=1, marker=markers[i], rasterized=True)
    if legend:
        ax.legend(loc='upper left', markerscale=5.0, fontsize=14)
    ax.grid(True, which='both', color='tab:grey', linestyle='--', alpha=0.5, linewidth=0.5)
    if ylabel:
        ax.set_ylabel('Error', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    #ax.set_yscale('log')
    #ax.set_xscale('symlog')
    #ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    ax.set_ylim(ylim)
    return fig
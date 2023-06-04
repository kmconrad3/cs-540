from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0)
    return x

def get_covariance(dataset):
    cov = (np.dot(np.transpose(dataset), dataset)/(len(dataset)-1))
    return cov

def get_eig(S, m):
    SS = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    lamb = np.diag( np.sort(SS[0])[::-1] )
    U = np.fliplr(SS[1])
    return lamb, U

def get_eig_prop(S, prop):
    SS= eigh(S , subset_by_value=[prop*S.trace(), np.inf])
    lamb = np.diag( np.sort(SS[0])[::-1] )
    U = np.fliplr(SS[1])
    return lamb, U

def project_image(image, U):
    sum_aij= np.dot(np.transpose(U), image)
    return np.dot( U, sum_aij )

def display_image(orig, proj):
    fig, axs = plt.subplots(1,2, sharey=True)
    im_o= axs[0].imshow( np.reshape(orig, (32,32), order="F"), aspect='equal') 
    im_p= axs[1].imshow( np.reshape(proj, (32,32), order="F"), aspect='equal')
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    fig.colorbar(im_o, ax=axs[0], shrink=0.7)
    fig.colorbar(im_p, ax=axs[1], shrink=0.7)
    plt.show()
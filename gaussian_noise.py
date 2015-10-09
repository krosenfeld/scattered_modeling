'''
Fit Gaussian model for uncorrelated, Gaussian noise.
'''
import numpy as np
import matplotlib.pyplot as plt
import emcee
import pdb
from source_model import gaussianModel as model
import time
import corner

plt.rcParams['image.cmap'] = 'hot'

def lnprior(theta):
    ftot,fwhm = theta
    if 0 < ftot and 0.0 < fwhm:
        #return -0.5*(ftot-3)**2/1. - 0.5*(fwhm-37.)**2/5.**2
        return 0.0
    return -np.inf

def lnlike(theta,m,d,w):
    m.set_all(*theta)
    return -0.5 * np.sum(w*(m.source-d)**2)

def lnprob(theta,m,d,w):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lnlike(theta,m,d,w) + lp

if __name__ == '__main__':

    # create our model
    nx = 128
    dx = 1.
    ftot = 3.
    fwhm = 37.
    snr = 100
    m = model(ftot,fwhm,nx,dx)
    #w = m/m.sum()

    # add noise
    noise = m.source.max()/snr*np.random.randn(m.nx,m.nx)
    mNoisy = m.source + noise
    w = 1./noise.std()**2

    # initialize sampler
    initial = [ftot,fwhm]
    ndim = len(initial)
    nwalkers = 100
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=[m,mNoisy,w])

    # burn
    tic = time.time()
    p0 = [initial + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(p0,100)
    print 'fit took %0.2fs' % (time.time()-tic)

    # results
    results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(sampler.flatchain, [16, 50, 84],

                                            axis=0)))
    for r in results:
        print r
    bestfit = sampler.flatchain[np.argmax(sampler.flatlnprobability),:]
    print 'bestfit %0.2f, %0.2f' % (bestfit[0],bestfit[1])

    m.set_all(*bestfit)
    fit = m.source

    # figures
    plt.subplot(121); plt.imshow(mNoisy)
    plt.subplot(122); plt.imshow(fit)
    fig = corner.corner(sampler.flatchain, labels=["$flux$", "$fwhm$"],
                      truths=[ftot,fwhm])

    plt.show()

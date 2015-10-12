import slimscat
import numpy as np
from source_model import gaussianModel as model
import matplotlib.pyplot as plt
import time
import emcee
import corner

#bestfit 3.02, 35.23
#takes 337s

plt.rcParams['image.cmap'] = 'hot'

def initialize(m,dx,screenfile='gaussian_screen.bin'):
    # generate screen
    slimscat.generate_screen(screenfile=screenfile,dx=dx)
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def addNoise(m,dx,screenfile='gaussian_screen.bin'):
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def getKernel(screenfile='gaussian_screen.bin'):
    ''' isotropic case '''
    hdr = slimscat.fetch_hdr(screenfile=screenfile)
    theta = (hdr['wavelength'] * np.log(4)**0.5) / \
            (hdr['r0']*np.pi*(1+hdr['m']))
    sigma = (theta/(2*np.sqrt(2*np.log(2))))
    return sigma*206265*1e6/hdr['dx']

def lnprior(theta):
    ftot,fwhm = theta
    if 0 < ftot and 0.0 < fwhm:
        return 0.0
    return -np.inf

def lnlike(theta,m,d,w,sigma):
    m.set_all(*theta)
    return -0.5 * np.sum(w*(m.source-d)**2)
    #mBroad = m.broaden(sigma)
    #return -0.5 * np.sum(w*(mBroad-d)**2)

def lnprob(theta,*args):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lnlike(theta,*args) + lp

if __name__ == '__main__':
    # create model
    ftot = 3.
    fwhm = 37.
    snr = 100
    dx = 4; nx = 32;
    m = model(ftot,fwhm,nx=nx,dx=dx)
    #mNoisy = initialize(m.source,dx)
    mNoisy = addNoise(m.source,dx)
    noise = mNoisy - m.source

    plt.subplot(121); plt.imshow(m.source)
    plt.subplot(122); plt.imshow(mNoisy)

    plt.figure(); plt.imshow(noise)
    #plt.show()

    # fetch scattering kernel
    sigma = getKernel()
    kernel = m.broaden(sigma)

    # estimate noise
    #w = 1./noise.std()**2
    w = m.source.sum()/np.sum(noise**2*m.source)
    # initialize sampler
    #initial = np.array([ftot, fwhm])
    initial = [  3.,  40.5]
    nwalkers = 100
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
      for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, \
            args=[m,mNoisy,w,sigma])

    # run
    tic = time.time()
    sampler.run_mcmc(p0,1000)
    print 'fit took %0.2fs' % (time.time()-tic)

    # results
    samples = sampler.flatchain
    results = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(samples, [16, 50, 84],
                                                    axis=0)))
    for r in results:
        print r

    bestfit = samples[np.argmax(sampler.flatlnprobability),:]
    print 'bestfit %0.2f, %0.2f' % (bestfit[0],bestfit[1])

    mFit = model(*bestfit,nx=nx,dx=dx)
    mFit.broaden(sigma)

    # figures
    plt.subplot(121); plt.imshow(mFit.source-m.source)
    plt.subplot(122); plt.imshow(mFit.source)


    fig = corner.corner(samples, labels=["$flux$", "$fwhm$"],
                      truths=[ftot,fwhm])
    plt.savefig('gaussian_slim.png')
    #plt.show()

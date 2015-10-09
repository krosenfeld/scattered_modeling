import slimscat
import numpy as np
from source_model import gaussianModel as model
from source_model import getKernel
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import george
from george import kernels

plt.rcParams['image.cmap'] = 'hot'
__screenfile__ = 'gaussian_screen.bin'

# takes about 50s per iteration

def initialize(m,dx,screenfile='gaussian_screen.bin'):
    # generate screen
    slimscat.generate_screen(screenfile=screenfile)
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def addNoise(m,dx,screenfile='gaussian_screen.bin'):
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def lnprior(theta):
    lna, lntau, ftot,fwhm = theta
    if -5 < lna < 5 and  -5 < lntau < 5 and 0 < ftot and 0.0 < fwhm:
        return 0.0
    return -np.inf

def lnlike(theta,m,d,w,sigma,t):
    a, tau = np.exp(theta[:2])
    gp = george.GP(0 * kernels.Matern32Kernel(tau,ndim=2))
    gp.compute(t, 1/np.sqrt(w))
    print theta
    print m.source.size/2*np.log(2*np.pi)
    m.set_all(*theta[2:])
    mBroad = m.broaden(sigma).flatten()
    return gp.lnlikelihood(d.flatten() - mBroad) + \
        0.5 * m.source.size*np.log(2*np.pi) + \
        0.5 * mNoisy.size*np.log(1./w)

def lnprob(theta,*args):
    ''' minimize this '''
    lp = lnprior(theta)
    ans = (lp + lnlike(theta, *args)) \
     if np.isfinite(lp) else -np.inf
    return -1 * ans

if __name__ == '__main__':
    # create model
    ftot = 3.
    fwhm = 37.
    snr = 100
    dx = 1
    m = model(ftot,fwhm,dx=dx)
    #mNoisy = initialize(m,dx)
    mNoisy = addNoise(m.source,dx)
    noise = mNoisy - m.source

    plt.subplot(121); plt.imshow(m.source)
    plt.subplot(122); plt.imshow(mNoisy)

    plt.figure(); plt.imshow(noise)
    plt.show()

    # fetch scattering kernel
    sigma = getKernel(__screenfile__)

    # estimate noise
    w = m.source.sum()/np.sum(noise**2*m.source)

    t1 = np.tile(np.arange(m.nx),m.nx)
    t0 = np.repeat(np.arange(m.nx),m.nx,axis=0)
    t = np.hstack([t0[:,np.newaxis],t1[:,np.newaxis]])

    # fit
    w = 1./noise.std()**2

    tic = time.time()
    initial = np.array([0.,0.,ftot, fwhm])
    res = minimize(lnprob,initial,\
       args=(m,mNoisy,w,sigma),\
       method='Nelder-Mead',\
       options={'disp':True,'maxiter':int(10)})
    print 'optimization took %0.2fs' % (time.time()-tic)

    m.set_all(*res.x)
    fit = m.source

    # figures
    plt.subplot(121); plt.imshow(mNoisy)
    plt.subplot(122); plt.imshow(fit)
    plt.show()

    plt.show()

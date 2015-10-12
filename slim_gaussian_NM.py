import slimscat
import numpy as np
from source_model import gaussianModel as model
from source_model import getKernel
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize

#Optimization terminated successfully.
#         Current function value: 1726.569850 
#         Iterations: 35
#         Function evaluations: 68
#optimization took 0.23s

plt.rcParams['image.cmap'] = 'hot'
__screenfile__ = 'gaussian_screen.bin'


def initialize(m,dx,screenfile='gaussian_screen.bin'):
    # generate screen
    slimscat.generate_screen(screenfile=screenfile,dx=dx)
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def addNoise(m,dx,screenfile='gaussian_screen.bin'):
    return slimscat.run_slimscat(m,dx,screenfile=screenfile)


def lnprior(theta):
    ftot,fwhm = theta
    if 0 < ftot and 0.0 < fwhm:
        return 0.0
    return -np.inf

def lnlike(theta,m,d,w,sigma):
    m.set_all(*theta)
    #return -0.5 * np.sum(w*(m.source-d)**2)
    mBroad = m.broaden(sigma)
    return -0.5 * np.sum(w*(mBroad-d)**2)

def lnprob(theta,*args):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return -1 * (lnlike(theta,*args) + lp)

if __name__ == '__main__':
    # create model
    ftot = 3.
    fwhm = 37.
    snr = 100
    #dx = 1; nx = 128;
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
    sigma = getKernel(__screenfile__)
    kernel = m.broaden(sigma)

    # estimate noise
    w = m.source.sum()/np.sum(noise**2*m.source)

    tic = time.time()
    initial = np.array([ftot, fwhm])
    res = minimize(lnprob,initial,\
        args=(m,mNoisy,w,sigma),\
        method='Nelder-Mead',\
        options={'disp':True,'maxiter':int(100*100)})
    print 'optimization took %0.2fs' % (time.time()-tic)

    print 'result:', res.x
    m.set_all(*res.x)
    fit = m.source

    # figures
    plt.subplot(121); plt.imshow(mNoisy)
    plt.subplot(122); plt.imshow(fit)
    plt.show()

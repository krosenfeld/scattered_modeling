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
plt.rcParams['image.interpolation'] = 'none'
__screenfile__ = 'gaussian_screen.bin'

#array([-11.88828322,  -1.37461595,   2.98841522,  40.49427074])

#result: [-11.88432122  -1.36211244   2.987896    33.88887017]
#result: [-11.88438505  -1.36204286   2.98784193  33.88807813]



def initialize(m,dx,screenfile='gaussian_screen.bin'):
    # generate screen
    slimscat.generate_screen(screenfile=screenfile,dx=dx)
    return slimscat.run_slimscat(m.source,dx,screenfile=screenfile)

def addNoise(m,dx,screenfile='gaussian_screen.bin'):
    ''' 
    Remember that if you aren't careful, this can return an
    image with a non-matching resolution!
    '''

    return slimscat.run_slimscat(m,dx,screenfile=screenfile)

def lnprior(theta):
    return 0.
    lna, lntau, ftot,fwhm = theta
    if -20 < lna < 5 and  -5 < lntau < 10 and 0 < ftot and 0.0 < fwhm:
        return 0.0
    return -np.inf

def lnlike(theta,m,d,w,sigma,t):
    print theta
    m.set_all(*theta[2:])
    a, tau = np.exp(theta[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau,ndim=2))
    #gp.compute(t, 1/np.sqrt(w))  # doesn't work.
    _m = m.broaden(sigma).flatten()
    #_m = m.source.flatten()
    gp.compute(t, _m/_m.sum()/np.sqrt(w))
    return gp.lnlikelihood(d.flatten() - _m)

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
    dx = 4.
    nx = 32
    m = model(ftot,fwhm,dx=dx,nx=nx)
    #mNoisy = initialize(m,dx)
    mNoisy = addNoise(m.source,dx)
    mTrue = m.source
    noise = mNoisy - m.source
    mTrue = m.source.copy()

    # check that our model and simulation have matching resolutions
    hdr = slimscat.fetch_hdr(__screenfile__)
    assert hdr['dx'] == dx

    #plt.subplot(121); plt.imshow(m.source); plt.subplot(122); plt.imshow(mNoisy)
    #plt.figure(); plt.imshow(noise)
    #plt.show()

    # fetch scattering kernel
    sigma = getKernel(__screenfile__)

    # sample positions
    t1 = np.tile(np.arange(m.nx),m.nx)
    t0 = np.repeat(np.arange(m.nx),m.nx,axis=0)
    t = np.hstack([t0[:,np.newaxis],t1[:,np.newaxis]])

    # fit
    w = m.source.sum()/np.sum(noise**2*m.source)

    tic = time.time()
    #initial = np.array([np.log(1./w),np.log(0.5),ftot, fwhm])
    initial = np.array([np.log(1./w),np.log(2),ftot, fwhm])
    res = minimize(lnprob,initial,\
       args=(m,mNoisy,w,sigma,t),\
       method='Nelder-Mead',\
       options={'disp':True,'maxiter':int(1000)})
    print 'optimization took %0.2fs' % (time.time()-tic)

    print 'result:',res.x

    #tic = time.time()
    #args = (m,mNoisy,w,sigma,t)
    #f = lnprob(initial,*args)
    #print '1 exec took %0.2f' % (time.time() - tic)

    # best fit noise model
    #p = initial
    p = res.x
    m.set_all(*p[2:])
    fit = m.source
    mBroad = m.broaden(sigma)
    tic = time.time()
    a,tau = np.exp(p[:2])
    k = a * kernels.Matern32Kernel(tau,ndim=2)
    gp = george.GP(k)
    #gp.compute(t, 1/np.sqrt(w))
    gp.compute(t, mBroad.flatten()/mBroad.sum()/np.sqrt(w))
    mu = gp.predict(mNoisy.flatten()-mBroad.flatten(),t,mean_only=True)
    print 'one exec took %0.2fs' % (time.time() - tic)

    # figures
    plt.figure()
    plt.imshow(mu.reshape(fit.shape))
    #plt.subplot(121); plt.imshow(mTrue); plt.title('true')
    #plt.subplot(122); plt.imshow(fit); plt.title('fit')

    plt.figure()
    vmin,vmax = 0,mNoisy.max()
    plt.subplot(122); plt.imshow(mNoisy,vmin=0,vmax=vmax); plt.title('noisy')
    plt.subplot(121); plt.imshow(mu.reshape(fit.shape) + m.broaden(sigma),vmin=vmin,vmax=vmax); plt.title('broad fit + noise')

    print np.sum((mNoisy - (mu.reshape(fit.shape) + m.broaden(sigma)))**2)*w
    
    plt.show()

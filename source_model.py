import numpy as np
import slimscat
from scipy.ndimage.filters import gaussian_filter

def getKernel(screenfile):
    ''' isotropic case '''
    hdr = slimscat.fetch_hdr(screenfile=screenfile)
    theta = (hdr['wavelength'] * np.log(4)**0.5) / \
            (hdr['r0']*np.pi*(1+hdr['m']))
    sigma = (theta/(2*np.sqrt(2*np.log(2))))
    return sigma*206265*1e6/hdr['dx']

class SourceModel(object):

    def __init__(self,nx=128,dx=1.):
        self.nx = nx
        self.dx = dx
        self.source = None

    def generate(self):
        raise NotImplementedError("Generate method must be implemented"
        "by subclass")

    def broaden(self,sigma):
        '''
        Broaden by a Gaussian.
        '''
        return gaussian_filter(self.source,sigma,mode='constant')

class gaussianModel(SourceModel):
    def __init__(self,ftot,fwhm,*args,**kwargs):
        super(gaussianModel,self).__init__(*args,**kwargs)
        self.set_all(ftot,fwhm)

    def set_all(self,ftot,fwhm):
        self.ftot=ftot
        self.fwhm=fwhm
        self.source = self.generate()

    def generate(self):
        sigma = self.fwhm/(2*np.sqrt(2*np.log(2)))
        xx,yy = np.meshgrid(self.dx*(np.arange(self.nx)-self.nx/2),\
                self.dx*(np.arange(self.nx)-self.nx/2))
        self.source = np.exp(-1./(2.*sigma**2)*(xx**2 + yy**2))
        self.source *= self.ftot / self.source.sum()
        return self.source

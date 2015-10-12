import numpy as np

class Model(object):
    def __init__(self,p,nx,dx):
        self.p=p # fraction when normalized visibility is 1
        self.nx=nx
        self.dx=dx
        self.I=None
        self.Q=None
        self.U=None
        self.mt=None
        self.generate_stokes()
        self.generate_mt()

    def generate_stokes(self):
        raise NotImplementedError("Must be implemented by subclass")

    def generate_mt(self):
        It = np.fft.fft2(self.I)
        Qt = np.fft.fft2(self.Q)
        Ut = np.fft.fft2(self.U)
        self.mt = np.fft.fftshift((Qt + 1j*Ut)/It)

class gaussian(Model):
    def __init__(self,flux,fwhm,pol,nx,dx):
        self.flux=flux
        self.fwhm=fwhm
        super(gaussian,self).__init__(pol,nx,dx)

    def generate_stokes(self):

        # I
        sigma = self.fwhm/(2*np.sqrt(2*np.log(2)))
        xx,yy = np.meshgrid(self.dx*(np.arange(self.nx)-self.nx/2),\
                self.dx*(np.arange(self.nx)-self.nx/2))
        self.I = np.exp(-1./(2.*sigma**2)*(xx**2 + yy**2))
        self.I *= 1. / self.I.sum() # normalized visibility is 1

        self.Q = self.p*self.I/np.sqrt(2)
        self.U = self.Q.copy()

    def set_fwhm(self,flux):
        self.fwhm = fwhm


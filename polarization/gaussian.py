from models import gaussian as model
import numpy as np
import matplotlib.pyplot as plt
from palettable import colorbrewer

#plt.rcParams['image.cmap'] = colorbrewer.sequential.Blues_9.mpl_colormap
cmap = colorbrewer.sequential.Blues_9.mpl_colormap

if __name__ == "__main__":
    nx,dx = 32, 2.
    flux = 3.
    fwhm = 37.
    p = 0.052

    m = model(flux,fwhm,p,nx,dx)

    # plot model
    psi = np.arctan(m.U/m.Q)/2.
    plt.imshow(m.I,cmap=cmap)
    plt.quiver(np.cos(psi)*m.I,np.sin(m.I),headwidth=0)

    #plt.ion()
    plt.show()

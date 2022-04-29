####################################################################################################
#
# script : processSED.py
#
# process the library of SEDs and project them onto the filters, (for the mean fct of the GP)
# (which may take a few minutes depending on the settings you set):
#
# output file : sed_name + '_fluxredshiftmod.txt'
######################################################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.special as sc

from delight.io import *
from delight.utils import *

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')

def gammaincc_ext(k, x):
    """

    gammaincc_ext(k, x)
    
    Compute the regularized upper incomplete gamma function as defined in scipy.special.
    If k>=0, returns scipy.special.gammaincc(k, x)
    
    Else, uses the recursive formula in order to extend the definition of the function to negative k.
    """
    
    if k>=0.:
        y = sc.gammaincc(k, x)
    else: #recursive call
        y = ( 1/(k*sc.gamma(k)) ) * ( gammaincc_ext(k+1., x)*sc.gamma(k+1.) - np.float_power(x, k)*np.exp(-x) )
    return y

def computeBias(k=1.0, xmin=1.0e0):
    """
    
    gammaincc_ext(k, x)
    
    Compute the bias of the Luminosity function that leads to a gamma distribution, when the lower bound is a luminosity threshold > 0.
    Available for negative k, however this leads to a negative bias which is not usable in practical cases.
    For usable cases, k must be strictly positive, which is inconsistent with the measurements of the local group that lead to k = -0.25.
    """
    
    return gammaincc_ext(k+1.0, xmin) / gammaincc_ext(k, xmin)

def processSEDs(configfilename, bias=False, k=1.0, xmin=1.0e0):
    """

    processSEDs(configfilename, bias=False, k=1.0, xmin=1.0e0)

    Compute the The Flux expected in each band for  redshifts in the grid
    : input file : the configuration file

    :return: produce the file of flux-redshift in bands. If bias is True, the flux-redshift templates are "flattened" at higher z
    to try and include the effect of Eddington-Malmquist detection bias that favors detection of brighter galaxies near the detection threshold.
    """



    logger.info("--- Process SED ---")

    # decode the parameters
    params = parseParamFile(configfilename, verbose=False, catFilesNeeded=False)
    #print(f"configfilename: {configfilename}")
    #print("\n\n\n\n\n\nFULL LIST OF PARAMS:")
    #print(params)
    bandNames = params['bandNames']
    dir_seds = params['templates_directory']
    dir_filters = params['bands_directory']
    lambdaRef = params['lambdaRef']
    sed_names = params['templates_names']
    #fmt = '.dat'
    sed_fmt = params['sed_fmt']
    
    # Luminosity Distnace
    DL = approx_DL()

    #redshift grid
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    numZ = redshiftGrid.size

    # Loop over SEDs
    # create a file per SED of all possible flux in band
    for sed_name in sed_names:
        tmpsedname = sed_name + "." + sed_fmt
        path_to_sed = os.path.join(dir_seds, tmpsedname)
        seddata = np.genfromtxt(path_to_sed)
        seddata[:, 1] *= seddata[:, 0] # SDC : multiply luminosity by wl ?
        # SDC: OK if luminosity is in wl bins ! To be checked !!!!
        ref = np.interp(lambdaRef, seddata[:, 0], seddata[:, 1])
        seddata[:, 1] /= ref  # normalisation at lambdaRef
        sed_interp = interp1d(seddata[:, 0], seddata[:, 1]) # interpolation

        # container of redshift/ flux : matrix n_z x n_b for each template
        # each column correspond to fluxes in the different bands at a a fixed redshift
        # redshift along row, fluxes along column
        # model of flux as a function of redshift for each template
        f_mod = np.zeros((redshiftGrid.size, len(bandNames)))

        # Loop over bands
        # jf index on bands
        for jf, band in enumerate(bandNames):
            fname_in = dir_filters + '/' + band + '.res'
            data = np.genfromtxt(fname_in)
            xf, yf = data[:, 0], data[:, 1]
            #yf /= xf  # divide by lambda
            # Only consider range where >1% max
            ind = np.where(yf > 0.01*np.max(yf))[0]
            lambdaMin, lambdaMax = xf[ind[0]], xf[ind[-1]]
            norm = np.trapz(yf/xf, x=xf) # SDC: probably Cb

            # iz index on redshift
            for iz in range(redshiftGrid.size):
                opz = (redshiftGrid[iz] + 1)
                xf_z = np.linspace(lambdaMin / opz, lambdaMax / opz, num=5000)
                yf_z = interp1d(xf / opz, yf)(xf_z)
                ysed = sed_interp(xf_z)
                f_mod[iz, jf] = np.trapz(ysed * yf_z, x=xf_z) / norm
                f_mod[iz, jf] *= opz**2. / DL(redshiftGrid[iz])**2. / (4*np.pi)
        suffix=''
        if bias:
            f_mod *= computeBias(k, xmin)
            suffix = '-bias'
        # for each SED, save the flux at each redshift (along row) for each
        tmpoutpath = os.path.join(dir_seds, sed_name + '_fluxredshiftmod'\
                                  + suffix + '.txt')
        np.savetxt(tmpoutpath, f_mod)


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start processSEDs.py"
    logger.info(msg)
    logger.info("--- Process SEDs ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    processSEDs(sys.argv[1])

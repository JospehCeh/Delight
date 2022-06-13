##################################################################################################################################
#
# script : delight-learn_paramSpec.py
#
#  input : 'training_catFile'
#  output : localData or reducedData usefull for Gaussian Process in 'training_paramFile'
#  - find the normalisation of the flux and the best galaxy type
############################################################################################################################
import sys
import numpy as np
from delight.io import *
from delight.utils import *
from photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import scipy.special as sc
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import cost, Minuit
# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares
# display iminuit version
import iminuit

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
    
    #return gammaincc_ext(k+1.0, xmin) / gammaincc_ext(k, xmin)
    return sc.gammaincc(k+1.0, xmin) / sc.gammaincc(k, xmin)

def delightLearn_paramSpec(configfilename, V_C=-1.0, V_L=-1.0, alpha_C=-1.0, alpha_L=-1.0, plot=False, autofitTemplates=False, uniformFmod=False):
    """

    :param configfilename:
    :return:
    """
    
    threadNum = 0
    numThreads = 1

    #parse arguments

    params = parseParamFile(configfilename, verbose=False)

    if threadNum == 0:
        logger.info("--- DELIGHT-LEARN ---")

    # Read filter coefficients, compute normalization of filters
    bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
    numBands = bandCoefAmplitudes.shape[0]

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))

    msg= 'Number of Training Objects ' + str(numObjectsTraining)
    logger.info(msg)

    firstLine = int(threadNum * numObjectsTraining / numThreads)
    lastLine = int(min(numObjectsTraining,(threadNum + 1) * numObjectsTraining / numThreads))
    numLines = lastLine - firstLine
    
    crossValidate = params['training_crossValidate']
    
    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    
    if not autofitTemplates:
        f_mod = readSEDs(params)
        if uniformFmod:
            uniform = np.mean(f_mod)
            f_mod[:,:] = uniform
    else:
        ######################################################
        ## Begin fit a photodetection bias on training data ##
        ######################################################
        
        #x_data, DUM_normedRefFlux, DUM_bands, y_dataArr, yerr_dataArr, DUM_bandsCV, DUM_fluxesCV, DUM_fluxesVarCV, DUMX, DUMY, DUMYVAR =\
        #getDataFromFile(params, firstLine, lastLine, prefix="training_", getXY=False, CV=False)
        
        #x_data, DUM_normedRefFlux, DUM_bands, y_dataArr, yerr_dataArr = getDataFromFile(params, firstLine, lastLine, prefix="training_", getXY=False, CV=False)
        
        trainingDataIterFit = getDataFromFile(params, firstLine, lastLine, prefix="training_", getXY=False, CV=False)
        x_data = np.empty(numObjectsTraining)
        y_dataArr = np.empty((numObjectsTraining, numBands))
        yerr_dataArr = np.empty((numObjectsTraining, numBands))
        loc=-1
        for z, _DUM_normedRefFlux, _DUM_bands, flux, fluxerr, *_DUMOTHER_ in trainingDataIterFit:
            loc+=1
            x_data[loc] = z
            y_dataArr[loc, :] = flux
            yerr_dataArr[loc, :] = fluxerr
            
        print("DEBUG: ", loc, x_data[-1], y_dataArr[-1, :])

        f_mod = np.zeros((len(params['templates_names']),
                          len(params['bandNames'])), dtype=object)
        
        Z_GRID = np.linspace(redshiftGrid[0], redshiftGrid[-1], 10)

        for jf in range(len(params['bandNames'])):
            y_data = -2.5*np.log10(y_dataArr[:, jf])
            yerr_data = np.abs(-2.5*np.log10(1+yerr_dataArr[:, jf]/y_dataArr[:, jf]))
            Z_IDX = [ np.where(np.logical_and(x_data >= Z_GRID[i], x_data < Z_GRID[i+1]))[0]\
                     for i in range(len(Z_GRID)-1)]
            y_av = [ np.mean(y_data[indexes]) for indexes in Z_IDX ]
            y_std = [ np.std(y_data[indexes]) for indexes in Z_IDX ]

            meanInterp = interp1d(Z_GRID[:-1] + 0.5*np.diff(Z_GRID), y_av, bounds_error = False, fill_value=(y_av[0], y_av[-1]))
            stdInterp = interp1d(Z_GRID[:-1] + 0.5*np.diff(Z_GRID), y_std, bounds_error = False, fill_value=(y_std[0], y_std[-1]))

            array_k = np.empty(len(Z_GRID)-1)
            array_xmin = np.empty(len(Z_GRID)-1)

            def averageMag(x_array, B=1.0):#, k=1.0, xmin=1.0):
                magAvg = np.zeros_like(x_array)
                for it, sed_name in enumerate(params['templates_names']):
                    data = np.loadtxt(params['templates_directory'] +
                                      '/' + sed_name + '_fluxredshiftmod.txt')[:, jf]
                    data_interp = interp1d(redshiftGrid, data, bounds_error = False, fill_value=(data[0], data[-1]))
                    magAvg += -2.5*np.log10(B * data_interp(x_array))
                return magAvg/len(params['templates_names']) # np array shaped as redshift and magnitude columns

            lsq_avg = cost.LeastSquares(Z_GRID, meanInterp(Z_GRID), stdInterp(Z_GRID),\
                                        averageMag, loss='soft_l1')

            m0 = Minuit(lsq_avg, B=1.0)#, k=1.0, xmin=xmin)
            m0.errordef = Minuit.LEAST_SQUARES
            m0.limits['B']=(0.01,np.inf)
            m0.migrad()
            #m0.hesse()

            for z_ind in range(len(Z_GRID)-1):
                    def averageMag(x_array, k=1.0, xmin=1.0):
                        magAvg = np.zeros_like(x_array)
                        for it, sed_name in enumerate(params['templates_names']):
                            data = np.loadtxt(params['templates_directory'] +
                                              '/' + sed_name + '_fluxredshiftmod.txt')[:, jf]
                            data_interp = interp1d(redshiftGrid, data, bounds_error = False, fill_value=(data[0], data[-1]))
                            magAvg += -2.5*np.log10(m0.values['B'] * data_interp(x_array) * computeBias(k, xmin))
                        return magAvg/len(params['templates_names']) # np array shaped as redshift and magnitude columns

                    lsq_avg = cost.LeastSquares(x_data[Z_IDX[z_ind]], meanInterp(x_data[Z_IDX[z_ind]]),\
                                                stdInterp(x_data[Z_IDX[z_ind]]), averageMag, loss='soft_l1')

                    xmin=200.0*np.min(np.power(10, -0.4*y_data[Z_IDX[z_ind]]))/np.mean(np.power(10, -0.4*y_data[Z_IDX[z_ind]]))

                    m1 = Minuit(lsq_avg, k=1.0, xmin=xmin)
                    m1.errordef = Minuit.LEAST_SQUARES
                    m1.limits['k']=(0.001,np.inf)
                    m1.limits['xmin']=(0.0,np.inf)
                    m1.migrad()
                    #m1.hesse()
                    array_k[z_ind] = m1.values['k']
                    array_xmin[z_ind] = m1.values['xmin']
            
            k_interp = interp1d(Z_GRID[:-1] + 0.5*np.diff(Z_GRID), array_k, bounds_error = False, fill_value=(array_k[0], array_k[-1])) 
            
            #array_k[0] = array_k[1] ## artifice pour gérer la dimensionnalité des k.
                                    ## Peut-être qu'il vaut carrément mieux ne conserver qu'une valeur... mais laquelle?
            
                
            def averageMag(x_array, B=1.0, xmin=1.0):#, k=1.0):
                magAvg = np.zeros_like(x_array)
                for it, sed_name in enumerate(params['templates_names']):
                    data = np.loadtxt(params['templates_directory'] +
                                      '/' + sed_name + '_fluxredshiftmod.txt')[:, jf]
                    data_interp = interp1d(redshiftGrid, data, bounds_error = False, fill_value=(data[0], data[-1]))
                    magAvg += -2.5*np.log10(B * data_interp(x_array) * computeBias(k_interp(x_array), xmin))
                return magAvg/len(params['templates_names']) # np array shaped as redshift and magnitude columns

            lsq_avg = cost.LeastSquares(Z_GRID, meanInterp(Z_GRID), stdInterp(Z_GRID),\
                                        averageMag, loss='soft_l1')

            m2 = Minuit(lsq_avg, B=m0.values['B'], xmin=np.max(array_xmin[:]))#, k=1.0, xmin=xmin)
            m2.errordef = Minuit.LEAST_SQUARES
            m2.limits['B']=(0.01,np.inf)
            m2.limits['xmin']=(0.0,np.inf)
            m2.migrad()
            #m2.hesse()

            print("DEBUG : B={}, xmin={}, ks={}".format(m2.values['B'], m2.values['xmin'], array_k))
            figDebug, axDebug = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
            axDebug = axDebug.ravel()
            axDebug[0].scatter(x_data, y_dataArr[:, jf], alpha=0.05)
            axDebug[1].plot(redshiftGrid, k_interp(redshiftGrid))
            axDebug[1].scatter(Z_GRID[:-1] + 0.5*np.diff(Z_GRID), array_k)
            for it, sed_name in enumerate(params['templates_names']):
                data = np.loadtxt(params['templates_directory'] +
                                  '/' + sed_name + '_fluxredshiftmod.txt')[:, jf]
                #f_mod[it, jf] = interp1d(redshiftGrid, m2.values['B']*data*computeBias(k_interp(redshiftGrid), m2.values['xmin']),\
                #                         kind='linear', bounds_error=False, fill_value='extrapolate')
                f_mod[it, jf] = interp1d(redshiftGrid, data*computeBias(k_interp(redshiftGrid), m2.values['xmin']),\
                                         kind='linear', bounds_error=False, fill_value='extrapolate')
                axDebug[0].plot(redshiftGrid, data, ls=":")
                axDebug[0].plot(redshiftGrid, f_mod[it, jf](redshiftGrid))
            axDebug[0].set_yscale('log')
        ############################################################################################################
        ## End of photodetection bias fitting process.                                                            ##
        ## Normally, the f_mod function now matches the distorted flux-redshift SEDs instead of the original one. ## 
        ## This is what goes then into the GP.                                                                    ##
        ############################################################################################################
    
    msg ='Thread ' +  str(threadNum) + ' , analyzes lines ' + str(firstLine) + ' , to ' + str(lastLine)
    logger.info(msg)

    DL = approx_DL()
    
    # Default values read from paramFile
    if V_C  < 0:
        V_C = params['V_C']
    if V_L  < 0:
        V_L = params['V_L']
    if alpha_C  < 0:
        alpha_C = params['alpha_C']
    if alpha_L  < 0:
        alpha_L = params['alpha_L']
        
    print("Creation of GP with V_C = {}, V_L = {}, alpha_C = {}, alpha_L = {}.".format(V_C, V_L, alpha_C, alpha_L))
    
    gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,\
                  params['lines_pos'], params['lines_width'],\
                  V_C, V_L,\
                  alpha_C, alpha_L,\
                  redshiftGridGP, use_interpolators=True)

    B = numBands
    numCol = 3 + B + B*(B+1)//2 + B + f_mod.shape[0]
    localData = np.zeros((numLines, numCol))
    fmt = '%i ' + '%.12e ' * (localData.shape[1] - 1)

    loc = - 1
    trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,prefix="training_", getXY=True,CV=crossValidate)

    
    if crossValidate:
        chi2sLocal = None
        bandIndicesCV, bandNamesCV, bandColumnsCV,bandVarColumnsCV, redshiftColumnCV = readColumnPositions(params, prefix="training_CV_", refFlux=False)

    for z, normedRefFlux,\
        bands, fluxes, fluxesVar,\
        bandsCV, fluxesCV, fluxesVarCV,\
        X, Y, Yvar in trainingDataIter1:

        loc += 1

        themod = np.zeros((1, f_mod.shape[0], bands.size))
        for it in range(f_mod.shape[0]):
            for ib, band in enumerate(bands):
                themod[0, it, ib] = f_mod[it, band](z)

        # really calibrate the luminosity parameter l compared to the model
        # according the best type of galaxy
        chi2_grid, ellMLs = scalefree_flux_likelihood(fluxes,fluxesVar,themod,returnChi2=True)

        bestType = np.argmin(chi2_grid)  # best type
        ell = ellMLs[0, bestType]        # the luminosity factor
        X[:, 2] = ell

        gp.setData(X, Y, Yvar, bestType)
        lB = bands.size
        localData[loc, 0] = lB
        localData[loc, 1] = z
        localData[loc, 2] = ell
        localData[loc, 3:3+lB] = bands
        localData[loc, 3+lB:3+f_mod.shape[0]+lB+lB*(lB+1)//2+lB] = gp.getCore()

        if crossValidate:
            model_mean, model_covar = gp.predictAndInterpolate(np.array([z]), ell=ell)
            if chi2sLocal is None:
                chi2sLocal = np.zeros((numObjectsTraining, bandIndicesCV.size))

            ind = np.array([list(bandIndicesCV).index(b) for b in bandsCV])

            chi2sLocal[firstLine + loc, ind] = - 0.5 * (model_mean[0, bandsCV] - fluxesCV)**2 /(model_covar[0, bandsCV] + fluxesVarCV)
        
        # Plot MargLike avec fonction incluse
        quot = (lastLine - firstLine)//3
        if loc % quot == 0 and plot:
            values = np.logspace(-1, 6, 50)
            allMargLike = []
            figMargLike, axMargLike = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
            for val in values:
                allMargLike.append(gp.updateHyperparamatersAndReturnMarglike_continuum(pars=(val, alpha_C)))
            axMargLike.plot(values, allMargLike, label="Varying $V_C$")
            dummy = gp.updateHyperparamatersAndReturnMarglike_continuum(pars=(V_C, alpha_C))
            
            allMargLike = []
            for val in values:
                allMargLike.append(gp.updateHyperparamatersAndReturnMarglike_continuum(pars=(V_C, val)))
            axMargLike.plot(values, allMargLike, label="Varying $alpha_C$")
            dummy = gp.updateHyperparamatersAndReturnMarglike_continuum(pars=(V_C, alpha_C))
            
            allMargLike = []
            for val in values:
                allMargLike.append(gp.updateHyperparamatersAndReturnMarglike_lines(pars=(val, alpha_L)))
            axMargLike.plot(values, allMargLike, label="Varying $V_L$")
            dummy = gp.updateHyperparamatersAndReturnMarglike_lines(pars=(V_L, alpha_L))
            
            allMargLike = []
            for val in values:
                allMargLike.append(gp.updateHyperparamatersAndReturnMarglike_lines(pars=(V_L, val)))
            axMargLike.plot(values, allMargLike, label="Varying $alpha_L$")
            dummy = gp.updateHyperparamatersAndReturnMarglike_lines(pars=(V_L, alpha_L))
            
            axMargLike.set_xlabel("$V_C$, $alpha_C$, $V_L$, $alpha_L$")
            axMargLike.set_ylabel("GP Marginal Likelihood")
            axMargLike.set_yscale('log')
            figMargLike.legend(loc='lower center')
            figMargLike.suptitle("GP marginal likelihood in function of hyperparameters for training galaxy No. {}, z = {}".format(loc, z))


   
    if threadNum == 0:
        reducedData = np.zeros((numObjectsTraining, numCol))

    if crossValidate:
        chi2sGlobal = np.zeros_like(chi2sLocal)
        #comm.Allreduce(chi2sLocal, chi2sGlobal, op=MPI.SUM)
        #comm.Barrier()
        chi2sGlobal = chi2sLocal

    firstLines = [int(k*numObjectsTraining/numThreads) for k in range(numThreads)]
    lastLines = [int(min(numObjectsTraining, (k+1)*numObjectsTraining/numThreads)) for k in range(numThreads)]
    sendcounts = tuple([(lastLines[k] - firstLines[k]) * numCol for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numCol for k in range(numThreads)])

    reducedData = localData
  

    # parameters for the GP process on traniing data are transfered to reduced data and saved in file
    #'training_paramFile'
    if threadNum == 0:
        np.savetxt(params['training_paramFile'], reducedData, fmt=fmt)
        if crossValidate:
            np.savetxt(params['training_CVfile'], chi2sGlobal)


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start Delight Learn.py"
    logger.info(msg)
    logger.info("--- Process Delight Learn ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    delightLearn_paramSpec(sys.argv[1])

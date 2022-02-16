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

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')

def delightLearn_paramSpec(configfilename, V_C=-1.0, V_L=-1.0, alpha_C=-1.0, alpha_L=-1.0):
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

    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)

    f_mod = readSEDs(params)

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))

    msg= 'Number of Training Objects ' + str(numObjectsTraining)
    logger.info(msg)


    firstLine = int(threadNum * numObjectsTraining / numThreads)
    lastLine = int(min(numObjectsTraining,(threadNum + 1) * numObjectsTraining / numThreads))
    numLines = lastLine - firstLine

  
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
    
    gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              V_C, V_L,
              alpha_C, alpha_L,
              redshiftGridGP, use_interpolators=True)

    B = numBands
    numCol = 3 + B + B*(B+1)//2 + B + f_mod.shape[0]
    localData = np.zeros((numLines, numCol))
    fmt = '%i ' + '%.12e ' * (localData.shape[1] - 1)

    loc = - 1
    crossValidate = params['training_crossValidate']
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
        if loc % quot == 0:
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

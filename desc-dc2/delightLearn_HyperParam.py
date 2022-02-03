##################################################################################################################################
#
# script : delight-learn.py
#
#  input : 'training_catFile'
#  output : localData or reducedData usefull for Gaussian Process in 'training_paramFile'
#  - find the normalisation of the flux and the best galaxy type
#
#  Added analysis functionalities to try and understand the influence of hyperparameters during the learning process
#
############################################################################################################################
import sys
import numpy as np
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
import matplotlib.pyplot as plt

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')

def delightLearn_HyperParam(configfilename, hyperParam_name="", hyperParam_list=[]):
    """

    :param configfilename:
    :return:
    """
    # Sensitivity study or normal run?
    sensitivity = False
    
    if (hyperParam_name!="V_C" and hyperParam_name!="V_L" and hyperParam_name!="alpha_C" \
        and hyperParam_name!="alpha_L" and hyperParam_name!="ellSigmaPrior") or len(hyperParam_list)<1 :
        print("Invalid hyperparameter for sensitivity study. Running with default values.")
    else:
        print("Running sensitivity study for hyperparameter "+hyperParam_name+" .")
        sensitivity = True
    
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
    gp = PhotozGP(f_mod, bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
              params['lines_pos'], params['lines_width'],
              params['V_C'], params['V_L'],
              params['alpha_C'], params['alpha_L'],
              redshiftGridGP, use_interpolators=True)

    B = numBands
    numCol = 3 + B + B*(B+1)//2 + B + f_mod.shape[0]
    localData = np.zeros((numLines, numCol))
    fmt = '%i ' + '%.12e ' * (localData.shape[1] - 1)

    crossValidate = params['training_crossValidate']

    if crossValidate:
        chi2sLocal = None
        bandIndicesCV, bandNamesCV, bandColumnsCV,bandVarColumnsCV, redshiftColumnCV = readColumnPositions(params, prefix="training_CV_", refFlux=False)

    if sensitivity and hyperParam_name != "ellSigmaPrior":
        margLike_list=[]
        abscissa_list=[]
        for hyperParamVal in hyperParam_list:
            loc = - 1
            trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,prefix="training_", getXY=True,CV=crossValidate)
            if hyperParam_name == "V_C":
                print("Creating GP for {} = {}".format(hyperParam_name, hyperParamVal))
                gp = PhotozGP(f_mod,
                          bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                          params['lines_pos'], params['lines_width'],
                          hyperParamVal, params['V_L'],
                          params['alpha_C'], params['alpha_L'],
                          redshiftGridGP, use_interpolators=True)
            elif hyperParam_name == "V_L":
                print("Creating GP for {} = {}".format(hyperParam_name, hyperParamVal))
                gp = PhotozGP(f_mod,
                          bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                          params['lines_pos'], params['lines_width'],
                          params['V_C'], hyperParamVal,
                          params['alpha_C'], params['alpha_L'],
                          redshiftGridGP, use_interpolators=True)
            elif hyperParam_name == "alpha_C":
                print("Creating GP for {} = {}".format(hyperParam_name, hyperParamVal))
                gp = PhotozGP(f_mod,
                          bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                          params['lines_pos'], params['lines_width'],
                          params['V_C'], params['V_L'],
                          hyperParamVal, params['alpha_L'],
                          redshiftGridGP, use_interpolators=True)
            elif hyperParam_name == "alpha_L":
                print("Creating GP for {} = {}".format(hyperParam_name, hyperParamVal))
                gp = PhotozGP(f_mod,
                          bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                          params['lines_pos'], params['lines_width'],
                          params['V_C'], params['V_L'],
                          params['alpha_C'], hyperParamVal,
                          redshiftGridGP, use_interpolators=True)
                
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
                ### CALCUL MARGINAL LIKELIHODD OU AUTRE METRIQUE ###
                margLike_list.append(gp.margLike())
                abscissa_list.append(hyperParamVal)

        ### PLOT METRIQUE ###
        ## Plot for this iteration on ellPriorSigma:
        alpha = 0.9
        s = 5
        fig, ax = plt.subplots(constrained_layout=True)
        #print(abscissa_list)
        #print(len(margLike_list), len(abscissa_list))
        vs = ax.hist2d(abscissa_list, margLike_list, bins=[100, 20],\
                       density=True, cmap="Reds", alpha=alpha,\
                       range=[[np.min(abscissa_list), np.max(abscissa_list)], [-10, 200]])
        ax.set_xlabel(hyperParam_name)
        ax.set_ylabel('GP Marginal likelihood')
        ax.set_title('Effect of hyperparameter '+hyperParam_name+' on GP marginal likelihood during training process.')
        fig.show()
        
    else:
        loc = - 1
        trainingDataIter1 = getDataFromFile(params, firstLine, lastLine,prefix="training_", getXY=True,CV=crossValidate)
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

    if threadNum == 0:
        reducedData = np.zeros((numObjectsTraining, numCol))

    if crossValidate:
        chi2sGlobal = np.zeros_like(chi2sLocal)
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

    delightLearn_HyperParam(sys.argv[1])

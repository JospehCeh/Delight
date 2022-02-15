
import sys
#from mpi4py import MPI
import numpy as np
from delight.io import *
from delight.utils import *
from photoz_gp import PhotozGP
from delight.photoz_kernels import Photoz_mean_function, Photoz_kernel
from delight.utils_cy import approx_flux_likelihood_cy
from time import time
from astropy.visualization import hist as astrohist
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import coloredlogs
import logging


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger,fmt='%(asctime)s,%(msecs)03d %(programname)s, %(name)s[%(process)d] %(levelname)s %(message)s')



def delightApply_paramSpecPlot(configfilename, V_C=-1.0, V_L=-1.0, alpha_C=-1.0, alpha_L=-1.0, sensitivity=False, ellPriorSigma_list=[]):
    """

    :param configfilename:
    :return:
    """

    threadNum = 0
    numThreads = 1

    params = parseParamFile(configfilename, verbose=False) 
       
    # Default values read from paramFile
    if V_C < 0:
        V_C = params['V_C']
    if V_L  < 0:
        V_L = params['V_L']
    if alpha_C  < 0:
        alpha_C = params['alpha_C']
    if alpha_L  < 0:
        alpha_L = params['alpha_L']

    if threadNum == 0:
        logger.info("--- DELIGHT-APPLY ---")

    # Read filter coefficients, compute normalization of filters
    bandCoefAmplitudes, bandCoefPositions, bandCoefWidths, norms = readBandCoefficients(params)
    numBands = bandCoefAmplitudes.shape[0]
    band_name=["DC2LSST_u","DC2LSST_g","DC2LSST_r","DC2LSST_i","DC2LSST_z","DC2LSST_y"]

    redshiftDistGrid, redshiftGrid, redshiftGridGP = createGrids(params)
    f_mod_interp = readSEDs(params)
    nt = f_mod_interp.shape[0]
    nz = redshiftGrid.size

    dir_seds = params['templates_directory']
    dir_filters = params['bands_directory']
    lambdaRef = params['lambdaRef']
    sed_names = params['templates_names']
    f_mod_grid = np.zeros((redshiftGrid.size, len(sed_names),len(params['bandNames'])))

    for t, sed_name in enumerate(sed_names):
        f_mod_grid[:, t, :] = np.loadtxt(dir_seds + '/' + sed_name +'_fluxredshiftmod.txt')

    numZbins = redshiftDistGrid.size - 1
    numZ = redshiftGrid.size

    numObjectsTraining = np.sum(1 for line in open(params['training_catFile']))
    numObjectsTarget = np.sum(1 for line in open(params['target_catFile']))
    redshiftsInTarget = ('redshift' in params['target_bandOrder'])
    Ncompress = params['Ncompress']

    firstLine = int(threadNum * numObjectsTarget / float(numThreads))
    lastLine = int(min(numObjectsTarget,(threadNum + 1) * numObjectsTarget / float(numThreads)))
    numLines = lastLine - firstLine

    if threadNum == 0:
        msg= 'Number of Training Objects ' +  str(numObjectsTraining)
        logger.info(msg)

        msg='Number of Target Objects ' + str(numObjectsTarget)
        logger.info(msg)

    msg= 'Thread '+ str(threadNum) + ' , analyzes lines ' +  str(firstLine) + ' to ' + str( lastLine)
    logger.info(msg)

    DL = approx_DL()

    # Create local files to store results
    numMetrics = 7 + len(params['confidenceLevels'])
    localPDFs = np.zeros((numLines, numZ))
    localMetrics = np.zeros((numLines, numMetrics))
    localCompressIndices = np.zeros((numLines,  Ncompress), dtype=int)
    localCompEvidences = np.zeros((numLines,  Ncompress))

    # Looping over chunks of the training set to prepare model predictions over
    numChunks = params['training_numChunks']
    
    for chunk in range(numChunks):
        TR_firstLine = int(chunk * numObjectsTraining / float(numChunks))
        if numObjectsTarget >= numObjectsTraining:
            TR_lastLine = int(min(numObjectsTraining, (chunk + 1) * numObjectsTarget / float(numChunks)))
        else:
            TR_lastLine = int(numObjectsTraining)
        targetIndices = np.arange(TR_firstLine, TR_lastLine)
        numTObjCk = TR_lastLine - TR_firstLine
        redshifts = np.zeros((numTObjCk, ))
        model_mean = np.zeros((numZ, numTObjCk, numBands))
        model_covar = np.zeros((numZ, numTObjCk, numBands))
        bestTypes = np.zeros((numTObjCk, ), dtype=int)
        ells = np.zeros((numTObjCk, ), dtype=int)

        # loop on training data and training GP coefficients produced by delight_learn
        # It fills the model_mean and model_covar predicted by GP
        loc = TR_firstLine - 1
        trainingDataIter = getDataFromFile(params, TR_firstLine, TR_lastLine,prefix="training_", ftype="gpparams")
        
        print("Creation of GP with V_C = {}, V_L = {}, alpha_C = {}, alpha_L = {}.".format(V_C, V_L, alpha_C, alpha_L))

        gp = PhotozGP(f_mod_interp,
                  bandCoefAmplitudes, bandCoefPositions, bandCoefWidths,
                  params['lines_pos'], params['lines_width'],
                  V_C, V_L,
                  alpha_C, alpha_L,
                  redshiftGridGP, use_interpolators=True)

        # loop on training data to load the GP parameter
        for loc, (z, ell, bands, X, B, flatarray) in enumerate(trainingDataIter):
            t1 = time()
            redshifts[loc] = z              # redshift of all training samples
            gp.setCore(X, B, nt,flatarray[0:nt+B+B*(B+1)//2])
            bestTypes[loc] = gp.bestType   # retrieve the best-type found by delight-learn
            ells[loc] = ell                # retrieve the luminosity parameter l

            # here is the model prediction of Gaussian Process for that particular trainning galaxy
            model_mean[:, loc, :], model_covar[:, loc, :] = gp.predictAndInterpolate(redshiftGrid, ell=ell)
            t2 = time()
            # print(loc, t2-t1)
            quot = (TR_lastLine - TR_firstLine)//10
            if loc % quot == 0:
                figMeanCov, axMeanCov = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
                #print(bands)
                #print(model_mean[:, loc, 0].shape)
                for band in bands:
                    #print(band_name[int(band)])
                    axMeanCov.errorbar(redshiftGrid, model_mean[:, loc, int(band)], yerr=model_covar[:, loc, int(band)], label=band_name[int(band)], fmt='o', markersize=3, capsize=0)
                    #axMeanCov[1].plot(redshiftGrid, model_covar[:, loc, int(band)], label=band_name[int(band)])
                axMeanCov.set_xlabel("redshiftGrid")
                axMeanCov.set_ylabel("Mean flux - training No {}]".format(loc))
                axMeanCov.set_yscale('log')
                axMeanCov.set_title("Predicted mean flux")
                #axMeanCov[1].set_xlabel("redshiftGrid")
                #axMeanCov[1].set_ylabel("model_covar[:, {}, band]".format(loc))
                #axMeanCov[1].set_yscale('log')
                #axMeanCov[1].set_title("Model covar")
                axMeanCov.legend(loc='upper center')
                figMeanCov.suptitle("GP prediction and interpolation results for training galaxy No. {}, z = {}".format(loc, z))
        

        #Redshift prior on training galaxy
        # p_t = params['p_t'][bestTypes][None, :]
        # p_z_t = params['p_z_t'][bestTypes][None, :]
        # compute the prior for taht training sample
        prior = np.exp(-0.5*((redshiftGrid[:, None]-redshifts[None, :]) /params['zPriorSigma'])**2)
        #print(prior.shape)
        figPrior, axPrior = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
        for TRind in range(prior.shape[1]):
            quot = (prior.shape[1])//5
            if TRind % quot == 0:
                axPrior.plot(redshiftGrid, prior[:, TRind], label="z-training = {}".format(redshifts[TRind]))
        axPrior.set_xlabel("redshiftGrid")
        axPrior.set_ylabel("prior")
        axPrior.set_yscale('log')
        figPrior.suptitle("Prior in delightApply, zPriorSigma = {}".format(params['zPriorSigma']))
        figPrior.legend(loc='upper right')
                
        
        # prior[prior < 1e-6] = 0
        # prior *= p_t * redshiftGrid[:, None] *
        # np.exp(-0.5 * redshiftGrid[:, None]**2 / p_z_t) / p_z_t

        if params['useCompression'] and params['compressionFilesFound']:
            fC = open(params['compressMargLikFile'])
            fCI = open(params['compressIndicesFile'])
            itCompM = itertools.islice(fC, firstLine, lastLine)
            iterCompI = itertools.islice(fCI, firstLine, lastLine)

        if sensitivity and len(ellPriorSigma_list) > 0:
            print("Study of the influence of ellPriorSigma on likelihood and evidences")
            # ~ nbCol = 2
            # ~ nbLin = (len(ellPriorSigma_list)+1) // 2
            # ~ fig, axs = plt.subplots(nbLin, nbCol, figsize=(nbCol*6, nbLin*5), constrained_layout=True)
            # ~ ligne, colonne = 0, 0
            fig2, axs2 = plt.subplots(1, 2, figsize=(12,10), constrained_layout=True)
            
            for ellPriorSigma in ellPriorSigma_list:
                allEv = []
                allTrainingZx = []
                allTargetZy = []
                targetZ = []
                zMaxEv_list=[]
                zAvg_list=[]
                errZmaxEv_list = []
                errZavg_list = []
                figEv, axEv = plt.subplots(2, 2, figsize=(12,20), constrained_layout=True)
                ligne, colonne = 0,0
                print("Computation of likelihood and evidences for ellPriorSigma = {}".format(ellPriorSigma))
                
                targetDataIter = getDataFromFile(params, firstLine, lastLine,prefix="target_", getXY=False, CV=False)
                # ~ loc, (z, normedRefFlux, bands, fluxes, fluxesVar, bCV, dCV, dVCV) = list(enumerate(targetDataIter))[0]
                for loc, (z, normedRefFlux, bands, fluxes, fluxesVar, bCV, dCV, dVCV) in enumerate(targetDataIter):
                    t1 = time()
                    ell_hat_z = normedRefFlux * 4 * np.pi * params['fluxLuminosityNorm'] * (DL(redshiftGrid)**2. * (1+redshiftGrid))
                    ell_hat_z[:] = 1
                    if params['useCompression'] and params['compressionFilesFound']:
                        indices = np.array(next(iterCompI).split(' '), dtype=int)
                        sel = np.in1d(targetIndices, indices, assume_unique=True)
                        # same likelihood as for template fitting
                        like_grid = approx_flux_likelihood(fluxes,fluxesVar,model_mean[:, sel, :][:, :, bands],
                        f_mod_covar=model_covar[:, sel, :][:, :, bands],
                        marginalizeEll=True, normalized=False,
                        ell_hat=ell_hat_z,
                        ell_var=(ell_hat_z*ellPriorSigma)**2)
                        like_grid *= prior[:, sel]
                    else:
                        like_grid = np.zeros((nz, model_mean.shape[1]))
                        # same likelihood as for template fitting, but cython
                        approx_flux_likelihood_cy(
                            like_grid, nz, model_mean.shape[1], bands.size,
                            fluxes, fluxesVar,  # target galaxy fluxes and variance
                            model_mean[:, :, bands],     # prediction with Gaussian process
                            model_covar[:, :, bands],
                            ell_hat=ell_hat_z,           # it will find internally the ell
                            ell_var=(ell_hat_z*ellPriorSigma)**2)
                        like_grid *= prior[:, :] #likelihood multiplied by redshift training galaxies priors
                    t2 = time()
                    localPDFs[loc, :] += like_grid.sum(axis=1)  # the final redshift posterior is sum over training galaxies posteriors

                    # compute the evidence for each model
                    targetZ.append(z)
                    evidences = np.trapz(like_grid, x=redshiftGrid, axis=0)
                    zMaxEv = redshifts[np.argmax(evidences)]
                    zMaxEv_list.append(zMaxEv)
                    zAvg = np.average(redshifts, weights=evidences)
                    zAvg_list.append(zAvg)
                    errZ = (zMaxEv - z) / z
                    errZmaxEv_list.append(errZ)
                    errZavg = (zAvg - z) / z
                    errZavg_list.append(errZavg)
                    for ev in evidences:
                        allEv.append(ev)
                        allTargetZy.append(z)

                    t3 = time()

                    if params['useCompression'] and not params['compressionFilesFound']:
                        if localCompressIndices[loc, :].sum() == 0:
                            sortind = np.argsort(evidences)[::-1][0:Ncompress]
                            localCompressIndices[loc, :] = targetIndices[sortind]
                            localCompEvidences[loc, :] = evidences[sortind]
                        else:
                            dind = np.concatenate((targetIndices,localCompressIndices[loc, :]))
                            devi = np.concatenate((evidences,localCompEvidences[loc, :]))
                            sortind = np.argsort(devi)[::-1][0:Ncompress]
                            localCompressIndices[loc, :] = dind[sortind]
                            localCompEvidences[loc, :] = devi[sortind]

                    if chunk == numChunks - 1\
                            and redshiftsInTarget\
                         and localPDFs[loc, :].sum() > 0:
                        localMetrics[loc, :] = computeMetrics(z, redshiftGrid,localPDFs[loc, :],params['confidenceLevels'])
                    t4 = time()
                    
                    if loc % 100 == 0:
                        print(loc, t2-t1, t3-t2, t4-t3)
                        print('Likelihoods shape : {}'.format(like_grid.shape))
                        print('Evidences shape : {}'.format(evidences.shape))
                        print('PDFs shape : {}'.format(localPDFs.shape))
                        
                    quot = (lastLine - firstLine) // 4
                    if loc % quot == 0:
                        alpha = 0.9
                        s = 5
                        evp=axEv[ligne, colonne].scatter(redshifts, evidences, label="Evidence", alpha=alpha, s=s)
                        axEv[ligne, colonne].axvline(z, label="Target z = {}".format(z), c=evp.get_facecolor()[-1], lw=1)
                        axEv[ligne, colonne].axvline(zMaxEv, c=evp.get_facecolor()[-1], lw=1, ls=':', label="Max. evid. z = {}".format(zMaxEv))
                        axEv[ligne, colonne].axvline(zAvg, c=evp.get_facecolor()[-1], lw=1, ls='-.', label="Avg z = {}".format(zAvg))
                        axEv[ligne, colonne].set_xlabel("Training z")
                        axEv[ligne, colonne].set_ylabel("Evidence")
                        axEv[ligne, colonne].legend()
                        # ~ axEv.set_yscale('log')
                        if colonne == 1:
                            ligne+=1
                            colonne=0
                        else:
                            colonne+=1
                figEv.suptitle("Evidence for training z for a given target z ; ellPriorSigma = {}".format(ellPriorSigma))
                #figEv.legend(loc='upper right')
                        
                        
                        
                        #print("fluxes = {},\nflux variances = {}".format(fluxes, fluxesVar))

                ### PLOT LIKE_GRID and/or EVIDENCES, for a SINGLE GALAXY MAYBE? ###
                ## Plot for this iteration on ellPriorSigma:
                # ~ plotInd = -1
                # ~ #for ligne in np.arange(4):
                    # ~ #for colonne in np.arange(2):
                # ~ plotInd += 1
                # ~ #print((like_grid[:, plotInd]).shape , len(redshiftGrid))
                zLims=[np.min(targetZ), np.max(targetZ)]
                axs2[0].plot(zLims, zLims, c='k', lw=1, zorder=0, alpha=1)
                zpt=axs2[0].plot(targetZ, zMaxEv_list, label='Z max. ev., ellPriorSigma = {}'.format(ellPriorSigma) )
                axs2[0].plot(targetZ, zAvg_list, ls=':', c=zpt[-1].get_color(), label='Z avg')
                axs2[0].set_xlabel('Target Z')
                axs2[0].set_ylabel('Training Z of maximum evidence')
                axs2[0].set_yscale('linear')
                axs2[0].set_title('Maximum evidence redshift')
                erzpt=axs2[1].plot(targetZ, errZmaxEv_list)
                axs2[1].plot(targetZ, errZavg_list, ls=':', c=erzpt[-1].get_color())
                axs2[1].set_xlabel('Target Z')
                axs2[1].set_ylabel('Error on Z')
                axs2[1].set_yscale('log')
                axs2[1].set_title('Error vs. target redshift')
                #axs2.legend(loc="center right")

                #print("Local PDF shape : {}".format(localPDFs.shape))
                # ~ alpha = 0.9
                # ~ s = 5
                # ~ cmap = "coolwarm_r"

                # ~ indEv = 0
                # ~ for ev in allEv:
                    # ~ allTrainingZx.append(redshifts[indEv%len(redshifts)])
                    # ~ indEv+=1
                
                # ~ allOrd = []
                # ~ allAbs = []
                # ~ allPdf = []
                # ~ targetInd = -1
                # ~ print(len(targetZ), len(redshiftGrid))
                # ~ for tgZ in targetZ:
                    # ~ targetInd+=1
                    # ~ specInd = -1
                    # ~ for spZ in redshiftGrid:
                        # ~ specInd+=1
                        # ~ allOrd.append(spZ)
                        # ~ allAbs.append(tgZ)
                        # ~ allPdf.append(localPDFs[targetInd, specInd])
                
                # ~ print("specInd = {}, targetInd = {}".format(specInd, targetInd))
                
                # ~ if nbLin > 1:
                    # ~ #axs[ligne, colonne].set_xlabel('Training z')
                    # ~ #axs[ligne, colonne].set_ylabel('evidence')
                    # ~ #axs[ligne, colonne].hist2d(allTrainingZx, allEv, bins=[100, 100],\
                    # ~ #                           density=True, cmap="Reds", alpha=alpha)
                    
                    # ~ #axs[ligne, colonne].set_yscale('log')
                    # ~ #axs[ligne, colonne].set_title('Evidences : likelihood integrated over spec-z')
                    # ~ #axs[ligne, colonne].set_title('ellPriorSigma = {}'.format(ellPriorSigma))
                    # ~ #axs[ligne, colonne].legend(loc="upper right")
                    # ~ #astrohist(allEv, ax=axs[ligne, colonne], bins='blocks')
                    
                    # ~ vs=axs[ligne, colonne].scatter(allAbs, allOrd, c=allPdf, cmap=cmap, label='ellPriorSigma = {}'.format(ellPriorSigma), alpha=alpha, s=s, norm=LogNorm())
                    # ~ axs[ligne, colonne].set_xlabel('Target z')
                    # ~ axs[ligne, colonne].set_ylabel('Spec z')
                    # ~ clb = plt.colorbar(vs, ax=axs[ligne,colonne])
                    # ~ clb.set_label('PDF')

                    # ~ axs[ligne, colonne].set_title('PDFs : probability of spec-z given target-z')
                    # ~ axs[ligne, colonne].set_title('ellPriorSigma = {}'.format(ellPriorSigma))
                    # ~ axs[ligne, colonne].legend(loc="upper right")
                # ~ else:
                    # ~ #axs[colonne].set_xlabel('Training z')
                    # ~ #axs[colonne].set_ylabel('evidence')
                    # ~ #axs[colonne].hist2d(allTrainingZx, allEv, bins=[100, 100],\
                    # ~ #                           density=True, cmap="Reds", alpha=alpha)
                    
                    # ~ #axs[colonne].set_yscale('log')
                    # ~ #axs[colonne].set_title('Evidences : likelihood integrated over spec-z')
                    # ~ #axs[colonne].set_title('ellPriorSigma = {}'.format(ellPriorSigma))
                    # ~ #axs[colonne].legend(loc="upper right")
                    # ~ #astrohist(allEv, ax=axs[colonne], bins='blocks')
                    
                    # ~ vs=axs[colonne].scatter(allAbs, allOrd, c=allPdf, cmap=cmap, label='ellPriorSigma = {}'.format(ellPriorSigma), alpha=alpha, s=s, norm=LogNorm())
                    # ~ axs[colonne].set_xlabel('Target z')
                    # ~ axs[colonne].set_ylabel('Spec z')
                    # ~ clb = plt.colorbar(vs, ax=axs[colonne])
                    # ~ clb.set_label('PDF')

                    # ~ axs[colonne].set_title('PDFs : probability of spec-z given target-z')
                    # ~ axs[colonne].set_title('ellPriorSigma = {}'.format(ellPriorSigma))
                    # ~ axs[colonne].legend(loc="upper right")
                
                # ~ if colonne < 1:
                    # ~ colonne+=1
                # ~ else:
                    # ~ ligne+=1
                    # ~ colonne=0
                ### WHAT ARE THE DIMENSIONS OF THE OBJECTS (8 SEDs in template fitting, how many here?) ###

            if params['useCompression'] and params['compressionFilesFound']:
                fC.close()
                fCI.close()
            #fig.legend()
            #fig.suptitle('V_C = {}, V_L = {}, alpha_C = {}, alpha_L = {}'.format(V_C, V_L, alpha_C, alpha_L))
            #fig.show()
            fig2.legend()
            #fig2.show()
        else:
            targetDataIter = getDataFromFile(params, firstLine, lastLine,prefix="target_", getXY=False, CV=False)
            # loop on target samples
            for loc, (z, normedRefFlux, bands, fluxes, fluxesVar, bCV, dCV, dVCV) in enumerate(targetDataIter):
                t1 = time()
                ell_hat_z = normedRefFlux * 4 * np.pi * params['fluxLuminosityNorm'] * (DL(redshiftGrid)**2. * (1+redshiftGrid))
                ell_hat_z[:] = 1
                if params['useCompression'] and params['compressionFilesFound']:
                    indices = np.array(next(iterCompI).split(' '), dtype=int)
                    sel = np.in1d(targetIndices, indices, assume_unique=True)
                    # same likelihood as for template fitting
                    like_grid2 = approx_flux_likelihood(fluxes,fluxesVar,model_mean[:, sel, :][:, :, bands],
                    f_mod_covar=model_covar[:, sel, :][:, :, bands],
                    marginalizeEll=True, normalized=False,
                    ell_hat=ell_hat_z,
                    ell_var=(ell_hat_z*params['ellPriorSigma'])**2)
                    like_grid *= prior[:, sel]
                else:
                    like_grid = np.zeros((nz, model_mean.shape[1]))
                    # same likelihood as for template fitting, but cython
                    approx_flux_likelihood_cy(
                        like_grid, nz, model_mean.shape[1], bands.size,
                        fluxes, fluxesVar,  # target galaxy fluxes and variance
                        model_mean[:, :, bands],     # prediction with Gaussian process
                        model_covar[:, :, bands],
                        ell_hat=ell_hat_z,           # it will find internally the ell
                        ell_var=(ell_hat_z*params['ellPriorSigma'])**2)
                    like_grid *= prior[:, :] #likelihood multiplied by redshift training galaxies priors
                t2 = time()
                localPDFs[loc, :] += like_grid.sum(axis=1)  # the final redshift posterior is sum over training galaxies posteriors

                # compute the evidence for each model
                evidences = np.trapz(like_grid, x=redshiftGrid, axis=0)
                t3 = time()

                if params['useCompression'] and not params['compressionFilesFound']:
                    if localCompressIndices[loc, :].sum() == 0:
                        sortind = np.argsort(evidences)[::-1][0:Ncompress]
                        localCompressIndices[loc, :] = targetIndices[sortind]
                        localCompEvidences[loc, :] = evidences[sortind]
                    else:
                        dind = np.concatenate((targetIndices,localCompressIndices[loc, :]))
                        devi = np.concatenate((evidences,localCompEvidences[loc, :]))
                        sortind = np.argsort(devi)[::-1][0:Ncompress]
                        localCompressIndices[loc, :] = dind[sortind]
                        localCompEvidences[loc, :] = devi[sortind]

                if chunk == numChunks - 1\
                        and redshiftsInTarget\
                     and localPDFs[loc, :].sum() > 0:
                    localMetrics[loc, :] = computeMetrics(z, redshiftGrid,localPDFs[loc, :],params['confidenceLevels'])
                t4 = time()
                if loc % 100 == 0:
                    print(loc, t2-t1, t3-t2, t4-t3)

            if params['useCompression'] and params['compressionFilesFound']:
                fC.close()
                fCI.close()

    if threadNum == 0:
        globalPDFs = np.zeros((numObjectsTarget, numZ))
        globalCompressIndices = np.zeros((numObjectsTarget, Ncompress), dtype=int)
        globalCompEvidences = np.zeros((numObjectsTarget, Ncompress))
        globalMetrics = np.zeros((numObjectsTarget, numMetrics))
    else:
        globalPDFs = None
        globalCompressIndices = None
        globalCompEvidences = None
        globalMetrics = None

    firstLines = [int(k*numObjectsTarget/numThreads) for k in range(numThreads)]
    lastLines = [int(min(numObjectsTarget, (k+1)*numObjectsTarget/numThreads)) for k in range(numThreads)]
    numLines = [lastLines[k] - firstLines[k] for k in range(numThreads)]

    sendcounts = tuple([numLines[k] * numZ for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numZ for k in range(numThreads)])
    globalPDFs = localPDFs

    sendcounts = tuple([numLines[k] * Ncompress for k in range(numThreads)])
    displacements = tuple([firstLines[k] * Ncompress for k in range(numThreads)])
    globalCompressIndices = localCompressIndices
    globalCompEvidences = localCompEvidences

    sendcounts = tuple([numLines[k] * numMetrics for k in range(numThreads)])
    displacements = tuple([firstLines[k] * numMetrics for k in range(numThreads)])
    globalMetrics = localMetrics

    if threadNum == 0:
        fmt = '%.2e'
        fname = params['redshiftpdfFileComp'] if params['compressionFilesFound']\
            else params['redshiftpdfFile']
        np.savetxt(fname, globalPDFs, fmt=fmt)
        if redshiftsInTarget:
            np.savetxt(params['metricsFile'], globalMetrics, fmt=fmt)
        if params['useCompression'] and not params['compressionFilesFound']:
            np.savetxt(params['compressMargLikFile'],globalCompEvidences, fmt=fmt)
            np.savetxt(params['compressIndicesFile'],globalCompressIndices, fmt="%i")


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start Delight Learn.py"
    logger.info(msg)
    logger.info("--- Process Delight Learn ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    delightApply_paramSpecPlot(sys.argv[1])

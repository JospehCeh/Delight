#!/usr/bin/env python
# coding: utf-8

# # Test Delight on DESC-DC2 simulation  in the context of  Vera C. Rubin Obs (LSST) 
#
#
# - author : Sylvie Dagoret-Campagne
# - affiliation : IJCLab/IN2P3/CNRS
# - creation date : January 22 2022
#
#
#
# - run at NERSC with **desc-python** python kernel.
#
#
# Instruction to have a **desc-python** environnement:
# https://confluence.slac.stanfo.edu/display/LSSTDESC/Getting+Started+with+Anaconda+Python+at+NERSC
#
#
# This environnement is a clone from the **desc-python** environnement where package required in requirements can be addded according the instructions here
# https://github.com/LSSTDESC/desc-python/wiki/Add-Packages-to-the-desc-python-environment

# We will use the parameter file "tmps/parametersTestRail.cfg".
# This contains a description of the bands and data to be used.
# In this example we will generate mock data for the ugrizy LSST bands,
# fit each object with our GP using ugi bands only and see how it predicts the rz bands.
# This is an example for filling in/predicting missing bands in a fully bayesian way
# with a flexible SED model quickly via our photo-z GP.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys,os
sys.path.append('../')
from delight.io import *
from delight.utils import *
from delight.photoz_gp import PhotozGP

from delight.interfaces.rail.processSEDs import processSEDs
from delight.interfaces.rail.processFilters import processFilters
from delight.interfaces.rail.templateFitting import templateFitting
from delight.interfaces.rail.delightLearn import delightLearn
from delight.interfaces.rail.delightApply import delightApply

### Fonction for external use             ###
### May be improved for sensitivity runs? ###
def run_full_delight_confFile(configFullFilename):
    processFilters(configFullFilename)
    processSEDs(configFullFilename)
    templateFitting(configFullFilename)
    delightLearn(configFullFilename)
    delightApply(configFullFilename)


#-----------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # execute only if run as a script


    msg="Start run_full_delight_conFile.py"
    logger.info(msg)
    logger.info("--- Process Full Delight ---")


    if len(sys.argv) < 2:
        raise Exception('Please provide a parameter file')

    run_full_delight_confFile(sys.argv[1])
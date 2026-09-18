import sctop as top
import pandas as pd
import numpy as np
import scipy
import plotly.io as pio
import anndata as ad
import h5py
import sys
sys.path.append('/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP')
import TopObject


def runTrial(name, seed=0):
    topObject = TopObject.TopObject(name, skipProcess=True, keep=True)
    proportionTestMap = topObject.getBestGenes(seed=seed, trialCount=1, proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
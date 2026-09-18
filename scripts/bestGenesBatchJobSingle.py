#!/usr/bin/env python
import sctop as top
import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import plotly.io as pio
import anndata as ad
import h5py
import sys
sys.path.append('/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Vilker_Helper_Files/scTOP')
import TopObject
import json

PREFIX = "/restricted/projectnb/crem-trainees/Kotton_Lab/Eitan/Results/BestGenesResults/"
def runTrial(name, seed):
    seed += 4
    keep = ["Basal", "AT1", "AT2", "Secretory", "Goblet", "Ciliated"]
    # topObject = TopObject.TopObject(name, skipProcess=True)
    topObject = TopObject.TopObject(name, skipProcess=True, manualInit=True)
    topObject.anndata = sc.read_h5ad(topObject.filePath)
    topObject.anndata.var.set_index("_index", inplace=True)
    topObject.setAnndata(topObject.anndata[topObject.anndata.obs[topObject.cellTypeColumn].isin(keep)])
    proportionTestMap = topObject.getBestGenes(seed=seed, trialCount=1, batchJob=True)
    with open(PREFIX + "output" + str(seed) +".json", "w") as file:
        json.dump(proportionTestMap, file)

if len(sys.argv) > 1:
    seed = int(sys.argv[1])
    name = sys.argv[2]
    runTrial(name, seed)
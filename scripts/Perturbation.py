import numpy as np
import pandas as pd
import sctop as top
import anndata as ad
import pickle
import matplotlib.pyplot as plt

# def processed_from_adata_like_basis(adata, cell_ids) -> pd.DataFrame:
#     """
#     Replicates the exact pattern you used for basis construction:
#       cell_data_sparse = adata[cell_ids, :].X.T
#       df = DataFrame(..., index=adata.var_names, columns=cell_ids)
#       process(df, average=False)
#     Returns genes x cells processed matrix.
#     """
#     if isinstance(cell_ids, (pd.Index, pd.Series)):
#         cell_ids = cell_ids.astype(str).tolist()
#     else:
#         cell_ids = list(cell_ids)
#     if len(cell_ids) == 0:
#         raise ValueError("cell_ids is empty.")

#     ad = adata[cell_ids, :]
#     X = ad.X
#     if hasattr(X, "toarray"):
#         X = X.toarray()
#     X = np.asarray(X, dtype=np.float32)           # cells x genes
#     df = pd.DataFrame(X.T, index=ad.var_names, columns=ad.obs_names)  # genes x cells
#     return top.process(df, average=False)


def scoreAndPredict(basis: pd.DataFrame, proc_sample_gxc: pd.DataFrame):
    proj = top.score(basis, proc_sample_gxc, full_output=False)  # celltypes x cells
    pred = proj.idxmax(axis=0)

    vals = proj.to_numpy(np.float32)
    part = np.partition(vals, kth=vals.shape[0] - 2, axis=0)
    margin = pd.Series(part[-1, :] - part[-2, :], index=proj.columns, dtype=float)
    return pred, margin, proj


def slerp_columns(U: np.ndarray, v: np.ndarray, alpha: float, eps: float = 1e-12) -> np.ndarray:
    """
    Slerp each column of U (GxN) toward v (G,).
    Assumes vectors are nonzero; normalizes defensively.
    """
    # Normalize inputs
    U = U / (np.linalg.norm(U, axis=0, keepdims=True) + eps)
    v = v / (np.linalg.norm(v) + eps)

    dots = np.clip((U * v[:, None]).sum(axis=0), -1.0, 1.0)
    omegas = np.arccos(dots)                      # (N,)
    so = np.sin(omegas) + eps

    near = omegas < 1e-6
    W = np.empty_like(U, dtype=np.float32)

    if np.any(~near):
        omega = omegas[~near]
        s0 = np.sin((1.0 - alpha) * omega) / so[~near]
        s1 = np.sin(alpha * omega) / so[~near]
        W[:, ~near] = (U[:, ~near] * s0[None, :]) + (v[:, None] * s1[None, :])

    if np.any(near):
        W[:, near] = (1.0 - alpha) * U[:, near] + alpha * v[:, None]
    
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + eps)
    return W.astype(np.float32)


def perturbTowardBasis(
    proc_source_gxc: pd.DataFrame | np.ndarray,   # genes x cells (processed)
    basis: pd.DataFrame,             # genes x celltypes (processed prototypes)
    targetLabel: str,
    alphas: np.ndarray,
    cols: np.ndarray | None = None
) -> list[pd.DataFrame]:
    """
    For each alpha, returns processed genes x cells matrix:
      z_alpha = slerp(z, b_target, alpha)
    IMPORTANT: we align genes by intersection and keep ordering consistent.
    """
    if targetLabel not in basis.columns:
        raise ValueError(f"targetLabel='{targetLabel}' not found in basis.columns")

    if type(proc_source_gxc) == pd.DataFrame:
        common = np.intersect1d(proc_source_gxc.index.values, basis.index.values)
        if common.size == 0:
            raise ValueError("No common genes between processed sample and basis.")
    
        Z = proc_source_gxc.loc[common].to_numpy(np.float32, copy=False)   # G x N
        b = basis.loc[common, targetLabel].to_numpy(np.float32, copy=False)  # (G,)
        cols = proc_source_gxc.columns
    else:
        common = basis.index.values
        b = basis.loc[:, targetLabel].to_numpy(np.float32, copy=False)  # (G,)

    out = []
    for a in np.nditer(alphas):
        Za = slerp_columns((proc_source_gxc if type(proc_source_gxc) == np.ndarray else Z), b, float(a))
        out.append(pd.DataFrame(Za, index=common, columns=cols))
    return out


def runProcessedSpaceExperiment(
    topObject,
    basis,
    sourceLabel,
    targetLabel,
    sourceLabelOverride: str | None = None,
    alphas: np.ndarray | None = None,
    maxCells: int | None = None,
    randomState: int = 0,
    filterToCorrectAtAlpha0: bool = True,
    verbose: bool = True,
    stopAt50: bool = False,
    precision: int = 0.01,
    initialData: tuple | None = None,
    diffCount: int | None = None,
    includeCriteria: pd.Series | None = None
):
    """
    Steps:
      1) build proc_source exactly like your manual pipeline
      2) sanity check alpha=0 predictions
      3) optionally keep only cells correctly predicted as source at alpha=0
      4) slerp toward target basis vector for each alpha
      5) report flip curves + per-cell thresholds
    """
    rng = np.random.default_rng(randomState)
    alphas = np.linspace(0, 1.0, 31) if alphas is None else alphas

    if initialData is None:
        validCells = topObject.annotations == sourceLabel if sourceLabelOverride is None else topObject.annotations == sourceLabelOverride
        validCells = validCells if includeCriteria is None else np.logical_and(validCells, includeCriteria) 
        srcIDs = np.array(topObject.df.columns[validCells], dtype=str)
        if srcIDs.size == 0:
            raise ValueError("No source cells provided.")
    
        if maxCells is not None and srcIDs.size > maxCells:
            srcIDs = rng.choice(srcIDs, size=maxCells, replace=False)
    
        processed = topObject.processed if topObject.processed is not None else topObject.process()  #processed_from_adata_like_basis(adata, srcIDs)
        processed = processed.loc[:, srcIDs]
        # processed = top.process(topObject.df.loc[:, srcIDs])
    
        pred0, margin0, proj0 = scoreAndPredict(basis, processed)
    
        if verbose:
            frac0 = float((pred0 == sourceLabel).mean())
            print(f"[Sanity] alpha=0 P(pred=={sourceLabel}) = {frac0:.3f}")
    
        if filterToCorrectAtAlpha0:
            keep = pred0[pred0 == sourceLabel].index
            if keep.size == 0:
                raise ValueError(
                    f"After filtering to alpha=0 correct cells, none remain for sourceLabel={sourceLabel}. "
                    f"Check naming: sourceLabel must match basis.columns exactly."
                )
            processed = processed[keep]
            pred0 = pred0.loc[keep]
            margin0 = margin0.loc[keep]

    else:
        processed, cols = initialData

    # Filter to only most relevant genes
    if diffCount is not None:
        diffs = pd.DataFrame({"diff": basis[targetLabel] - basis[sourceLabel]}, index=basis.index)
        upTarget = diffs.sort_values(by="diff", ascending=False).iloc[:diffCount]["diff"]
        downTarget = diffs.sort_values(by="diff", ascending=True).iloc[:diffCount]["diff"]
        validGenes = list(upTarget) + list(downTarget)
        processed = processed.loc[validGenes, :]
        basis = basis.loc[validGenes, :]

    if stopAt50:
        # theta = processedMeanT.dot(basis[targetLabel])
        theta = np.arccos(processed.T.dot(basis[targetLabel]).mean())
        # theta = 1
        lastAlpha = alphas[0]
        for alpha in alphas:
            proc_df = perturbTowardBasis(processed, basis, targetLabel=targetLabel, cols=cols, alphas=np.array(alpha, dtype=float))[0]
            pred, margin, _ = scoreAndPredict(basis, proc_df)
            if float((pred == sourceLabel).mean()) <= 0.5 and float((pred == targetLabel).mean()) >= 0.5:
                for granularAlpha in np.arange(lastAlpha + precision, alpha - precision / 4, precision):
                    proc_df = perturbTowardBasis(processed, basis, targetLabel=targetLabel, cols=cols, alphas=np.array(granularAlpha, dtype=float))[0]
                    pred, margin, _ = scoreAndPredict(basis, proc_df)
                    if float((pred == sourceLabel).mean()) <= 0.5 and float((pred == targetLabel).mean()) >= 0.5:
                        return granularAlpha * theta
                return alpha * theta
            lastAlpha = alpha
        return None
    else:
        print("Perturbing toward basis...")
        proc_list = perturbTowardBasis(processed, basis, targetLabel=targetLabel, alphas=alphas)

    preds = []
    frac_stay = []
    frac_target = []

    print("Scoring all results...")
    for proc_df in proc_list:
        pred, margin, _ = scoreAndPredict(basis, proc_df)
        preds.append(pred)
        frac_stay.append(float((pred == sourceLabel).mean()))
        frac_target.append(float((pred == targetLabel).mean()))

    frac_stay = np.array(frac_stay, dtype=float)
    frac_target = np.array(frac_target, dtype=float)

    print("Organizing results...")
    cells = preds[0].index
    alpha_leave = pd.Series(np.nan, index=cells, dtype=float)
    alpha_hit = pd.Series(np.nan, index=cells, dtype=float)
    for cell in cells:
        for a, pred in zip(alphas, preds):
            if np.isnan(alpha_leave[cell]) and pred.loc[cell] != sourceLabel:
                alpha_leave[cell] = float(a)
            if np.isnan(alpha_hit[cell]) and pred.loc[cell] == targetLabel:
                alpha_hit[cell] = float(a)
            if not np.isnan(alpha_leave[cell]) and not np.isnan(alpha_hit[cell]):
                break

    idx_leave50 = np.where(frac_stay <= 0.5)[0]
    alpha50_leave = float(alphas[idx_leave50[0]]) if idx_leave50.size else np.nan

    idx_hit50 = np.where(frac_target >= 0.5)[0]
    alpha50_hit = float(alphas[idx_hit50[0]]) if idx_hit50.size else np.nan

    return {
        "alphas": alphas,
        "proc_source_used": processed,         # processed baseline cells used
        "pred_alpha0": pred0,
        "margin_alpha0": margin0,
        "frac_stay_source": frac_stay,
        "frac_become_target": frac_target,
        "preds_per_alpha": preds,
        "alpha_leave_source_per_cell": alpha_leave,
        "alpha_become_target_per_cell": alpha_hit,
        "alpha50_leave_source": alpha50_leave,
        "alpha50_become_target": alpha50_hit,
    }


def plot_flip_curves(result, title="Perturbation flip curves", outFile=None):
    alphas = result["alphas"]
    plt.figure(figsize=(6,4))
    plt.plot(alphas, result["frac_stay_source"], label="P(stay source)")
    plt.plot(alphas, result["frac_become_target"], label="P(become target)")
    plt.xlabel("alpha")
    plt.ylabel("fraction")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outFile is not None:
        plt.savefig(outFile)
    plt.show()


def getSpaceDimensions(
    topObject,
    basis,
    sourceLabel,
    sourceLabelOverride: str | None = None,
    alphas: np.ndarray | None = None,
    maxCells: int | None = None,
    randomState: int = 0,
    filterToCorrectAtAlpha0: bool = True,
    verbose: bool = True,
    includeCriteria: pd.Series | None = None,
    precision: int = 0.01,
    cellTypes: list | None = None,
    diffCount: int | None = None,
):

    alphas = np.linspace(0, 0.8, 17) if alphas is None else alphas
    rng = np.random.default_rng(randomState)

    # Process source label data
    validCells = topObject.annotations == sourceLabel if sourceLabelOverride is None else topObject.annotations == sourceLabelOverride
    validCells = validCells if includeCriteria is None else np.logical_and(validCells, includeCriteria) 
    srcIDs = np.array(topObject.df.columns[validCells], dtype=str)
    if srcIDs.size == 0:
        raise ValueError("No source cells provided.")

    if maxCells is not None and srcIDs.size > maxCells:
        srcIDs = rng.choice(srcIDs, size=maxCells, replace=False)

    processed = topObject.processed if topObject.processed is not None else topObject.process()
    processed = processed.loc[:, srcIDs]
    # processedMeanT = processed.mean(axis=1).T
    cols = processed.columns
    processed = processed.to_numpy(np.float32, copy=False)   # G x N

    # Find 50% perturbation point for each cell type in the basis
    dimensionsMap = {}
    for targetLabel in basis.columns if cellTypes is None else cellTypes:
        if targetLabel != sourceLabel:
            print("Testing " + sourceLabel + " vs " + targetLabel)
            dimensionsMap[targetLabel] = runProcessedSpaceExperiment(topObject, basis, sourceLabel, targetLabel, alphas=alphas, verbose=False, 
                                                                     stopAt50=True, precision=precision, initialData=(processed, cols))
        else:
            dimensionsMap[targetLabel] = 0
    return dimensionsMap


# Find 50% perturbation points for every pair of cell types in a dataset
def getSpaceDimensionsAll(
    topObject,
    basis,
    alphas: np.ndarray | None = None,
    maxCells: int | None = None,
    randomState: int = 0,
    filterToCorrectAtAlpha0: bool = True,
    verbose: bool = True,
    includeCriteria: pd.Series | None = None,
    precision: int = 0.01,
    cellTypes: list | None = None,
    diffCount: int | None = None
):

    alphas = np.linspace(0, 0.8, 17) if alphas is None else alphas
    dimensionsMap = {}
    for sourceLabel in basis.columns if cellTypes is None else cellTypes:
        dimensionsMap[sourceLabel] = getSpaceDimensions(topObject, basis, sourceLabel, alphas=alphas, maxCells=maxCells, randomState=randomState, verbose=False, 
                                            includeCriteria=includeCriteria, precision=precision, cellTypes=cellTypes)

    return dimensionsMap


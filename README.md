# DIHS Correlator

Python implementation and API of the DIHS-based tephra correlation framework presented in Aymerich et al.'s "A New Machine Learning Approach for Interpretable Tephra-Source Correlation: Introducing the Depth-Integrated Harmonic Score (DIHS)" (2026).

## Layout

- `core/`: transformations, recursive clustering, DIHS metrics
- `workflows/`: single-run, perturbative uncertainty propagation and pseudo-unknown run
- `viz/`: HS-depth curves, pairwise plots, summary plots
- `io/`: loading and output path/writer helpers

## Public API

Import from package root:

```python
from DIHS_Correlator import (
    simple_run,
    triple_run,
    perturbative_simple_run,
    perturbative_triple_run,
    pseudo_unknown_run,
)
```

Simple run (single model without uncertainty propagation):

```python
hs_dict = simple_run(
    df=my_dataframe, # table with geochemical data
    model_type="kmeans", # clustering model for recursive binary partition of the dataset (choose from "kmeans", "gaussian" or "agglomerative")
    transform_type="clr", # CoDa transformation of the data (choose from none, "scaled", "clr" or "ilr")
    class_column="volcano", # column that encodes classes
    unknown_sample="Caio", # unknown sample to correlate
    random_state=19062026, # fixes an initialization seed for the non-deterministic "kmeans" and "gaussian" models; does nothing if "agglomerative" is chosen; if None, random initialization for "kmeans" and "gaussian" is set
    compute_pairwise=True, # (switch) computes DIHS between all featured classes, not only the unknown one; unknown class still controls max depth if no max_depth is set; increases computation time if it is not needed
    plot_everything=False, # (switch) save all available plots to plot_output_dir
    write_files=False, # (switch) save metrics/tree CSV outputs
    output_dir = "./Results", # file output directory
    plot_output_dir = "./Plots" # plot output directory
    max_depth=100, # maximum tree depth; otherwise controled by stopping conditions (Aymerich et al., 2026)
    exclude_columns=(), # columns/features to be excluded from consideration
    save_cluster_data = False, # (switch) save per-partition cluster datasets to output_dir/ClusterData; can be used independently of write_files
    save_untransformed = False, # (switch) does nothing if save_cluster_data = False; if True, cluster data is saved untransformed
    verbose = True, # (switch) if True, progress comments are printed on terminal
    return_details = False, # (switch) receive DIHS totals, pairwise outputs, and artifact paths on final dictionary (hs_dict)
)
```

Triple run (agglomerative + kmeans + gaussian, without uncertainty propagation):

```python
hs_all_models = triple_run(
    df = my_dataframe,
    transform_type = "clr",
    unknown_sample = "Caio",
    class_column = "volcano",
    random_state = None,
    compute_pairwise = True,
    plot_everything = False,
    write_files = False,
    output_dir = "./Results_triple",
    plot_output_dir = None,
    max_depth = 100,
    exclude_columns = (),
    save_cluster_data = False,
    save_untransformed = False,
    verbose = True,
    return_details = False,
)
```

Perturbative run (single model with uncertainty propagation):

```python
hs_mean_df = perturbative_simple_run(
    df = my_dataframe,
    model_type = "agglomerative",
    transform_type = "scaled",
    unknown_sample = "Caio",
    class_column = "sample",
    random_state = None,
    n_iterations = 100, # number of newly perturbed versions of the original dataset to generate for uncertainty propagation
    major_cols =['SiO2','CaO','MgO','MnO','Al2O3','FeO','TiO2','K2O','Na2O'], # explicit list of major-element columns to perturb; pass these explicitly if your dataset does not use the package defaults
    trace_cols = ['Zr','La','Ba','Ce','Eu','Nb'], # explicit list of trace-element columns to perturb; explicit lists raise if none of the requested columns resolve
    major_error = 0.02, # uncertainty associated with major_cols; perturbations for this feature subset will be within +- this float
    trace_error = 0.10, # uncertainty associated with trace_cols; perturbations for this feature subset will be within +- this float
    perturbation_seed = 26122001, # fixes a seed for perturbation; otherwise random
    compute_pairwise = True, # (switch) computes DIHS and ensemble DIHS average between all featured classes, not only the unknown one; unknown class still controls max depth if no max_depth is set; increases computation time if it is not needed
    plot_everything = False,
    write_files = False,
    output_dir = "./Results_perturbative",
    plot_output_dir = None,
    max_depth = 100,
    exclude_columns = (),
    save_cluster_data = False,
    save_untransformed = False,
    verbose = True,
    return_details = False, # (switch) receive HS mean per depth, individual HS iterations, DIHS summary, individual DIHS iterations, Top-1 class frequency, DIHS margin summary, DIHS margin per iteration and optional pairwise mean/std matrices
)
```
---------------------------------------------------------------
NOTE: For perturbative runs, `dihs_summary` is computed from per-iteration HS truncated to the ensemble maximum common depth (`common_depth_level`), then integrated on that shared depth.
---------------------------------------------------------------

If `major_cols` and `trace_cols` are left as `None`, the perturbative routines auto-detect normalized defaults such as `SIO2N`, `TIO2N`, `AL2O3N`, `FE2O3TN`, `CAON`, `MGON`, `MNON`, `NA2ON`, `K2ON`, `P2O5N`, `NbN`, `ZrN`, `LaN`, `CeN`, `SrN`, `BaN`, and `RbN`. If no defaults resolve, the run proceeds without perturbation and emits a warning; if you pass explicit lists and none of their columns resolve, the run raises an error.

Perturbative triple run (agglomerative + kmeans + gaussian with uncertainty propagation):

```python
hs_mean_dict = perturbative_triple_run(
    df = my_dataframe,
    transform_type = "scaled",
    unknown_sample = "Caio",
    class_column = "sample",
    random_state = None,
    n_iterations = 100,
    major_cols =['SiO2','CaO','MgO','MnO','Al2O3','FeO','TiO2','K2O','Na2O'],
    trace_cols = ['Zr','La','Ba','Ce','Eu','Nb'], 
    major_error = 0.02,
    trace_error = 0.10, 
    perturbation_seed = 12345, 
    compute_pairwise = True, 
    plot_everything = False,
    write_files = False,
    output_dir = "./Results_perturbative",
    plot_output_dir = None,
    max_depth = 100,
    exclude_columns = (),
    save_cluster_data = False,
    save_untransformed = False,
    verbose = True,
    return_details = False, 
)
```

Pseudo-unknown calibration run (single model):

```python
pseudo_dict = pseudo_unknown_run(
    df = my_dataframe,
    model_type = "gaussian",
    transform_type = "clr",
    class_column = "unit",
    sample_size = 5, # number of datapoints to take of each class as unknown per iteration in the pseudo-unknown framework
    n_iterations = 10, # number of subsampling iterations for each class
    excluded_classes = None, # classes to exclude from pseudo-unknown metrics computation; during processing they are still part of the dataset as potential-source classes and affect tree construction
    random_state = None,
    max_depth = 100,
    exclude_columns=(),
    target_precision = 0.95,
    reported_precisions = None,
    min_runs_above_threshold = 1,
    plot_everything = True, # (switch) save pseudo-unknown plots to plot_output_dir
    write_files = False, # (switch) save pseudo-unknown CSV outputs
    output_dir = "./Results_pseudo_unknown",
    plot_output_dir = None,
    verbose = True,
    return_details = False,
)
```

Set `return_details=True` to get the full calibration bundle, including:

- `run_results`
- `summary_by_class`, `summary_by_case`
- `threshold_curve`, `threshold_summary`
- `thresholds_by_target_precision`
- `resolvedness_threshold`
- optional artifact paths for plots and CSV outputs

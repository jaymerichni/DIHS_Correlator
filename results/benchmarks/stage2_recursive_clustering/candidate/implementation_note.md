# Stage 2 Implementation Note

The recursive clustering core in `src/DIHS_Correlator/core/clustering.py` was refactored to keep one shared feature matrix, one shared class-value array, and one shared row-id array for the full recursive descent.

Removed dataframe churn:

1. Removed per-node `df[features].to_numpy(...)` regeneration by materializing the feature matrix once at the root.
2. Removed recursive `sub_df.copy()` calls by recursing on integer row-position subsets instead of child dataframes.
3. Removed repeated `df[df[col] == label]` child dataframe construction from the hot path and replaced it with row-position selection plus ordered subtree traversal.
4. Removed recursive `pd.concat(result)` assembly by computing subtree row order during recursion and materializing the output dataframe once at the end.
5. Replaced repeated dataframe-based class-cardinality checks with array-based same-class checks on the shared class-value array.
6. Replaced the terminal all-rows-identical guard with a direct shared-array comparison helper, avoiding deeper dataframe work when a node has no feature variability.

Behavior preserved:

1. Output depth-column naming and sparsity semantics match the pre-refactor behavior.
2. Cluster metadata paths, labels, and stored row indices remain baseline-compatible.
3. Workflow-level DIHS and ranking outputs matched the frozen baseline snapshots across the benchmark matrix.

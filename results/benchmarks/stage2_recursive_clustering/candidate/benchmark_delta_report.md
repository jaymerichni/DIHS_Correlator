# Stage 2 Recursive Clustering Benchmark Delta Report

| scenario | dataset_profile | method | baseline_seconds_median | candidate_seconds_median | delta_seconds | percent_change | speedup_ratio | sample_count | pass_or_fail | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| run_single_model_workflow | medium | kmeans | 2.746039 | 1.260236 | 1.485803 | 54.11 | 2.179 | 5 | pass | persistent_overlap profile; compute_pairwise=False; write_files=False |
| run_single_model_workflow | medium | agglomerative | 0.570655 | 0.432537 | 0.138117 | 24.20 | 1.319 | 5 | pass | persistent_overlap profile; compute_pairwise=False; write_files=False |
| run_single_model_workflow | stress | kmeans | 5.309668 | 2.059465 | 3.250203 | 61.21 | 2.578 | 5 | pass | two-scenario overlap bundle (persistent_overlap + shallow_only_overlap); compute_pairwise=False; write_files=False |
| run_single_model_workflow | stress | agglomerative | 1.842920 | 0.639682 | 1.203239 | 65.29 | 2.881 | 5 | pass | two-scenario overlap bundle (persistent_overlap + shallow_only_overlap); compute_pairwise=False; write_files=False |
| perturbative_simple_run_workflow | medium | kmeans | 4.453092 | 1.999189 | 2.453904 | 55.11 | 2.227 | 5 | pass | persistent_overlap profile; n_iterations=2; fixed perturbation_seed=321 |
| perturbative_simple_run_workflow | medium | agglomerative | 1.481445 | 0.748616 | 0.732829 | 49.47 | 1.979 | 5 | pass | persistent_overlap profile; n_iterations=2; fixed perturbation_seed=321 |
| run_pseudo_unknown_experiments | medium | kmeans | 8.557897 | 4.054178 | 4.503719 | 52.63 | 2.111 | 5 | pass | persistent_overlap profile; sample_size=5; n_iterations=1; excluded_classes=['C','X'] |
| perturbative_triple_run_with_resolvedness_workflow | small | kmeans (triple workflow timed) | 17.104455 | 12.113891 | 4.990564 | 29.18 | 1.412 | 5 | pass | compact_separated profile; n_iterations=1; pseudo_unknown_iterations=1; timing covers the full triple resolvedness workflow |

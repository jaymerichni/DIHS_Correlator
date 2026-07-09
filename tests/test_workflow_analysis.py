import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from DIHS_Correlator.workflows.analysis import (
    _compute_margin_from_hs_metrics,
    _resolve_major_trace_columns,
)


class AnalysisWorkflowTestCase(unittest.TestCase):
    def test_compute_margin_from_hs_metrics_excludes_unknown_neighbor(self):
        metrics_df = pd.DataFrame(
            [
                {"depth_level": 0, "neighbor_unit": 0, "harmonic_score": 0.5, "unknown_class": 0},
                {"depth_level": 0, "neighbor_unit": 1, "harmonic_score": 1.0, "unknown_class": 0},
                {"depth_level": 1, "neighbor_unit": 1, "harmonic_score": 1.5, "unknown_class": 0},
                {"depth_level": 1, "neighbor_unit": 2, "harmonic_score": 0.5, "unknown_class": 0},
            ]
        )

        result = _compute_margin_from_hs_metrics(metrics_df, integration_depth=1)

        self.assertEqual(result["unknown_class"], 0)
        self.assertAlmostEqual(result["top1"], 1.25)
        self.assertAlmostEqual(result["top2"], 0.25)
        self.assertAlmostEqual(result["dihs_margin"], 1.0)

    def test_resolve_major_trace_columns_uses_default_numeric_columns(self):
        df = pd.DataFrame(
            {
                "controlcode": [0, 1],
                "SIO2N": [1.0, 2.0],
                "TIO2N": [3.0, 4.0],
                "NbN": [5.0, 6.0],
                "category": ["a", "b"],
            }
        )

        major, trace = _resolve_major_trace_columns(df, None, None)

        self.assertEqual(major, ["SIO2N", "TIO2N"])
        self.assertEqual(trace, ["NbN"])


if __name__ == "__main__":
    unittest.main()

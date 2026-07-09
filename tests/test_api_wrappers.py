import os
import sys
import unittest
import pandas as pd
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from DIHS_Correlator import api


class APITestCase(unittest.TestCase):
    def test_simple_run_delegates_to_workflow(self):
        df = pd.DataFrame({"controlcode": [0, 1], "a": [1.0, 2.0], "b": [3.0, 4.0]})

        with mock.patch.object(
            api,
            "run_single_model_workflow",
            return_value={
                "hs_per_depth": "hs",
                "dihs_total": "dihs",
                "artifacts": {},
            },
        ) as mocked:
            result = api.simple_run(df=df, model_type="kmeans")

        mocked.assert_called_once()
        self.assertEqual(result, "hs")

    def test_package_imports_from_repo_root(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        original_path = list(sys.path)
        sys.path[:] = [repo_root] + [entry for entry in original_path if os.path.abspath(entry) != repo_root]

        try:
            sys.modules.pop("DIHS_Correlator", None)
            imported = __import__("DIHS_Correlator")
            self.assertTrue(hasattr(imported, "simple_run"))
        finally:
            sys.path[:] = original_path
            sys.modules.pop("DIHS_Correlator", None)

    def test_workflow_modules_do_not_import_api_layer(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        workflows_root = os.path.join(repo_root, "workflows")

        violations = []
        for root, _, files in os.walk(workflows_root):
            for filename in files:
                if not filename.endswith(".py"):
                    continue
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                if (
                    "from DIHS_Correlator.api import" in content
                    or "import DIHS_Correlator.api" in content
                ):
                    rel_path = os.path.relpath(file_path, repo_root).replace("\\", "/")
                    violations.append(rel_path)

        self.assertEqual(
            violations,
            [],
            msg=(
                "Workflow modules must not import the API layer. "
                f"Violations: {violations}"
            ),
        )


if __name__ == "__main__":
    unittest.main()

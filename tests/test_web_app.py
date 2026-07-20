import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from DIHS_Correlator import __version__  # noqa: E402
import DIHS_Correlator.api as api_module  # noqa: E402
from DIHS_Correlator.web import create_app  # noqa: E402
from DIHS_Correlator.web.__main__ import main as module_main  # noqa: E402

web_app_module = importlib.import_module("DIHS_Correlator.web.app")


class WebAppSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = create_app().test_client()

    def test_index_page_renders_expected_title_and_assets(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DIHS Tephra Correlator", response.data)
        self.assertIn(b"/static/app.css", response.data)
        self.assertIn(b"/static/app.js", response.data)

    def test_package_version_is_exposed(self) -> None:
        self.assertEqual(__version__, "0.1.0")

    def test_module_entrypoint_is_callable(self) -> None:
        self.assertTrue(callable(module_main))

    def test_background_job_uses_parsed_dataset_dataframe(self) -> None:
        parsed_df = pd.DataFrame({"SiO2": [74.2], "controlcode": ["A"]})
        parsed = {
            "mode": "simple_run",
            "dataset_df": parsed_df,
            "common": {"output_dir": "unused", "write_files": False},
            "advanced": {},
            "form_state": {},
            "temp_root": Path("unused"),
        }

        with (
            patch.object(web_app_module, "_update_job") as update_job,
            patch.object(web_app_module, "_store_result", return_value="result-1") as store_result,
            patch.object(
                web_app_module,
                "_run_selected_mode",
                return_value={"resolvedness_summary": pd.DataFrame([{"score": 1.0}])},
            ) as run_selected_mode,
        ):
            web_app_module._run_analysis_job(
                job_id="job-1",
                dataset_id="dataset-1",
                parsed=parsed,
            )

        run_selected_mode.assert_called_once()
        self.assertTrue(run_selected_mode.call_args.args[1].equals(parsed_df))
        store_result.assert_called_once()
        self.assertEqual(update_job.call_args_list[-1].kwargs["status"], "completed")

    def test_resolvedness_api_accepts_progress_callback(self) -> None:
        df = pd.DataFrame({"controlcode": ["A"], "SiO2": [74.2]})
        callback = lambda payload: payload

        with patch.object(
            api_module,
            "perturbative_triple_run_with_resolvedness_workflow",
            return_value={"summary": pd.DataFrame([{"model": "kmeans"}])},
        ) as workflow:
            api_module.perturbative_triple_run_with_resolvedness(
                df=df,
                return_details=True,
                progress_callback=callback,
            )

        self.assertIs(workflow.call_args.kwargs["progress_callback"], callback)


if __name__ == "__main__":
    unittest.main()

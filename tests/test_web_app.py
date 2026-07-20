import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from DIHS_Correlator import __version__  # noqa: E402
from DIHS_Correlator.web import create_app  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()

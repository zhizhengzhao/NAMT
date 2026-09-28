from __future__ import annotations

import unittest
from pathlib import Path

from namt.config import CALIBRATIONS, SCENES, SEEDS, load_config
from reproduce import experiments, artifact_path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseInputsTest(unittest.TestCase):
    def test_all_experiments_have_reference_artifacts(self):
        jobs = experiments("all")
        self.assertEqual(len(jobs), 825)
        paths = [artifact_path(ROOT / "reference", job) for job in jobs]
        self.assertEqual(len(set(paths)), len(paths))
        observed = {path.resolve() for suite in ("benchmark", "ablation") for path in (ROOT / "reference" / suite).rglob("*.npz")}
        self.assertEqual({path.resolve() for path in paths}, observed)

    def test_selected_runs_have_input_data_and_calibrations(self):
        config = load_config()
        for seed in SEEDS:
            for scene in ("blank", *SCENES):
                with self.subTest(seed=seed, scene=scene):
                    self.assertTrue((ROOT / "data" / f"seed_{seed}" / f"A_{scene}.npz").is_file())
            for calibration in CALIBRATIONS:
                for method in ("namt_3p", "namt_4p"):
                    filename = config["methods"][method]["calibration"]
                    with self.subTest(seed=seed, calibration=calibration, method=method):
                        self.assertTrue((ROOT / "assets" / f"seed_{seed}" / calibration / filename).is_file())


if __name__ == "__main__":
    unittest.main()

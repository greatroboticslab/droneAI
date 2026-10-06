import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_dashboard as td  # noqa: E402


def write_run(root, name, metrics, predictions=None):
    run = Path(root) / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(metrics))
    if predictions is not None:
        pd.DataFrame(predictions).to_csv(run / "predictions.csv", index=False)


class TrainingDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        write_run(self.tmp.name, "old_clip_split", {"accuracy": 0.9, "split_mode": "clip"})
        write_run(self.tmp.name, "grouped", {
            "accuracy": 0.5, "accuracy_std": 0.1, "macro_f1": 0.4, "split_mode": "grouped_kfold_video",
            "folds": 2, "label_counts": {"takeoff": 3, "land": 1},
            "per_fold_best_model": [{"fold": 0, "accuracy": 0.6, "val_clips": 2}],
            "all_model_results": [{"model": "svm_rbf", "accuracy": 0.4}],
        }, predictions={
            "clip_group": ["a", "b", "c", "d"], "video_id": ["v1", "v1", "v2", "v3"],
            "true_label": ["takeoff", "takeoff", "takeoff", "land"],
            "pred_label": ["takeoff", "takeoff", "land", "takeoff"],
            "prob_takeoff": [0.9, 0.8, 0.3, 0.7], "prob_land": [0.1, 0.2, 0.7, 0.3],
        })

    def test_honest_run_is_shown_first_and_old_split_is_flagged(self):
        data = td.load_training_dashboard(runs_dir=self.tmp.name)
        self.assertEqual(data["run"]["name"], "grouped")
        old = next(r for r in data["runs"] if r["name"] == "old_clip_split")
        self.assertFalse(old["honest"])

    def test_per_event_scores_majority_and_mistakes(self):
        run = td.load_training_dashboard(runs_dir=self.tmp.name)["run"]
        takeoff = next(c for c in run["per_class"] if c["event"] == "takeoff")
        self.assertEqual((takeoff["found"], takeoff["clips"], takeoff["predicted"]), (2, 3, 3))
        self.assertAlmostEqual(run["majority"], 0.75)
        self.assertEqual(run["mistakes"][0]["clip"], "c")      # most confident wrong answer first
        self.assertEqual(run["models"][0]["name"], "SVM (RBF kernel)")
        self.assertIsNone(run["models"][0]["accuracy_std"])

    def test_old_run_without_predictions_still_loads(self):
        run = td.load_training_dashboard("old_clip_split", runs_dir=self.tmp.name)["run"]
        self.assertEqual(run["per_class"], [])
        self.assertEqual(run["folds"], [])

    def test_feature_names_are_readable(self):
        self.assertEqual(td.feature_label("frame_mag_norm_per_sec_start_mean"), "Whole-frame speed, at the start")
        self.assertEqual(td.feature_label("det_max_downward_vy"), "Drone box fastest drop")
        self.assertEqual(td.feature_label("something_new"), "something_new")


if __name__ == "__main__":
    unittest.main()

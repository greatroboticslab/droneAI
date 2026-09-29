"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repo_paths import LABELGUI_DIR, REPO_ROOT, repo_rel, resolve_path  # noqa: E402


class RepoRelTests(unittest.TestCase):
    def test_path_inside_repo_is_relative_posix(self):
        p = REPO_ROOT / "LabelGUI" / "TrainingRuns" / "run1" / "metrics.json"
        self.assertEqual(repo_rel(p), "LabelGUI/TrainingRuns/run1/metrics.json")

    def test_foreign_windows_path_is_remapped(self):
        text = r"C:\Users\someone\droneAI\LabelGUI\OpticalFlowResults\flow_v1\features.csv"
        self.assertEqual(repo_rel(text), "LabelGUI/OpticalFlowResults/flow_v1/features.csv")

    def test_path_outside_repo_stays_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            out = repo_rel(Path(d) / "x.csv")
            self.assertTrue(out.startswith("/"))
            self.assertTrue(out.endswith("/x.csv"))
            self.assertNotIn("\\", out)

    def test_foreign_path_without_repo_folder_is_kept(self):
        self.assertEqual(repo_rel(r"D:\datasets\sintel\clean"), "D:/datasets/sintel/clean")

    def test_none_and_empty(self):
        self.assertEqual(repo_rel(None), "")
        self.assertEqual(repo_rel(""), "")


class ResolvePathTests(unittest.TestCase):
    def test_empty_values_give_none(self):
        for v in (None, "", "  ", float("nan"), "nan"):
            self.assertIsNone(resolve_path(v))

    def test_repo_relative_path_works_from_any_folder(self):
        old = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as d:
                os.chdir(d)
                got = resolve_path("LabelGUI/repo_paths.py", must_exist=True)
        finally:
            os.chdir(old)
        self.assertEqual(got, (LABELGUI_DIR / "repo_paths.py").resolve())

    def test_labelgui_relative_path_works(self):
        got = resolve_path("repo_paths.py", must_exist=True)
        self.assertEqual(got, (LABELGUI_DIR / "repo_paths.py").resolve())

    def test_backslashes_in_relative_path(self):
        got = resolve_path(r"LabelGUI\tests\test_repo_paths.py", must_exist=True)
        self.assertEqual(got, Path(__file__).resolve())

    def test_foreign_windows_path_is_remapped_into_this_repo(self):
        text = r"C:\Users\someone\droneAI\LabelGUI\repo_paths.py"
        self.assertEqual(resolve_path(text, must_exist=True), LABELGUI_DIR / "repo_paths.py")

    def test_foreign_posix_path_is_remapped_into_this_repo(self):
        text = "/home/someone/code/droneAI/LabelGUI/repo_paths.py"
        self.assertEqual(resolve_path(text, must_exist=True), LABELGUI_DIR / "repo_paths.py")

    def test_existing_absolute_path_is_kept(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.csv"
            f.write_text("x")
            self.assertEqual(resolve_path(str(f)), f.resolve())

    def test_extra_base_is_searched(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "frames").mkdir()
            got = resolve_path("frames", d, must_exist=True)
            self.assertEqual(got, (Path(d) / "frames").resolve())

    def test_missing_path_raises_when_required(self):
        with self.assertRaises(FileNotFoundError):
            resolve_path("LabelGUI/does_not_exist.csv", must_exist=True)

    def test_missing_path_returns_repo_guess(self):
        self.assertEqual(resolve_path("LabelGUI/nope.csv"), (REPO_ROOT / "LabelGUI" / "nope.csv").resolve())


if __name__ == "__main__":
    unittest.main()

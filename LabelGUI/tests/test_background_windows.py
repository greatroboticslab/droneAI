"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # test_clip_dataset helpers

from background_windows import (  # noqa: E402
    background_windows,
    free_intervals,
    nearest_event_distance,
    usable_duration,
)

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


class BackgroundWindowTests(unittest.TestCase):
    def test_free_intervals(self):
        self.assertEqual(free_intervals(60, [10, 12, 40], 3), [(0, 7), (15, 37), (43, 60)])
        self.assertEqual(free_intervals(20, [], 3), [(0.0, 20)])
        self.assertEqual(free_intervals(10, [1, 9], 3), [(4, 6)])

    def test_windows_keep_the_gap_and_stay_inside(self):
        events = [10.0, 12.0, 40.0, 95.0]
        wins = background_windows(120.0, events, length=12.0, gap=3.0, max_windows=50)
        self.assertTrue(wins)
        for a, b in wins:
            self.assertAlmostEqual(b - a, 12.0)
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(b, 120.0)
            for t in events:
                self.assertGreaterEqual(max(a - t, t - b), 3.0 - 1e-9, (a, b, t))
        # Non-overlapping
        for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
            self.assertLessEqual(b1, a2 + 1e-9)

    def test_max_windows_spreads_over_the_video(self):
        wins = background_windows(600.0, [], length=12.0, gap=3.0, max_windows=5)
        self.assertEqual(len(wins), 5)
        self.assertLess(wins[0][0], 60)
        self.assertGreater(wins[-1][1], 540)

    def test_no_room(self):
        self.assertEqual(background_windows(20.0, [5.0, 12.0], length=12.0, gap=3.0, max_windows=5), [])
        self.assertEqual(background_windows(0.0, [], length=12.0, gap=3.0, max_windows=5), [])

    def test_video_without_events_is_all_background(self):
        self.assertEqual(len(background_windows(36.0, [], length=12.0, gap=3.0, max_windows=10)), 3)

    def test_usable_duration_stops_where_the_labeler_stopped(self):
        self.assertEqual(usable_duration(120.0, {"watched_until_sec": 45.0}), (45.0, True))
        self.assertEqual(usable_duration(120.0, {"watched_until_sec": 0.0}), (120.0, False))
        self.assertEqual(usable_duration(0.0, {"duration_sec": 80.0, "watched_until_sec": 0}), (80.0, False))
        # Finished early at 45 s: no window may reach into the unwatched part.
        wins = background_windows(usable_duration(120.0, {"watched_until_sec": 45.0})[0], [10.0],
                                  length=12.0, gap=3.0, max_windows=10)
        self.assertTrue(wins)
        self.assertTrue(all(b <= 45.0 for _, b in wins))

    def test_nearest_event_distance(self):
        self.assertEqual(nearest_event_distance(20, 32, [10, 40]), 8)
        self.assertEqual(nearest_event_distance(20, 32, []), "")

    @unittest.skipUnless(HAVE_CV2, "opencv not installed")
    def test_cut_background_frames(self):
        from clip_dataset import cut_segments, open_video
        from test_clip_dataset import frame_numbers, make_numbered_video

        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / "v.mp4"
            make_numbered_video(video, seconds=30, fps=10)
            wins = background_windows(30.0, [5.0], length=4.0, gap=3.0, max_windows=2)
            cap, fps, _ = open_video(video)
            paths = [Path(d) / f"bg_{k}.mp4" for k in range(len(wins))]
            counts = cut_segments(cap, fps, [(a, b, p) for (a, b), p in zip(wins, paths)])
            cap.release()
            for (a, b), p, n in zip(wins, paths, counts):
                nums = frame_numbers(p)
                self.assertEqual(nums, list(range(round(a * 10), round(b * 10) + 1)))
                self.assertEqual(n, len(nums))
                self.assertTrue(all(abs(x / 10 - 5.0) >= 3.0 for x in nums))


if __name__ == "__main__":
    unittest.main()

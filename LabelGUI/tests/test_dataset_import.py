"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
"""
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from dataset_import import (  # noqa: E402
    build_dataset_items_from_excel,
    build_video_file_item,
    normalize_source,
    normalize_view_type,
    save_uploaded_video,
    youtube_video_id,
)
from db.db_store import DBStore  # noqa: E402


def excel_bytes(rows):
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False)
    return buf.getvalue()


class FakeUpload:
    """The bits of werkzeug's FileStorage that save_uploaded_video uses."""

    def __init__(self, filename, data: bytes):
        self.filename = filename
        self.stream = io.BytesIO(data)


class ExcelImportTests(unittest.TestCase):
    def test_old_two_column_sheet_still_works(self):
        data = excel_bytes([{"Persona Name": "Ann", "Youtube Link": "https://youtu.be/BV7vi3VJgKI"}])
        items = build_dataset_items_from_excel(data, "ds")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["video_id"], "BV7vi3VJgKI")
        self.assertEqual(items[0]["view_type"], "Unknown")
        self.assertEqual(items[0]["source"], "Unknown")
        self.assertEqual(items[0]["local_path"], "")

    def test_optional_columns_are_read_and_normalized(self):
        data = excel_bytes([{
            "Persona Name": "Ann",
            "Youtube Link": "https://www.youtube.com/watch?v=Gd12g9XqSo4",
            "Video ID": "V001",
            "View Type": "fpv",
            "Source": "sim",
        }])
        item = build_dataset_items_from_excel(data, "ds")[0]
        self.assertEqual(item["video_id"], "V001")
        self.assertEqual(item["view_type"], "FPV")
        self.assertEqual(item["source"], "Simulation")

    def test_rows_without_link_are_skipped_and_numbering_stays_dense(self):
        data = excel_bytes([
            {"Name": "A", "Link": "https://youtu.be/aaaaaaaaaaa"},
            {"Name": "B", "Link": None},
            {"Name": "C", "Link": "https://youtu.be/ccccccccccc"},
        ])
        items = build_dataset_items_from_excel(data, "ds")
        self.assertEqual([i["row_index"] for i in items], [1, 2])
        self.assertEqual([i["person_name"] for i in items], ["A", "C"])

    def test_non_youtube_link_gets_stable_id(self):
        data = excel_bytes([{"Name": "A", "Link": "https://example.com/v.mp4"}])
        a = build_dataset_items_from_excel(data, "ds1")[0]["video_id"]
        b = build_dataset_items_from_excel(data, "ds2")[0]["video_id"]
        self.assertTrue(a.startswith("url-"))
        self.assertEqual(a, b)

    def test_missing_required_columns_raises(self):
        with self.assertRaises(ValueError):
            build_dataset_items_from_excel(excel_bytes([{"Foo": 1}]), "ds")

    def test_normalizers(self):
        self.assertEqual(normalize_view_type("Third Person"), "Third-person")
        self.assertEqual(normalize_view_type("?"), "Unknown")
        self.assertEqual(normalize_source("Real flight"), "Real")
        self.assertEqual(youtube_video_id("not a link"), "")


class VideoUploadTests(unittest.TestCase):
    def test_same_content_same_id_stored_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "videos"
            id1, p1 = save_uploaded_video(FakeUpload("a.MP4", b"frames"), d)
            id2, p2 = save_uploaded_video(FakeUpload("copy.mp4", b"frames"), d)
            id3, _ = save_uploaded_video(FakeUpload("b.mov", b"other"), d)
            self.assertEqual(id1, id2)
            self.assertEqual(p1, p2)
            self.assertNotEqual(id1, id3)
            self.assertEqual(p1.suffix, ".mp4")
            self.assertEqual(sorted(x.suffix for x in d.iterdir()), [".mov", ".mp4"])

    def test_video_item_uses_relative_path_and_validates_choices(self):
        repo = Path("/repo")
        item = build_video_file_item(
            "ds", 3, "file-abc", repo / "LabelGUI/Uploads/videos/file-abc.mp4",
            "flight 1.mp4", "", "FPV", "bogus", repo,
        )
        self.assertEqual(item["local_path"], "LabelGUI/Uploads/videos/file-abc.mp4")
        self.assertEqual(item["person_name"], "flight 1")
        self.assertEqual(item["source"], "Unknown")
        self.assertEqual(item["item_key"], "ds:3")


class DBStoreTests(unittest.TestCase):
    def test_migrates_old_dataset_items_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            conn = sqlite3.connect(path)
            conn.execute("""
                CREATE TABLE dataset_items (
                    item_key TEXT PRIMARY KEY, dataset_key TEXT NOT NULL,
                    row_index INTEGER NOT NULL, person_name TEXT DEFAULT '',
                    youtube_link TEXT NOT NULL, status TEXT DEFAULT 'not_labeled',
                    labeled_by TEXT DEFAULT '', locked_by TEXT DEFAULT '',
                    scenario_type TEXT DEFAULT '', updated_at TEXT,
                    UNIQUE(dataset_key, row_index))
            """)
            conn.execute("INSERT INTO dataset_items (item_key, dataset_key, row_index, youtube_link) "
                         "VALUES ('ds:1', 'ds', 1, 'https://youtu.be/x')")
            conn.commit()
            conn.close()

            db = DBStore(str(path))
            item = db.get_dataset_item("ds:1")
            self.assertEqual(item["youtube_link"], "https://youtu.be/x")
            self.assertEqual(item["local_path"], "")

    def test_add_items_appends_after_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DBStore(str(Path(tmp) / "t.sqlite"))
            db.save_dataset("ds", "Test", "x.xlsx", b"", "me", is_active=True)
            self.assertEqual(db.next_row_index("ds"), 1)
            db.replace_dataset_items("ds", [{"item_key": "ds:1", "row_index": 1,
                                             "youtube_link": "https://youtu.be/x"}])
            self.assertEqual(db.next_row_index("ds"), 2)
            db.add_dataset_items("ds", [{
                "item_key": "ds:2", "row_index": 2, "youtube_link": "a.mp4",
                "video_id": "file-1", "view_type": "FPV", "source": "Real",
                "local_path": "LabelGUI/Uploads/videos/file-1.mp4",
            }])
            items = db.list_dataset_items("ds")
            self.assertEqual([i["row_index"] for i in items], [1, 2])
            self.assertEqual(items[1]["view_type"], "FPV")
            self.assertEqual(items[1]["local_path"], "LabelGUI/Uploads/videos/file-1.mp4")


if __name__ == "__main__":
    unittest.main()

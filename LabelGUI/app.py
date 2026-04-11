from flask import Flask, render_template, request, Response, redirect, url_for, jsonify, session, send_file
import os
import logging
import pandas as pd
import io
import zipfile
import base64
import uuid

from datetime import datetime
from functools import wraps
from mqtt_client import MQTTManager
from flask import send_file, session
from pathlib import Path
from functools import wraps
from db.db_store import DBStore
from mqtt_client import MQTTManager

# ===================== VALIDATION BACKEND =====================
from validation_backend import (
    start_validation_thread,
    get_crash_count,
    mark_event_now,
    is_video_done,
    get_extraction_progress,
    toggle_pause,
    skip_video,
    get_current_validation_link,
)

# ===================== TRAINING BACKEND =====================
from training_backend import (
    start_training_session,
    resume_training_session,
    is_training_done,
    get_training_status,
    get_training_progress,
    generate_training_video_stream,
    finalize_training_session,
    generate_training_preview_stream,
    set_current_label,
    load_label_group_file,
    save_label_group_file,
)

# ===================== CRASH VERIFY BACKEND (DB-backed) =====================
from crash_verify_backend import (
    list_items_for_table,
    start_session,
    stream_video,
    mark_plus_one,
    finish_and_save,
    save_label_later,
    export_excel,
    get_session,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "droneai-secret-key"

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"

db = DBStore(str(DB_PATH))
mqtt_mgr = MQTTManager()

TEAM_PASSWORD = "droneai2025"

mqtt_mgr = MQTTManager()
TEAM_PASSWORD = os.environ.get("DRONEAI_TEAM_PASSWORD", "droneai2025")

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def leader_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        if session.get("role") != "leader":
            return "Leader access only.", 403
        return fn(*args, **kwargs)
    return wrapper


def _db_path():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_dir, "db", "droneai.sqlite")


def is_video_locked(lock_key, current_user):
    info = mqtt_mgr.locks.get(lock_key)
    if not info:
        return False, None

    status = info.get("status", "")
    by_user = info.get("by", "")

    if status == "claimed" and by_user != current_user:
        return True, by_user

    return False, by_user

def _normalize_dataset_columns(df):
    cols = {str(c).strip().lower(): c for c in df.columns}

    person_col = None
    link_col = None

    person_candidates = ["persona name", "person name", "person", "name"]
    link_candidates = ["youtube link", "youtube url", "youtube", "link", "url"]

    for c in person_candidates:
        if c in cols:
            person_col = cols[c]
            break

    for c in link_candidates:
        if c in cols:
            link_col = cols[c]
            break

    if not person_col or not link_col:
        raise ValueError("Excel must contain a person/name column and a YouTube link column.")

    return person_col, link_col

def build_dataset_items_from_excel(file_bytes: bytes, dataset_key: str):
    import pandas as pd
    from io import BytesIO

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)
    person_col, link_col = _normalize_dataset_columns(df)

    items = []
    row_num = 0
    for _, row in df.iterrows():
        person = str(row.get(person_col) or "").strip()
        link = str(row.get(link_col) or "").strip()
        if not link:
            continue
        row_num += 1
        items.append({
            "item_key": f"{dataset_key}:{row_num}",
            "row_index": row_num,
            "person_name": person,
            "youtube_link": link,
            "status": "not_labeled",
            "labeled_by": "",
            "locked_by": "",
            "scenario_type": "",
        })
    return items

def active_dataset_with_stats():
    active = db.get_active_dataset()
    if not active:
        return None, {"total": 0, "not_labeled": 0, "in_progress": 0, "labeled": 0}
    stats = db.dataset_stats(active["dataset_key"])
    return active, stats

def handle_mqtt_event(data: dict):
    event_type = data.get("type", "")

    if event_type == "dataset_uploaded":
        dataset_key = data.get("dataset_key")
        if not dataset_key or db.dataset_exists(dataset_key):
            return

        file_b64 = data.get("file_b64", "")
        file_bytes = base64.b64decode(file_b64.encode("utf-8"))

        db.save_dataset(
            dataset_key=dataset_key,
            dataset_name=data.get("dataset_name", "Dataset"),
            original_filename=data.get("original_filename", "dataset.xlsx"),
            file_blob=file_bytes,
            uploaded_by=data.get("uploaded_by", ""),
            is_active=bool(data.get("is_active", False)),
        )
        items = build_dataset_items_from_excel(file_bytes, dataset_key)
        db.replace_dataset_items(dataset_key, items)

    elif event_type == "dataset_set_active":
        dataset_key = data.get("dataset_key")
        if dataset_key:
            db.set_active_dataset(dataset_key)

    elif event_type == "queue_claimed":
        item_key = data.get("item_key")
        if item_key:
            db.update_dataset_item(
                item_key=item_key,
                status="in_progress",
                locked_by=data.get("by", ""),
                scenario_type=data.get("scenario_type", ""),
            )

    elif event_type == "item_labeled":
        item_key = data.get("item_key")
        if item_key:
            db.update_dataset_item(
                item_key=item_key,
                status="labeled",
                labeled_by=data.get("by", ""),
                locked_by="",
                scenario_type=data.get("scenario_type", ""),
            )

    elif event_type == "queue_released":
        item_key = data.get("item_key")
        if item_key:
            db.update_dataset_item(
                item_key=item_key,
                status=data.get("status", "not_labeled"),
                locked_by="",
            )

mqtt_mgr.message_handler = handle_mqtt_event
    
# ============ In-memory cache of Excel entries ============
EXCEL_ENTRIES = []       # [{"id": 1, "person": "...", "link": "..."}]
LAST_EXCEL_PATH = None   # used so we can reuse uploaded Excel path (optional)


###############################################################################
#                           VALIDATION WORKFLOW                               #
###############################################################################
@app.route("/validation_index", methods=["GET", "POST"])
def validation_index():
    if request.method == "POST":
        youtube_link = request.form.get("youtube_link", "")
        folder_name = request.form.get("folder_name", "")
        delete_original = True if request.form.get("delete_original") == "on" else False

        current_user = session.get("user", "unknown")
        lock_key = f"val:{youtube_link}"

        locked, by_user = is_video_locked(lock_key, current_user)
        if locked:
            return f"This video is currently being labeled by {by_user}. Please wait or import the latest DB first.", 400

        start_validation_thread(
            youtube_link=youtube_link,
            folder_name=folder_name,
            delete_original=delete_original,
        )

        mqtt_mgr.publish_event("validation_started", {
            "by": current_user,
            "youtube_link": youtube_link,
            "folder_name": folder_name
        })
        mqtt_mgr.publish_lock(lock_key, current_user, "claimed")

        return redirect(url_for("validation_view_stream", source="manual"))

    return render_template("validation_index.html")


@app.route("/validation_view_stream")
@login_required
def validation_view_stream():
    source = request.args.get("source", "manual")
    return render_template("validation_results.html", source=source)


@app.route("/validation_video_feed")
def validation_video_feed():
    from validation_backend import generate_video_stream
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/mark_event", methods=["POST"])
def mark_event():
    data = request.get_json() or {}
    event_type = data.get("event", "unknown")
    mark_event_now(event_type)
    return jsonify(success=True, event=event_type)


@app.route("/check_status")
def check_status():
    return "done" if is_video_done() else "running"


@app.route("/get_crash_count")
def get_crash_count_api():
    return str(get_crash_count())


@app.route("/get_extraction_progress")
def get_extraction_progress_api():
    return jsonify(get_extraction_progress())


@app.route("/toggle_pause", methods=["POST"])
def pause_resume():
    paused = toggle_pause()
    return jsonify({"paused": paused}), 200


@app.route("/rewind", methods=["POST"])
def rewind_video():
    skip_video(-10)
    return ("", 204)


@app.route("/fast_forward", methods=["POST"])
def fast_forward_video():
    skip_video(10)
    return ("", 204)


###############################################################################
#                              EXCEL IMPORT                                   #
###############################################################################
@app.route("/import_excel", methods=["GET", "POST"])
def import_excel():
    """
    GET: Show upload form.
    POST: Save + parse .xlsx into EXCEL_ENTRIES then redirect to picker.
    """
    global EXCEL_ENTRIES, LAST_EXCEL_PATH

    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == "POST":
        file = request.files.get("excel_file")
        if not file or file.filename == "":
            return "Please choose an Excel file (.xlsx).", 400

        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)
        LAST_EXCEL_PATH = save_path

        try:
            df = pd.read_excel(save_path, sheet_name=0)
        except Exception as e:
            return f"Error reading Excel: {e}", 400

        if "Persona Name" not in df.columns or "Youtube Link" not in df.columns:
            return "Excel must contain 'Persona Name' and 'Youtube Link' columns.", 400

        EXCEL_ENTRIES = []
        next_id = 1
        for _, row in df.iterrows():
            person = str(row.get("Persona Name") or "").strip()
            link = str(row.get("Youtube Link") or "").strip()
            if not person or not link:
                continue
            EXCEL_ENTRIES.append({"id": next_id, "person": person, "link": link})
            next_id += 1

        return redirect(url_for("pick_from_excel"))

    return render_template("validation_import.html")


@app.route("/pick_from_excel", methods=["GET"])
def pick_from_excel():
    from validation_backend import get_progress_summary
    progress = get_progress_summary()

    decorated = []
    for e in EXCEL_ENTRIES:
        prefix = (e["person"] or "").strip()[:4] or "User"
        rec = progress.get(prefix, {"sessions": 0, "total_events": 0})
        decorated.append({**e, "prefix": prefix, "labeled": rec["sessions"] > 0})

    return render_template("validation_pick.html", entries=decorated)


@app.route("/start_from_excel", methods=["POST"])
def start_from_excel():
    entry_id = int(request.form.get("entry_id", "0"))
    scenario_base = request.form.get("scenario_base", "Simulation")
    delete_original = True if request.form.get("delete_original") == "on" else False

    match = next((e for e in EXCEL_ENTRIES if e["id"] == entry_id), None)
    if not match:
        return "Invalid selection.", 400

    current_user = session.get("user", "unknown")
    lock_key = f"val:{match['link']}"

    locked, by_user = is_video_locked(lock_key, current_user)
    if locked:
        return f"This video is currently being labeled by {by_user}. Please wait or import the latest DB first.", 400

    start_validation_thread(
        youtube_link=match["link"],
        folder_name=None,
        delete_original=delete_original,
        person_name=match["person"],
        scenario_base=scenario_base,
    )

    mqtt_mgr.publish_event("validation_started", {
        "by": session.get("user", "unknown"),
        "youtube_link": match["link"],
        "person": match["person"],
        "scenario": scenario_base
    })
    mqtt_mgr.publish_lock(f"val:{match['link']}", session.get("user", "unknown"), "claimed")

    return redirect(url_for("validation_view_stream", source="excel"))



###############################################################################
#                 CRASH ANALYSIS + MANUAL VERIFICATION (DB-backed)            #
###############################################################################
@app.route("/crash_analysis")
def crash_analysis():
    items = list_items_for_table()
    return render_template("crash_analysis.html", items=items)


@app.route("/crash_run", methods=["POST"])
def crash_run():
    person = (request.form.get("person") or "").strip()
    scenario = (request.form.get("scenario") or "").strip()
    youtube_link = (request.form.get("youtube_link") or "").strip()
    if not (person and scenario and youtube_link):
        return "Missing person/scenario/link", 400

    sample_fps = float(request.form.get("sample_fps", "2") or 2)
    conf = float(request.form.get("conf", "0.5") or 0.5)

    sid = start_session(person, scenario, youtube_link, sample_fps, conf)
    return redirect(url_for("crash_verify", sid=sid))


@app.route("/crash_verify/<sid>")
def crash_verify(sid):
    s = get_session(sid)
    return render_template(
        "crash_verify.html",
        sid=sid,
        person=s.get("person", ""),
        scenario=s.get("scenario", ""),
        youtube_link=s.get("youtube_link", ""),
        pred_crashes=s.get("pred_crash_events", 0),
        pred_per_min=f'{float(s.get("pred_crashes_per_min", 0) or 0):.3f}',
        ver_crashes=s.get("verified_crash_events", 0),
        ver_per_min=f'{float(s.get("verified_crashes_per_min", 0) or 0):.3f}',
        notes=s.get("notes", ""),
    )


@app.route("/crash_video_feed/<sid>")
def crash_video_feed(sid):
    return Response(stream_video(sid), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/crash_mark/<sid>", methods=["POST"])
def crash_mark(sid):
    mark_plus_one(sid)
    s = get_session(sid)
    return jsonify(ok=True, verified_crash_events=int(s.get("verified_crash_events", 0)))


@app.route("/crash_save_later/<sid>", methods=["POST"])
def crash_save_later(sid):
    data = request.get_json() or {}
    notes = (data.get("notes") or "").strip()
    return jsonify(save_label_later(sid, notes))


@app.route("/crash_finish/<sid>", methods=["POST"])
def crash_finish(sid):
    data = request.get_json() or {}
    notes = (data.get("notes") or "").strip()
    return jsonify(finish_and_save(sid, notes))


@app.route("/crash_export_excel")
def crash_export_excel():
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "analysis", "results", "crash_verification.xlsx"
    )
    export_excel(out_path)
    return redirect(url_for("crash_analysis"))


###############################################################################
#                              TRAINING WORKFLOW                              #
###############################################################################
@app.route("/training_index", methods=["GET", "POST"])
def training_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, "TrainingResults")
    meta_folder = os.path.join(training_dir, "MetaDataLocation")
    os.makedirs(meta_folder, exist_ok=True)

    existing_groups = []
    for fname in os.listdir(meta_folder):
        if fname.endswith(".lblgroup"):
            display_name = os.path.splitext(fname)[0]
            existing_groups.append((fname, display_name))

    if request.method == "POST":
        action = request.form.get("action", "start")
        youtube_link = request.form.get("youtube_link", "")
        user_name = request.form.get("user_name", "")
        delete_original = True if request.form.get("delete_original") == "on" else False
        capture_mode = request.form.get("capture_mode", "10fps")
        custom_fps = float(request.form.get("custom_fps", "1") or 1)
        keep_metadata = True if request.form.get("keep_metadata") == "on" else False

        label_mode = request.form.get("label_mode", "")
        label_group_file = request.form.get("label_group_file", "")

        if action == "resume":
            metadata_path = os.path.join(meta_folder, "metadata.csv")
            if os.path.exists(metadata_path):
                resume_training_session(metadata_file=metadata_path)
                return redirect(url_for("training_view_stream"))
            return "No paused session found. Please start a new session.", 400

        if label_mode == "load_group":
            full_path = os.path.join(meta_folder, label_group_file)
            labels_and_colors = load_label_group_file(full_path)
            if not labels_and_colors:
                return "Error: Label group file was empty or invalid.", 400

        elif label_mode == "create_new":
            new_group_name = request.form.get("new_group_name", "").strip()
            if not new_group_name:
                return "Error: You must provide a label group name when creating a new group.", 400

            label_names = request.form.getlist("label_name[]")
            label_colors = request.form.getlist("label_color[]")
            labels_and_colors = list(zip(label_names, label_colors))

            group_filename = f"{new_group_name}.lblgroup"
            full_path = os.path.join(meta_folder, group_filename)
            save_label_group_file(full_path, labels_and_colors)

        else:
            return "Please select an existing label group or create a new one.", 400

        current_user = session.get("user", "unknown")
        lock_key = f"train:{youtube_link}"

        locked, by_user = is_video_locked(lock_key, current_user)
        if locked:
            return f"This training video is currently being labeled by {by_user}.", 400


        start_training_session(
            youtube_link=youtube_link,
            user_name=user_name,
            delete_original=delete_original,
            capture_mode=capture_mode,
            custom_fps=custom_fps,
            labels_and_colors=labels_and_colors,
            keep_metadata=keep_metadata,
        )

        mqtt_mgr.publish_event("training_started", {
            "by": current_user,
            "youtube_link": youtube_link,
            "user_name": user_name
        })

        mqtt_mgr.publish_lock(lock_key, current_user, "claimed")

        return redirect(url_for("training_preview"))

    return render_template("training_index.html", custom_groups=existing_groups)


@app.route("/training_preview")
def training_preview():
    from training_backend import get_current_labels
    label_list = get_current_labels()
    return render_template("training_preview.html", label_list=label_list)


@app.route("/training_preview_feed")
def training_preview_feed():
    return Response(generate_training_preview_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/training_set_initial_label", methods=["POST"])
def training_set_initial_label():
    chosen_label = request.form.get("chosen_label", "")
    set_current_label(chosen_label)
    return redirect(url_for("training_view_stream"))


@app.route("/training_view_stream")
def training_view_stream():
    from training_backend import get_current_labels
    label_list = get_current_labels()
    return render_template("training_results.html", label_list=label_list)


@app.route("/training_video_feed")
def training_video_feed():
    return Response(
        generate_training_video_stream(auto_finalize=True),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/training_check_status")
def training_check_status():
    return "done" if is_training_done() else "running"


@app.route("/training_get_progress")
def training_get_progress():
    return jsonify(get_training_progress())


@app.route("/training_update_label", methods=["POST"])
def training_update_label():
    new_label = request.form.get("label_name", "")
    set_current_label(new_label)
    return ("", 204)


@app.route("/training_pause", methods=["POST"])
def training_pause():
    from video_utils import toggle_pause_flag
    new_state = toggle_pause_flag()
    return jsonify({"paused": new_state}), 200


@app.route("/training_rewind", methods=["POST"])
def training_rewind():
    from video_utils import schedule_skip
    schedule_skip(-10)
    return ("", 204)


@app.route("/take_a_break", methods=["POST"])
def take_a_break():
    finalize_training_session(do_final_pass=False)
    return redirect(url_for("training_break"))


@app.route("/training_break")
def training_break():
    return render_template("training_break.html")


@app.route("/training_final")
def training_final():
    status = get_training_status()
    return render_template("training_final.html", status=status)


@app.route("/training_get_status")
def training_get_status_api():
    return jsonify(get_training_status())


@app.route("/training_time_info")
def training_time_info():
    from video_utils import get_current_time_sec, get_video_duration
    return jsonify({"current_sec": get_current_time_sec(), "total_sec": get_video_duration()})

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "team_member").strip()

        if not username:
            return render_template("login.html", error="Please enter a username.")

        if password != TEAM_PASSWORD:
            return render_template("login.html", error="Invalid password.")

        session["user"] = username
        session["role"] = role
        return redirect(url_for("home_page"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/db_tools")
@login_required
def db_tools():
    return render_template("db_tools.html")

@app.route("/db_download")
@login_required
def db_download():
    path = _db_path()
    if not os.path.exists(path):
        return "DB file not found.", 404
    return send_file(path, as_attachment=True, download_name="droneai.sqlite")

@app.route("/db_upload", methods=["POST"])
@login_required
def db_upload():
    f = request.files.get("db_file")
    if not f or f.filename == "":
        return render_template("db_tools.html", error="Please choose a .sqlite file to upload.")

    path = _db_path()
    db_dir = os.path.dirname(path)
    os.makedirs(db_dir, exist_ok=True)

    # Backup current DB first
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(db_dir, f"droneai_backup_{stamp}.sqlite")
        try:
            with open(path, "rb") as src, open(backup, "wb") as dst:
                dst.write(src.read())
        except Exception as e:
            return render_template("db_tools.html", error=f"Backup failed: {e}")

    # Replace DB
    try:
        f.save(path)
    except Exception as e:
        return render_template("db_tools.html", error=f"Upload failed: {e}")

    return render_template("db_tools.html", message="Imported DB successfully. You can now continue labeling using the imported labels.")

@app.route("/db_export_excel")
@login_required
def db_export_excel():
    """
    Export all DB tables to one Excel file (basic, useful for grading/reporting).
    """
    import sqlite3
    import pandas as pd

    path = _db_path()
    if not os.path.exists(path):
        return "DB file not found.", 404

    conn = sqlite3.connect(path)

    # read tables safely
    def read_table(name):
        try:
            return pd.read_sql_query(f"SELECT * FROM {name}", conn)
        except Exception:
            return None

    tables = [
        "verifications",
        "validation_sessions",
        "validation_events",
        "training_sessions",
        "training_chunks",
    ]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for t in tables:
            df = read_table(t)
            if df is not None:
                df.to_excel(writer, sheet_name=t[:31], index=False)

    conn.close()
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="droneai_export.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def _mqtt_cfg_view():
    return {
        "enabled": mqtt_mgr.enabled,
        "connected": mqtt_mgr.connected,
        "host": mqtt_mgr.host,
        "port": mqtt_mgr.port,
        "topic_prefix": mqtt_mgr.topic_prefix,
        "username": mqtt_mgr.username,
        "password": mqtt_mgr.password,
    }

@app.route("/mqtt", methods=["GET", "POST"])
@login_required
def mqtt_page():
    if request.method == "POST":
        enabled = True if request.form.get("enabled") == "on" else False
        host = request.form.get("host", "")
        port = request.form.get("port", "1883")
        topic_prefix = request.form.get("topic_prefix", "droneai")
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        mqtt_mgr.configure(enabled, host, int(port), topic_prefix, username, password)

        action = request.form.get("action", "")
        if action == "connect":
            ok, msg = mqtt_mgr.connect()
            return render_template("mqtt.html", cfg=_mqtt_cfg_view(), message=msg, ok=ok)

        if action == "disconnect":
            ok, msg = mqtt_mgr.disconnect()
            return render_template("mqtt.html", cfg=_mqtt_cfg_view(), message=msg, ok=ok)

        return render_template("mqtt.html", cfg=_mqtt_cfg_view(), message="Saved settings.", ok=True)

    return render_template("mqtt.html", cfg=_mqtt_cfg_view())

@app.route("/mqtt_status")
@login_required
def mqtt_status():
    return jsonify({
        "enabled": mqtt_mgr.enabled,
        "connected": mqtt_mgr.connected,
        "events": mqtt_mgr.events[-50:],
        "locks": mqtt_mgr.locks,
    })

@app.route("/")
@login_required
def home_page():
    active, stats = active_dataset_with_stats()
    datasets = db.list_datasets()
    return render_template(
        "home.html",
        username=session.get("user"),
        role=session.get("role"),
        active_dataset=active,
        stats=stats,
        datasets=datasets,
        mqtt_connected=mqtt_mgr.connected,
    )

@app.route("/datasets", methods=["GET", "POST"])
@leader_required
def datasets_page():
    message = None
    error = None

    if request.method == "POST":
        dataset_name = request.form.get("dataset_name", "").strip()
        make_active = True if request.form.get("make_active") == "on" else False
        f = request.files.get("excel_file")

        if not f or f.filename == "":
            error = "Please choose an Excel file."
        else:
            file_bytes = f.read()
            if not dataset_name:
                dataset_name = Path(f.filename).stem

            dataset_key = str(uuid.uuid4())
            db.save_dataset(
                dataset_key=dataset_key,
                dataset_name=dataset_name,
                original_filename=f.filename,
                file_blob=file_bytes,
                uploaded_by=session.get("user", ""),
                is_active=make_active or (db.get_active_dataset() is None),
            )

            items = build_dataset_items_from_excel(file_bytes, dataset_key)
            db.replace_dataset_items(dataset_key, items)

            mqtt_mgr.publish_event("dataset_uploaded", {
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "original_filename": f.filename,
                "uploaded_by": session.get("user", ""),
                "is_active": make_active or (db.get_active_dataset() is None),
                "file_b64": base64.b64encode(file_bytes).decode("utf-8"),
            })

            message = f"Dataset '{dataset_name}' uploaded successfully."

    datasets = db.list_datasets()
    active = db.get_active_dataset()
    return render_template("datasets.html", datasets=datasets, active_dataset=active, message=message, error=error)

@app.route("/datasets/set_active/<dataset_key>", methods=["POST"])
@leader_required
def set_active_dataset(dataset_key):
    db.set_active_dataset(dataset_key)
    mqtt_mgr.publish_event("dataset_set_active", {
        "dataset_key": dataset_key,
        "by": session.get("user", ""),
    })
    return redirect(url_for("datasets_page"))

@app.route("/queue")
@login_required
def queue_page():
    dataset_key = request.args.get("dataset_key")
    if not dataset_key:
        active = db.get_active_dataset()
        if not active:
            return render_template("queue.html", dataset=None, items=[], stats={"total":0,"not_labeled":0,"in_progress":0,"labeled":0})
        dataset_key = active["dataset_key"]

    dataset = db.get_dataset(dataset_key)
    items = db.list_dataset_items(dataset_key)
    stats = db.dataset_stats(dataset_key)

    return render_template(
        "queue.html",
        dataset=dataset,
        items=items,
        stats=stats,
        username=session.get("user"),
        role=session.get("role"),
    )

@app.route("/queue/start/<item_key>", methods=["POST"])
@login_required
def queue_start_item(item_key):
    item = db.get_dataset_item(item_key)
    if not item:
        return "Item not found.", 404

    current_user = session.get("user", "unknown")
    if item["locked_by"] and item["locked_by"] != current_user:
        return f"This video is currently being labeled by {item['locked_by']}.", 400

    scenario_type = request.form.get("scenario_type", "Simulation")
    delete_original = True if request.form.get("delete_original") == "on" else False

    db.update_dataset_item(
        item_key=item_key,
        status="in_progress",
        locked_by=current_user,
        scenario_type=scenario_type,
    )

    mqtt_mgr.publish_event("queue_claimed", {
        "item_key": item_key,
        "dataset_key": item["dataset_key"],
        "by": current_user,
        "scenario_type": scenario_type,
    })
    mqtt_mgr.publish_lock(f"item:{item_key}", current_user, "claimed")

    session["current_item_key"] = item_key
    session["current_dataset_key"] = item["dataset_key"]
    session["current_youtube_link"] = item["youtube_link"]
    session["current_scenario_type"] = scenario_type

    start_validation_thread(
        youtube_link=item["youtube_link"],
        folder_name=None,
        delete_original=delete_original,
        person_name=item["person_name"],
        scenario_base=scenario_type,
    )

    mqtt_mgr.publish_event("validation_started", {
        "by": current_user,
        "youtube_link": item["youtube_link"],
        "person": item["person_name"],
        "scenario": scenario_type,
        "item_key": item_key,
    })

    return redirect(url_for("validation_view_stream", source="dataset"))

@app.route("/validation_release_lock", methods=["POST"])
@login_required
def validation_release_lock():
    current_user = session.get("user", "unknown")
    item_key = session.get("current_item_key")
    scenario_type = session.get("current_scenario_type", "")

    if item_key:
        db.update_dataset_item(
            item_key=item_key,
            status="labeled",
            labeled_by=current_user,
            locked_by="",
            scenario_type=scenario_type,
        )

        mqtt_mgr.publish_event("item_labeled", {
            "item_key": item_key,
            "by": current_user,
            "scenario_type": scenario_type,
        })
        mqtt_mgr.publish_lock(f"item:{item_key}", current_user, "released")

        session.pop("current_item_key", None)
        session.pop("current_dataset_key", None)
        session.pop("current_youtube_link", None)
        session.pop("current_scenario_type", None)

    return jsonify({"ok": True})


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.logger.disabled = True
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

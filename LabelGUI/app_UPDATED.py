from flask import Flask, render_template, request, Response, redirect, url_for, jsonify, flash
import os
import logging
import pandas as pd  # Excel support (pip install pandas openpyxl)
import csv
import uuid
import sys
import subprocess

from validation_backend import (
    start_validation_thread,
    get_crash_count,
    get_log_file_path,
    mark_event_now,
    is_video_done,
    get_extraction_progress,
    toggle_pause,
    skip_video
)

# NEW: crash verification backend (manual)
from crash_verify_backend import (
    start_crash_verify_session,
    generate_crash_verify_stream,
    mark_crash_now,
    get_crash_verify_status,
    finish_crash_verify_session
)

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
    save_label_group_file
)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = "droneai-secret-key"  # needed for flash messages

# ============ In-memory cache of Excel entries ============
EXCEL_ENTRIES = []   # [{"id": 1, "person": "Name", "link": "https://..."}, ...]
LAST_EXCEL_PATH = None  # used so we can write verification results into the same Excel

# ============ Crash Verification Sessions (single-active is OK) ============
CRASH_SESSIONS = {}  # sid -> dict(person, scenario, youtube_link, video_path, predicted, ...)

###############################################################################
#                         HOME PAGE with Buttons                               #
###############################################################################
@app.route('/')
def home_page():
    return render_template('home.html')


###############################################################################
#                        VALIDATION WORKFLOW ROUTES                           #
###############################################################################
@app.route('/validation_index', methods=['GET', 'POST'])
def validation_index():
    if request.method == 'POST':
        youtube_link = request.form.get('youtube_link', '')
        folder_name = request.form.get('folder_name', '')
        delete_original = True if request.form.get('delete_original') == 'on' else False

        start_validation_thread(
            youtube_link=youtube_link,
            folder_name=folder_name,
            delete_original=delete_original
        )
        return redirect(url_for('validation_view_stream'))

    return render_template('validation_index.html')


@app.route('/validation_view_stream')
def validation_view_stream():
    return render_template('validation_results.html')


@app.route('/validation_video_feed')
def validation_video_feed():
    from validation_backend import generate_video_stream
    return Response(
        generate_video_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/mark_event', methods=['POST'])
def mark_event():
    data = request.get_json()
    event_type = data.get('event', 'unknown')
    mark_event_now(event_type)
    return jsonify(success=True, event=event_type)


@app.route('/check_status')
def check_status():
    return "done" if is_video_done() else "running"


@app.route('/get_crash_count')
def get_crash_count_api():
    # returns total events (back-compat name)
    return str(get_crash_count())


@app.route('/get_extraction_progress')
def get_extraction_progress_api():
    return get_extraction_progress()


@app.route('/toggle_pause', methods=['POST'])
def pause_resume():
    paused = toggle_pause()
    return jsonify({"paused": paused}), 200


@app.route('/rewind', methods=['POST'])
def rewind_video():
    skip_video(-10)
    return ('', 204)


@app.route('/fast_forward', methods=['POST'])
def fast_forward_video():
    skip_video(10)
    return ('', 204)


###############################################################################
#                           EXCEL IMPORT WORKFLOW                             #
###############################################################################
@app.route('/import_excel', methods=['GET', 'POST'])
def import_excel():
    """
    GET: Show upload form.
    POST: Save + parse .xlsx into EXCEL_ENTRIES then redirect to picker.
    """
    global EXCEL_ENTRIES, LAST_EXCEL_PATH

    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        file = request.files.get('excel_file')
        if not file or file.filename == '':
            return "Please choose an Excel file (.xlsx).", 400

        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)
        LAST_EXCEL_PATH = save_path  # keep it so we can write verification results back into it

        try:
            df = pd.read_excel(save_path, sheet_name=0)
        except Exception as e:
            return f"Error reading Excel: {e}", 400

        # Expect these columns at minimum:
        if 'Persona Name' not in df.columns or 'Youtube Link' not in df.columns:
            return "Excel must contain 'Persona Name' and 'Youtube Link' columns.", 400

        EXCEL_ENTRIES = []
        next_id = 1
        for _, row in df.iterrows():
            person = str(row.get('Persona Name') or '').strip()
            link = str(row.get('Youtube Link') or '').strip()
            if not person or not link:
                continue
            EXCEL_ENTRIES.append({"id": next_id, "person": person, "link": link})
            next_id += 1

        return redirect(url_for('pick_from_excel'))

    # GET
    return render_template('validation_import.html')


@app.route('/pick_from_excel', methods=['GET'])
def pick_from_excel():
    from validation_backend import get_progress_summary
    global EXCEL_ENTRIES
    progress = get_progress_summary()  # { "JOHN": {"sessions":2,"total_events":5}, ... }

    decorated = []
    for e in EXCEL_ENTRIES:
        prefix = (e["person"] or "").strip()[:4] or "User"
        rec = progress.get(prefix, {"sessions": 0, "total_events": 0})
        decorated.append({
            **e,
            "prefix": prefix,
            "labeled": rec["sessions"] > 0
        })
    return render_template('validation_pick.html', entries=decorated)


@app.route('/start_from_excel', methods=['POST'])
def start_from_excel():
    """
    Use selection to start validation with nested privacy-safe folder structure:
      ValidationResults/<first4_of_name>/<Scenario N>/
    """
    global EXCEL_ENTRIES
    entry_id = int(request.form.get('entry_id', '0'))
    scenario_base = request.form.get('scenario_base', 'Simulation')  # "Simulation" | "Real flight"
    delete_original = True if request.form.get('delete_original') == 'on' else False

    match = next((e for e in EXCEL_ENTRIES if e["id"] == entry_id), None)
    if not match:
        return "Invalid selection.", 400

    start_validation_thread(
        youtube_link=match["link"],
        folder_name=None,
        delete_original=delete_original,
        person_name=match["person"],
        scenario_base=scenario_base
    )
    return redirect(url_for('validation_view_stream'))


###############################################################################
#                       CRASH ANALYSIS + MANUAL VERIFICATION                  #
###############################################################################
def _repo_root():
    # LabelGUI/app.py -> repo root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _analysis_paths():
    root = _repo_root()
    analysis_dir = os.path.join(root, "analysis")
    data_dir = os.path.join(analysis_dir, "data")
    results_dir = os.path.join(analysis_dir, "results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return analysis_dir, data_dir, results_dir

def _read_csv_any(path):
    """Reads csv/tsv safely and returns list[dict]."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except Exception:
            dialect = csv.get_dialect("excel")
        reader = csv.DictReader(f, dialect=dialect)
        return [dict(r) for r in reader]

def _write_single_manifest(path, person, scenario, youtube_link):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["person", "scenario", "youtube_link"])
        w.writerow([person, scenario, youtube_link])

def _latest_per_video_row(per_video_path, person, scenario, youtube_link):
    rows = _read_csv_any(per_video_path)
    # normalize keys
    norm = []
    for r in rows:
        norm.append({(k or "").strip().lower(): (v or "").strip() for k, v in r.items()})
    # find last match
    matches = [
        r for r in norm
        if r.get("person") == (person or "").strip()
        and r.get("scenario") == (scenario or "").strip()
        and r.get("youtube_link") == (youtube_link or "").strip()
    ]
    return matches[-1] if matches else None

def _read_verified_map(verified_csv_path):
    """
    verified.csv columns:
      person,scenario,youtube_link,verified_crash_events,verified_crashes_per_min,verified_times_sec,timestamp,notes
    """
    out = {}
    for r in _read_csv_any(verified_csv_path):
        key = (r.get("person","").strip(), r.get("scenario","").strip(), r.get("youtube_link","").strip())
        out[key] = r
    return out

@app.route('/crash_analysis')
def crash_analysis():
    """
    List videos from analysis/data/manifest.csv and show predicted + verified status.
    """
    analysis_dir, data_dir, results_dir = _analysis_paths()
    manifest_path = os.path.join(data_dir, "manifest.csv")
    per_video_path = os.path.join(results_dir, "per_video.csv")
    verified_path = os.path.join(results_dir, "verified.csv")

    manifest_rows = _read_csv_any(manifest_path)
    # enforce required columns
    items = []
    verified_map = _read_verified_map(verified_path)
    per_video_rows = _read_csv_any(per_video_path)

    # make a quick lookup for predicted
    pred_map = {}
    for r in per_video_rows:
        key = (r.get("person","").strip(), r.get("scenario","").strip(), r.get("youtube_link","").strip())
        pred_map[key] = r

    for r in manifest_rows:
        person = (r.get("person") or r.get("Persona Name") or r.get("name") or "").strip()
        scenario = (r.get("scenario") or r.get("Scenario") or "").strip()
        link = (r.get("youtube_link") or r.get("Youtube Link") or r.get("link") or "").strip()
        if not person or not scenario or not link:
            continue

        key = (person, scenario, link)
        pred = pred_map.get(key, {})
        ver = verified_map.get(key, {})

        items.append({
            "person": person,
            "scenario": scenario,
            "youtube_link": link,
            "pred_crash_events": pred.get("crash_events", ""),
            "pred_crashes_per_min": pred.get("crashes_per_min", ""),
            "verified_crash_events": ver.get("verified_crash_events", ""),
            "verified_crashes_per_min": ver.get("verified_crashes_per_min", ""),
            "verified": True if ver else False,
        })

    return render_template("crash_analysis.html", items=items)


@app.route('/crash_run', methods=['POST'])
def crash_run():
    """
    Run YOLO inference for 1 selected video (uses analysis/scripts/run_pipeline.py),
    then open manual verification page that lets the user click to count crashes.
    """
    person = (request.form.get("person") or "").strip()
    scenario = (request.form.get("scenario") or "").strip()
    youtube_link = (request.form.get("youtube_link") or "").strip()
    sample_fps = float(request.form.get("sample_fps") or 2)
    conf = float(request.form.get("conf") or 0.5)

    if not (person and scenario and youtube_link):
        return "Missing person/scenario/link.", 400

    analysis_dir, data_dir, results_dir = _analysis_paths()
    weights_path = os.path.join(analysis_dir, "weights", "best.pt")
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1024:
        return f"Missing or invalid weights file: {weights_path}", 400

    sid = uuid.uuid4().hex[:10]

    # single-manifest for this run
    single_manifest = os.path.join(data_dir, f"_single_manifest_{sid}.csv")
    _write_single_manifest(single_manifest, person, scenario, youtube_link)

    # run pipeline (downloads + inference + per_video.csv update)
    script_path = os.path.join(analysis_dir, "scripts", "run_pipeline.py")
    if not os.path.exists(script_path):
        return f"Missing script: {script_path}", 400

    downloads_dir = os.path.join(results_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    cmd = [
        sys.executable, script_path,
        "--manifest", single_manifest,
        "--weights", weights_path,
        "--out", results_dir,
        "--downloads", downloads_dir,
        "--sample_fps", str(sample_fps),
        "--conf", str(conf),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return f"Failed to run inference: {e}", 500

    if proc.returncode != 0:
        msg = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return f"Inference failed:\n\n{msg}", 500

    # read predicted row and locate video path
    per_video_path = os.path.join(results_dir, "per_video.csv")
    row = _latest_per_video_row(per_video_path, person, scenario, youtube_link)
    if not row:
        return "Inference finished, but could not find output row in per_video.csv.", 500

    video_path = row.get("video_path", "").strip()
    # make absolute if relative
    if video_path and not os.path.isabs(video_path):
        video_path = os.path.join(results_dir, video_path)
    if not os.path.exists(video_path):
        # try downloads directory fallback
        # (run_pipeline sometimes saves directly under downloads/)
        maybe = os.path.join(downloads_dir, os.path.basename(video_path))
        if os.path.exists(maybe):
            video_path = maybe

    if not video_path or not os.path.exists(video_path):
        return f"Inference succeeded but video file was not found. video_path={row.get('video_path','')}", 500

    # Start verification session (single active is OK)
    start_crash_verify_session(
        sid=sid,
        video_path=video_path,
        person=person,
        scenario=scenario,
        youtube_link=youtube_link,
        predicted_crash_events=row.get("crash_events", ""),
        predicted_crashes_per_min=row.get("crashes_per_min", ""),
    )

    # keep in app memory too (useful for template display)
    CRASH_SESSIONS[sid] = {
        "person": person,
        "scenario": scenario,
        "youtube_link": youtube_link,
        "video_path": video_path,
        "pred_crash_events": row.get("crash_events", ""),
        "pred_crashes_per_min": row.get("crashes_per_min", ""),
    }

    return redirect(url_for("crash_verify", sid=sid))


@app.route('/crash_verify/<sid>')
def crash_verify(sid):
    sess = CRASH_SESSIONS.get(sid, {})
    if not sess:
        return "Invalid or expired session.", 404
    return render_template("crash_verify.html", sid=sid, sess=sess)


@app.route('/crash_video_feed/<sid>')
def crash_video_feed(sid):
    return Response(
        generate_crash_verify_stream(sid),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/crash_mark/<sid>', methods=['POST'])
def crash_mark(sid):
    mark_crash_now(sid)
    return jsonify(success=True)


@app.route('/crash_status/<sid>')
def crash_status(sid):
    return jsonify(get_crash_verify_status(sid))


@app.route('/crash_finish/<sid>', methods=['POST'])
def crash_finish(sid):
    """
    Save verified results to analysis/results/verified.csv and also append to the uploaded Excel file
    under a sheet named 'CrashVerification' (if an Excel was uploaded in this app session).
    """
    global LAST_EXCEL_PATH
    notes = (request.form.get("notes") or "").strip()
    ok, msg = finish_crash_verify_session(sid=sid, excel_path=LAST_EXCEL_PATH, notes=notes)
    return jsonify({"ok": ok, "message": msg})


###############################################################################
#                          TRAINING WORKFLOW ROUTES                           #
###############################################################################
@app.route('/training_index', methods=['GET', 'POST'])
def training_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, 'TrainingResults')
    meta_folder = os.path.join(training_dir, 'MetaDataLocation')
    os.makedirs(meta_folder, exist_ok=True)

    existing_groups = []
    for fname in os.listdir(meta_folder):
        if fname.endswith('.lblgroup'):
            display_name = os.path.splitext(fname)[0]
            existing_groups.append((fname, display_name))

    if request.method == 'POST':
        action = request.form.get('action', 'start')
        youtube_link = request.form.get('youtube_link', '')
        user_name = request.form.get('user_name', '')
        delete_original = True if request.form.get('delete_original') == 'on' else False
        capture_mode = request.form.get('capture_mode', '10fps')
        custom_fps = float(request.form.get('custom_fps', '1') or 1)
        keep_metadata = True if request.form.get('keep_metadata') == 'on' else False

        label_mode = request.form.get('label_mode', '')
        label_group_file = request.form.get('label_group_file', '')

        if action == 'resume':
            metadata_path = os.path.join(meta_folder, 'metadata.csv')
            if os.path.exists(metadata_path):
                resume_training_session(metadata_file=metadata_path)
                return redirect(url_for('training_view_stream'))
            else:
                return "No paused session found. Please start a new session.", 400

        if label_mode == 'load_group':
            full_path = os.path.join(meta_folder, label_group_file)
            labels_and_colors = load_label_group_file(full_path)
            if not labels_and_colors:
                return "Error: Label group file was empty or invalid.", 400

        elif label_mode == 'create_new':
            new_group_name = request.form.get('new_group_name', '').strip()
            if not new_group_name:
                return "Error: You must provide a label group name when creating a new group."

            label_names = request.form.getlist('label_name[]')
            label_colors = request.form.getlist('label_color[]')
            labels_and_colors = list(zip(label_names, label_colors))

            group_filename = f"{new_group_name}.lblgroup"
            full_path = os.path.join(meta_folder, group_filename)
            save_label_group_file(full_path, labels_and_colors)

        else:
            return "Please select an existing label group or create a new one.", 400

        start_training_session(
            youtube_link=youtube_link,
            user_name=user_name,
            delete_original=delete_original,
            capture_mode=capture_mode,
            custom_fps=custom_fps,
            labels_and_colors=labels_and_colors,
            keep_metadata=keep_metadata
        )
        return redirect(url_for('training_preview'))

    # GET request
    return render_template('training_index.html', custom_groups=existing_groups)


@app.route('/training_preview')
def training_preview():
    from training_backend import get_current_labels
    label_list = get_current_labels()
    return render_template('training_preview.html', label_list=label_list)


@app.route('/training_preview_feed')
def training_preview_feed():
    return Response(
        generate_training_preview_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/training_set_initial_label', methods=['POST'])
def training_set_initial_label():
    chosen_label = request.form.get('chosen_label', '')
    set_current_label(chosen_label)
    return redirect(url_for('training_view_stream'))


@app.route('/training_view_stream')
def training_view_stream():
    from training_backend import get_current_labels
    label_list = get_current_labels()
    return render_template('training_results.html', label_list=label_list)


@app.route('/training_video_feed')
def training_video_feed():
    return Response(
        generate_training_video_stream(auto_finalize=True),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/training_check_status')
def training_check_status():
    return "done" if is_training_done() else "running"


@app.route('/training_get_progress')
def training_get_progress():
    return jsonify(get_training_progress())


@app.route('/training_update_label', methods=['POST'])
def training_update_label():
    new_label = request.form.get('label_name', '')
    set_current_label(new_label)
    return ('', 204)


@app.route('/training_pause', methods=['POST'])
def training_pause():
    from video_utils import toggle_pause_flag
    new_state = toggle_pause_flag()
    return jsonify({"paused": new_state}), 200


@app.route('/training_rewind', methods=['POST'])
def training_rewind():
    from video_utils import schedule_skip
    schedule_skip(-10)
    return ('', 204)


@app.route('/take_a_break', methods=['POST'])
def take_a_break():
    finalize_training_session(do_final_pass=False)
    return redirect(url_for('training_break'))


@app.route('/training_break')
def training_break():
    return render_template('training_break.html')


@app.route('/training_final')
def training_final():
    status = get_training_status()
    return render_template('training_final.html', status=status)


@app.route('/training_get_status')
def training_get_status_api():
    return jsonify(get_training_status())


@app.route('/training_time_info')
def training_time_info():
    from video_utils import get_current_time_sec, get_video_duration
    return jsonify({
        'current_sec': get_current_time_sec(),
        'total_sec': get_video_duration()
    })


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.logger.disabled = True
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import os
import logging
import pandas as pd   # <-- NEW

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

# ========= NEW: in-memory store for imported Excel rows =========
EXCEL_ENTRIES = []   # list of dicts: {"id": int, "person": str, "link": str}

@app.route('/')
def home_page():
    return render_template('home.html')


# ===================== NEW: Import Excel workflow =====================
@app.route('/import_excel', methods=['GET', 'POST'])
def import_excel():
    """
    GET: shows an upload form
    POST: accepts .xlsx and parses into EXCEL_ENTRIES
    """
    global EXCEL_ENTRIES

    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        file = request.files.get('excel_file')
        if not file or file.filename == '':
            return "Please choose an Excel file (.xlsx).", 400

        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)

        # Parse Excel
        try:
            df = pd.read_excel(save_path, sheet_name=0)
        except Exception as e:
            return f"Error reading Excel: {e}", 400

        # Expecting columns like:
        # 'Persona Name', 'Youtube Link', 'Simulation Videos', 'Real-flight videos'
        # We'll require just Persona Name + Youtube Link; others optional
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
    """
    Show a list (or dropdown) of parsed entries and let the user choose
    the scenario type: 'Simulation' or 'Real flight'
    """
    global EXCEL_ENTRIES
    return render_template('validation_pick.html', entries=EXCEL_ENTRIES)


@app.route('/start_from_excel', methods=['POST'])
def start_from_excel():
    """
    Start labeling from a chosen Excel row + chosen scenario base.
    """
    global EXCEL_ENTRIES
    entry_id = int(request.form.get('entry_id', '0'))
    scenario_base = request.form.get('scenario_base', 'Simulation')  # Simulation | Real flight

    match = next((e for e in EXCEL_ENTRIES if e["id"] == entry_id), None)
    if not match:
        return "Invalid selection.", 400

    youtube_link = match["link"]
    person_name = match["person"]
    delete_original = True if request.form.get('delete_original') == 'on' else False

    # Use new backend signature to get nested folder: <prefix>/<Scenario N>
    start_validation_thread(
        youtube_link=youtube_link,
        folder_name=None,  # not used when person_name & scenario_base provided
        delete_original=delete_original,
        person_name=person_name,
        scenario_base=scenario_base
    )
    return redirect(url_for('validation_view_stream'))
# ===================== END Import Excel workflow =====================


# ----------------- Existing validation routes (unchanged except mark_event) -----------------
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
    if is_video_done():
        return "done"
    else:
        return "running"


@app.route('/get_crash_count')
def get_crash_count_api():
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


# ----------------- (your training routes remain unchanged) -----------------

if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.logger.disabled = True
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

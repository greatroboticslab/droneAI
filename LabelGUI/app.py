from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
import os
import logging
import pandas as pd  # Excel support (pip install pandas openpyxl)

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

# ============ In-memory cache of Excel entries ============
EXCEL_ENTRIES = []   # [{"id": 1, "person": "Name", "link": "https://..."}, ...]

###############################################################################
#                         HOME PAGE with Two Buttons                          #
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
        return redirect(url_for('validation_view_stream', source='manual'))

    return render_template('validation_index.html')



@app.route('/validation_view_stream')
def validation_view_stream():
    source = request.args.get('source', 'manual')
    return render_template('validation_results.html', source=source)



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
    global EXCEL_ENTRIES

    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    if request.method == 'POST':
        file = request.files.get('excel_file')
        if not file or file.filename == '':
            return "Please choose an Excel file (.xlsx).", 400

        save_path = os.path.join(uploads_dir, file.filename)
        file.save(save_path)

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
    from validation_backend import get_full_progress
    global EXCEL_ENTRIES

    progress_data = get_full_progress()  # full JSON
    people = progress_data.get("people", {})

    decorated = []
    for e in EXCEL_ENTRIES:
        person = e["person"]
        link = e["link"]
        prefix = (person or "").strip()[:4] or "User"

        rec = people.get(prefix, {"sessions": []})
        sessions = rec.get("sessions", [])

        # Only sessions that match THIS specific video link
        matching_sessions = [s for s in sessions if s.get("youtube_link") == link]
        events_for_video = sum(int(s.get("events", 0)) for s in matching_sessions)

        decorated.append({
            **e,
            "prefix": prefix,
            "sessions": len(matching_sessions),      # not shown in table anymore, but kept if needed
            "total_events": events_for_video,        # events for THIS video
            "labeled": events_for_video > 0          # green if any labels exist
        })

    return render_template('validation_pick.html', entries=decorated)




@app.route('/start_from_excel', methods=['POST'])
def start_from_excel():
    from validation_backend import start_validation_thread
    global EXCEL_ENTRIES
    entry_id = int(request.form.get('entry_id', '0'))
    scenario_base = request.form.get('scenario_base', 'Simulation')
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
    return redirect(url_for('validation_view_stream', source='excel'))


    start_validation_thread(
        youtube_link=match["link"],
        folder_name=None,
        delete_original=delete_original,
        person_name=match["person"],
        scenario_base=scenario_base
    )
    return redirect(url_for('validation_view_stream'))


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
    if is_training_done():
        return "done"
    else:
        return "running"


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

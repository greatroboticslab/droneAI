from flask import Flask, render_template, request, Response, redirect, url_for
import os
import logging
from backend import (
    start_processing_thread,
    get_crash_count,
    get_log_file_path,
    set_crash_flag,
    is_video_done
)

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_link = request.form.get('youtube_link', '')
        folder_name = request.form.get('folder_name', '')
        delete_original = True if request.form.get('delete_original') == 'on' else False

        # Start video stream process
        start_processing_thread(
            youtube_link=youtube_link,
            folder_name=folder_name,
            delete_original=delete_original
        )

        return redirect(url_for('view_stream'))

    return render_template('index.html')

@app.route('/view_stream')
def view_stream():
    return render_template('results.html')

@app.route('/video_feed')
def video_feed():
    from backend import generate_video_stream
    return Response(
        generate_video_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/mark_crash', methods=['POST'])
def mark_crash():
    set_crash_flag()
    # Return an empty 204 so there's no pop-up or redirect
    return ('', 204)

@app.route('/check_status')
def check_status():
    if is_video_done():
        return "done"
    else:
        return "running"

# ---------------------------------------------------------------------
# NEW ENDPOINT: returns the current crash count as plain text
# ---------------------------------------------------------------------
@app.route('/get_crash_count')
def get_crash_count_api():
    count = get_crash_count()
    return str(count)
# ---------------------------------------------------------------------

@app.route('/final_results')
def final_results():
    crash_count = get_crash_count()
    log_file_path = get_log_file_path()
    return render_template(
        'final.html',
        crash_count=crash_count,
        log_file_path=log_file_path
    )

if __name__ == '__main__':
    # Suppress default request logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.logger.disabled = True

    app.run(host='0.0.0.0', port=5000, debug=False)

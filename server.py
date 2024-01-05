import cv2
import torch
from flask import Flask, Response, request, render_template, send_from_directory
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.data.augment import LetterBox
from copy import deepcopy

model = YOLO('yolov8n.pt')
app = Flask(__name__, template_folder='./')
cam = None

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print('Release camera')
    cam.release()

def is_in(select_xy, box):
    if select_xy[0] > box[0] and select_xy[0] < box[2]:
        if select_xy[1] > box[1] and select_xy[1] < box[3]:
            return True
    return False

def update(results, deselect_map, select_xy = None):
    if select_xy is None:
        return deselect_map
    
    pred_boxes = results.boxes
    for d in reversed(pred_boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        box = d.xyxy.squeeze()
        if is_in(select_xy, box):
            if id in deselect_map:
                deselect_map.remove(id)
            else:
                deselect_map.add(id)
            break

    return deselect_map


def plot(results, deselect_set):
    names = results.names
    pred_boxes, show_boxes = results.boxes, True
    pred_probs, show_probs = results.probs, True
    annotator = Annotator(
        deepcopy(results.orig_img),
        2,
        2,
        'Arial.ttf',
        (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names)
    
    # Plot Detect results
    if pred_boxes and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            if names[c] != 'person' or id in deselect_set:
                continue
            name = ('' if id is None else f'id:{id} ') + names[c]
            label = (f'{name} {conf:.2f}' if conf else name)
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    return annotator.result()

def capture():
    frame_id = 0
    deselect_map = set()
    select_xy = None
    while True:
        ok, img0=cam.read()
        img0 = cv2.flip(img0, 1)
        if(ok):
            results = model.track(img0, persist=True, tracker="bytetrack.yaml")
            deselect_map = update(results[0], deselect_map, select_xy)
            
            result_frame = plot(results[0], deselect_map)
            ret, jpeg = cv2.imencode('.jpg', result_frame)
            frame =  jpeg.tobytes()

            frame_id += 1
        else:
            cam.release()
            break

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n'
        b'Content-Length: ' + f'{len(frame)}'.encode() + b'\r\n'
        b'\r\n' + frame + b'\r\n')

@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/video_feed')
def video_feed():
    return Response(capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Flask server shutting down...'

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    app.run(host='localhost', port=16034, debug=True)
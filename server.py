import cv2
import torch
from flask import Flask, Response, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.data.augment import LetterBox
from copy import deepcopy
from flask_cors import CORS

cam = None
updated = False
select_xy = None

model = YOLO('yolov8n.pt')
# model.to('cuda')
app = Flask(__name__, template_folder='./')
CORS(app)

def is_in(box):
    if select_xy[0] > box[0] and select_xy[0] < box[2]:
        if select_xy[1] > box[1] and select_xy[1] < box[3]:
            return True
    return False

def update(results, deselect_map):
    if not updated:
        return deselect_map
    
    pred_boxes = results.boxes
    for d in reversed(pred_boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        box = d.xyxy.squeeze()
        if is_in(box):
            print('deselect map is : ', deselect_map)
            if id in deselect_map:
                deselect_map.remove(id)
                print(f'track {id}')
            else:
                deselect_map.add(id)
                print(f'untrack {id}')
            print('deselect map is : ', deselect_map)
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
            if id in deselect_set:
                continue
            name = ('' if id is None else f'id:{id} ') + names[c]
            label = (f'{name} {conf:.2f}' if conf else name)
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    return annotator.result()

def capture():
    global updated
    frame_id = 0
    deselect_map = set()
    while True:
        ok, img0=cam.read()
        img0 = cv2.flip(img0, 1)
        if(ok):
            results = model.track(img0, persist=True, tracker="default.yaml", classes=0, verbose = False)
            deselect_map = update(results[0], deselect_map)
            updated = False
            
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

@app.route('/data', methods=['GET'])
def get_mouse():
    coor=request.args.get("coor")
    coor=coor.split(',') # coor=[x,y,offsetLeft,offsetTop]
    x = coor[0]
    y = coor[1]
    print(f'x={x}, y={y}')
    global updated, select_xy
    updated = True
    select_xy = [float(x), float(y)]
    return jsonify({'message': 'Position received successfully', 'position': f'({x}, {y})'})

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    app.run(host='localhost', port=16034, debug=True)
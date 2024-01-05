import cv2
import torch
from flask import Flask, Response, request, render_template, send_from_directory, jsonify
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.data.augment import LetterBox
from copy import deepcopy
from flask_cors import CORS

model = YOLO('yolov8n.pt')
model.to('cuda')
app = Flask(__name__, template_folder='./')
CORS(app)
cam = None

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    print('Release camera')
    cam.release()

def plot(results, deselect_map):
    if img is None and isinstance(results.orig_img, torch.Tensor):
        img = (results.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = results.names
    pred_boxes, show_boxes = results.boxes, True
    pred_probs, show_probs = results.probs, True
    annotator = Annotator(
        deepcopy(results.orig_img if img is None else img),
        2,
        2,
        'Arial.ttf',
        (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names)
    
    # Plot Detect results
    if pred_boxes and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            if names[c] in deselect_map:
                continue
            name = ('' if id is None else f'id:{id} ') + names[c]
            label = (f'{name} {conf:.2f}' if conf else name)
            annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

    return annotator.result()

def capture():
    frame_id = 0
    while True:
        ok, img0=cam.read()
        if(ok):       
            img0=cv2.flip(img0,1)     
            results = model.track(img0, persist=True, classes=0)
            # result_frame = plot(results[0])
            result_frame = results[0].plot()
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
    x=coor[0]
    y=coor[1]
    print(f'x={x}, y={y}')
    return jsonify({'message': 'Position received successfully'})

@app.route('/shutdown', methods=['GET'])
def shutdown():
    response = jsonify('Flask server shutting down...')
    response.headers.add('Access-Control-Allow-Origin', '*')
    shutdown_server()
    return response

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    app.run(host='localhost', port=16034, debug=True)

import cv2
import numpy as np
import socket
import struct
from io import BytesIO

# Capture frame
cap = cv2.VideoCapture(0)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
client_socket.connect(('140.113.214.61', 8089))

while cap.isOpened():
    _, frame = cap.read()

    memfile = BytesIO()
    np.save(memfile, frame)
    memfile.seek(0)
    data = memfile.read()

    # Send form byte array: frame size + frame content
    client_socket.sendall(struct.pack("I", len(data)) + data)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
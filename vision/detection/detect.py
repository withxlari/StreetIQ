from ultralytics import YOLO 
from collections import defaultdict, deque
import cv2
import math

model = YOLO("yolov8n.pt")

video_path = "videos/traffic_01.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERRO: não foi possível abrir o video em", video_path)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
max_width = 1280
history_length = 15
track_history = defaultdict(lambda: deque(maxlen=history_length))

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

# maioria das mudanças é daq pra baixo

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes  # .boxes objeto q guarda tds as detecções do frame

    if boxes.id is not None:
        for box_id, cls, conf, xyxy in zip(boxes.id, boxes.cls, boxes.conf, boxes.xyxy):
            track_id =  int(box_id)
            class_name = model.names[int(cls)]
            x1, y1, x2, y2 = xyxy.tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            track_history[track_id].append((center_x, center_y))

            if len(track_history[track_id]) >= 2:
                start_x, start_y = track_history[track_id][0]
                end_x, end_y = track_history[track_id][-1]

                dx = end_x - start_x
                dy = end_y - start_y
                distance = math.hypot(dx, dy)

                frames_elapsed = len(track_history[track_id]) - 1
                speed_px_per_sec = (distance / frames_elapsed) * fps

                if abs(dx) > abs(dy):
                    direction = "direita" if dx > 0 else "esquerda"
                else:
                    direction = "baixo" if dy > 0 else "cima"

                print(f"id:{track_id} classe:{class_name} direcao:{direction} velocidade:{speed_px_per_sec:.1f}px/s")

    height, width = annotated_frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        annotated_frame = cv2.resize(annotated_frame, (max_width, int(height * scale)))

    cv2.imshow("StreetIQ - Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
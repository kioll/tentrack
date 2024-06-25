from ultralytics import YOLO

model = YOLO('yolov8m')

result = model.track('input_videos/input_video.mp4', conf =0.2,  save= True)


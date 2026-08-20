import cv2
from ultralytics import YOLO


def annotate_video(video_path, output_path, model_name="yolo26n.pt"):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        video_writer.write(annotated_frame)

    cap.release()
    video_writer.release()

    return output_path


def annotate_pose(video_path, output_path, model_name="yolo26n-pose.pt"):
    return annotate_video(video_path, output_path, model_name)


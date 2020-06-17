
from anomaly_detection import *

alert_mode = bool(sys.argv[1])
def run_local(alert_mode = False):
    s3_client = boto3.client("s3")
    frame_predictor = FramePredictor(s3_client)
    video = cv2.VideoCapture(0)
    frames = [0, 0, 0, 0, 0, 0, 0]
    for idx in range(7):
        ret, frame = video.read()
        if ret == 0:
            break
        frames[idx] = frame
    start_alert_time = time.time()
    while True:
        ret, frame = video.read()
        if ret == 0:
            break
        push_back(frames,frame)

        frame = frames[3]
        frame_d3 = frames[0]
        frame_p3 = frames[6]
        feature_vectors, bounding_boxes = get_feature_vectors_and_bounding_boxes(frame_predictor, frame, frame_d3, frame_p3)
        if feature_vectors.shape[0] > 0:
            feature_vectors = normalize_features(feature_vectors, 0)
        frame_score = 0
        boxes = []
        scores = []
        for idx, vector in enumerate(feature_vectors):
            score = frame_predictor.get_inference_score(vector)
            if score < frame_predictor.threshold:
                scores.append(score)
                c1, l1, c2, l2 = bounding_boxes[idx]
                boxes.append([c1, l1, c2, l2])
                end_alert_time = time.time()
                if end_alert_time - start_alert_time > 5 and alert_mode:
                    show_alert(bounding_boxes[idx],frame)
                    start_alert_time = time.time()

            if score > frame_score:
                frame_score = score
        if not show_boxes(boxes, scores, frame):
            break

run_local(alert_mode)
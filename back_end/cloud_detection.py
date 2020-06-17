from anomaly_detection import *

alert_mode = bool(sys.argv[1])

def run_in_cloud(alert_mode = False):
    addr = 'http://working-env.eu-west-1.elasticbeanstalk.com/'
    test_url = addr + '/upload'
    s3_client = boto3.client("s3")
    video = cv2.VideoCapture(0)
    frames = [0, 0, 0, 0, 0, 0, 0]
    frame_scores = []
    start_alert_time = time.time()
    while True:
        for idx in range(7):
            ret, frame = video.read()
            if ret == 0:
                break
            frames[idx] = frame
        frame = frames[3]
        frame_d3 = frames[0]
        frame_p3 = frames[6]
        frame_key = 'frame' + str(random.randint(0, 150000))
        path = '/tmp/' + frame_key + '.npy'
        path_d3 = '/tmp/' + frame_key + '_d3.npy'
        path_p3 = '/tmp/' + frame_key + 'p3.npy'
        paths_tuples = [(path, frame_key, s3_client), (path_d3, frame_key + '_d3',s3_client), (path_p3, frame_key + '_p3',s3_client)]
        np.save(path, frame)
        np.save(path_d3, frame_d3)
        np.save(path_p3, frame_p3)
        start_uploading = time.time()
        pool = ThreadPool(processes=4)
        pool.map(upload_image, paths_tuples)
        end_uploading = time.time()
        start_http_call = time.time()
        response = requests.post(test_url + "/" + frame_key)
        end_http_call = time.time()
        json = response.json()
        frame_scores.append(json['body'])
        bounding_boxes = json['boxes']
        scores =[]
        boxes = []
        for idx,bounding_box in bounding_boxes:
            if frame_scores[idx] > FramePredictor.threshold:
                scores.append(frame_scores[idx])
                c1, l1, c2, l2 = bounding_boxes[idx]
                boxes.append([c1, l1, c2, l2])
                end_alert_time = time.time()
                if end_alert_time - start_alert_time > 5 and alert_mode:
                    show_alert(bounding_boxes[idx], frame)
                    start_alert_time = time.time()

        if not show_boxes(boxes, scores, frame):
            break

run_in_cloud(alert_mode)
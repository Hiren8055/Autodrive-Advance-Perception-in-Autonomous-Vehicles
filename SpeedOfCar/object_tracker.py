import math
from perspective_transformation import inv_map
from perspective_transformation import get_inv_coor
from localization import get_corrected_top_view


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', r'.\checkpoints\yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    prev_TV_coors = []
    prev_count = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)


        
        
        # print(tuple(zip(bboxes, scores, names, features)))
        detections = []
        for bbox, score, class_name, feature in zip(bboxes, scores, names, features):
            detections.append(Detection(bbox, score, class_name, feature))
            # print()
        print("detect \n",len(detections))
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        
        # print("det")  
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        # setting the array for id
        id_array = []
        track_bbox_array= []
            

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # appending
            id_array.append(track.track_id)
            
            track_bbox_array.append(bbox)

            
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        width_resized,height_resized = 416,416
        frame_resized = cv2.resize(result, (width_resized, height_resized),
                            interpolation=cv2.INTER_LINEAR)
        img_3 = np.zeros([1200,1200,3],dtype=np.uint8)
        img_3.fill(255)
        
        # cv2.imshow("frame_resized",frame_resized)
        TV_frame, M = inv_map(frame_resized)
        
        # CREATE WHITE FRAME AND PLOT IN THE LOWER CENTER OF BB
        _,TV_coors =  get_inv_coor(track_bbox_array)
        TV_coors = get_corrected_top_view(TV_coors)
        print("shape",frame.shape)
        # resized_shape = frame_resized.shape
        # print(resized_shape[1]//2)
        # car = np.array([[(resized_shape[1]//2), (resized_shape[0])]], dtype='float32')
        # car = np.array([car])
        # pointsOut = cv2.perspectiveTransform(car, M)
        # cv2.circle(frame,((resized_shape[1]//2),(resized_shape[0])),10,(0,0,0),-1)
        # TV_car = get_corrected_top_view(pointsOut[0])
        # cv2.circle(TV_frame,(TV_car[0][0],TV_car[0][1],10,(0,0,0),-1))
        # # print("id",id_array)
        # print("points",TV_coors)
        #assumed value of car is 35 kmph
        
        if len(TV_coors):
            prev_TV_coors.append(TV_coors)
            print("prev_TV_coors",prev_TV_coors)
        # for the first frame
        # try:
            if prev_count >= 0:
                print("prev_count",prev_count)
                prev = prev_TV_coors[prev_count-1]
                print("prev",prev)
                sec  = (time.time() - start_time)
                print("hour",sec)
                for i in range(len(prev)):
                    dist = math.sqrt((prev[i][0] - TV_coors[i][0])**2+(prev[i][1] - TV_coors[i][1])**2)
                    cm_sec = dist/sec
                    print("dist",dist)
                    print("cm_sec",cm_sec)
                    car_speed = 35+int(cm_sec/27.778)
                    print("speed in kmh",car_speed)
                    cv2.putText(frame, class_name + "-" + str(car_speed),(int(track_bbox_array[i][0]), int(track_bbox_array[i][1])),0, 0.75, (255,255,255),2)
                    # Add this speed to front view 
            # it will remove previous values
            if prev_count >= 3:
                prev_TV_coors.pop(0)
                prev_count = 2
            else:
                prev_count+=1
        else:
            # prev_TV_coors = []
            print("skipped")

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        width_resized,height_resized = 416,416
        frame_resized = cv2.resize(result, (width_resized, height_resized),
                            interpolation=cv2.INTER_LINEAR)

        cv2.imshow("frame_resized",frame_resized)

        for i in range(len(track_bbox_array)):
            if len(track_bbox_array):
                cv2.circle(img_3,(int(track_bbox_array[i][0]),int(track_bbox_array[i][1])),10,(0,0,0),-1)
                cv2.putText(img_3, str(id_array[i]), (int(track_bbox_array[i][0]),int(track_bbox_array[i][1])-20), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        TV_frame_resized = cv2.resize(img_3,(512,512),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('TV_frame_resized', TV_frame_resized)
        cv2.imshow("TV_frame",TV_frame)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        
        
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

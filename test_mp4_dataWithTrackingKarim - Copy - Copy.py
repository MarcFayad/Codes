

##CAN RUN CODE NORMALLY, BUT IF CAP HAS ARGUMENT FILE_IN, MUST RUN FROM TERMINAL, PUT THE ARGUMENT AS: python test_mp4_data.py -i "path_of_video"
import argparse
from ctypes.wintypes import RGB
from email import parser
import cv2
import numpy as np
import os
#import torch
from oculi.simulation import autoscan, dvs
import sum_abs_diff_function
from RegionClass import region, findRegions2

# detectDeer = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # local model
'''
from keras.models import load_model
# deerClassifier = load_model('MobileNet_model.h5')


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import numpy as np
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobile_net_layers = hub.KerasLayer(mobilenet_v2, input_shape=(224,224,3))
mobile_net_layers.trainable = False

DeerClassifier = tf.keras.Sequential([
  mobile_net_layers,
  tf.keras.layers.Dropout(0.3),
  #tf.keras.layers.Dense(64,activation='relu'),
  tf.keras.layers.Dense(1,activation = 'sigmoid')
])

DeerClassifier.load_weights('MobileNetV2deerWeights.h5')

'''



imageCount = 1

videoName = '20230224_163942'

file_in = "C:\\Users\\User\\Desktop\\OculiScripts\\20230224_163942.mp4"


def RGB_to_gray_disp(color_image_3d):
    frame_gray                                                                      = cv2.cvtColor(color_image_3d,cv2.COLOR_RGB2GRAY)
    three_d_gray                                                                    = np.transpose([frame_gray,frame_gray,frame_gray])
    three_d_gray                                                                    = np.swapaxes(three_d_gray,0,1)
    return(three_d_gray)

def Convert_to_oculi_size(image):
    resized_image                                                                   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print(np.shape(image))
    resized_image                                                                   = cv2.resize(resized_image,(640,360))
    resized_image                                                                   = np.transpose([resized_image,resized_image,resized_image])
    resized_image                                                                   = np.swapaxes(resized_image,0,1)
    return(resized_image)

#TRACKER CODE HERE ALSO
tracker_type                                                                        = 'MOSSE'
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create() 
if tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
bbox_motion                                                                         = None#(170, 200, 180, 140)
bbox_sumAbsDiff                                                                     = None

DVS_TEST                                                                            = False
MOTION_TEST                                                                         = True
COLORS_TEST                                                                         = True
MOTION_AND_COLORS_TEST                                                              = False
MOTION_OR_COLORS_TEST                                                               = False
BACKGROUND_SUBTRACTION_TEST                                                         = False
SUM_ABS_DIFF_TEST                                                                   = False
AGG_FRAMES_TEST                                                                     = False

DOWNSAMPLE_TEST                                                                     = False
CROP_FILE                                                                           = True

DISPLAY                                                                             = True
SAVE                                                                                = False
SAVE_EVENT_OUTPUT                                                                   = False

FIRST_INDEX                                                                         = 1000
LAST_INDEX                                                                          = 1500


DVS_TH                                                                              = 1

MOTION_TLO                                                                          = 15
MOTION_THI                                                                          = -MOTION_TLO

COLORS_TLO                                                                          = 120
COLORS_THI                                                                          = 60

BACK_SUB_TLO                                                                        = 20
BACK_SUB_THI                                                                        = -BACK_SUB_TLO

N                                                                                   = 3
SUM_ABS_DIFF_TH                                                                     = 85


DOWNSAMPLE_VALUE                                                                    = 7
BANDWIDTH_VALUE                                                                     = 10

if MOTION_AND_COLORS_TEST or MOTION_OR_COLORS_TEST:
    MOTION_TRIG                                                                     = True
    COLORS_TRIG                                                                     = True
else:
    MOTION_TRIG                                                                     = False
    COLORS_TRIG                                                                     = False

tests                                                                               = [DVS_TEST, MOTION_TEST, COLORS_TEST, MOTION_AND_COLORS_TEST, MOTION_OR_COLORS_TEST, BACKGROUND_SUBTRACTION_TEST, SUM_ABS_DIFF_TEST]
test_number                                                                         = np.count_nonzero(tests)

cap                                                                                 = cv2.VideoCapture(file_in) #Can change argument from 1,0, or file_in in order to run code with either camera or random video
LAST_INDEX                                                                          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
cols                                                                                = 640 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #or 127
rows                                                                                = 360 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #or 127


frame_rate                                                                          = cap.get(cv2.CAP_PROP_FPS)

cum_frames                                                                          = None
downsample_counter                                                                  = 0
frame_index                                                                         = 0
sum_abs_diff_index                                                                  = 0
bandwidth_counter                                                                   = 0

frames_to_display                                                                   = list()

empty_frame                                                                         = np.zeros((rows + 200, cols, 3), dtype = np.uint8)

full_frame_header                                                                   = np.zeros((100, cols, 3), dtype = np.uint8)
dvs_frame_header                                                                    = np.zeros((100, cols, 3), dtype = np.uint8)
motion_frame_header                                                                 = np.zeros((100, cols, 3), dtype = np.uint8)
colors_frame_header                                                                 = np.zeros((100, cols, 3), dtype = np.uint8)
motion_and_colors_frame_header                                                      = np.zeros((100, cols, 3), dtype = np.uint8)
motion_or_colors_frame_header                                                       = np.zeros((100, cols, 3), dtype = np.uint8)
back_sub_frame_header                                                               = np.zeros((100, cols, 3), dtype = np.uint8)
sum_abs_diff_frame_header                                                           = np.zeros((100, cols, 3), dtype = np.uint8)

cv2.putText(full_frame_header, 'Full Frame', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(dvs_frame_header, 'Events (Polarity)', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(motion_frame_header, 'Smart Events (Motion)', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(colors_frame_header, 'Smart Events (Colors)', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(motion_and_colors_frame_header, 'Motion and Colors', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(motion_or_colors_frame_header, 'Motion or Colors', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(back_sub_frame_header, 'Background Subtraction', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
cv2.putText(sum_abs_diff_frame_header, 'Smart Events 2.0', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)


prev_box = None




if DISPLAY:
    wait                                                                            = 0
    wait_key                                                                        = wait
    cv2.namedWindow('Oculi® BionicVision®', cv2.WINDOW_KEEPRATIO)

if SAVE:
    video_file                                                                      = file_in.split('.')[0] + '_out.mp4'
    src_rate                                                                        = 30

    if test_number <= 3:
        video_cols                                                                  = cols * (test_number + 1)
        video_rows                                                                  = 200 + rows
    else:
        video_cols                                                                  = cols * 4
        video_rows                                                                  = (200 + rows) * 2

    fourcc                                                                          = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer                                                                    = cv2.VideoWriter(video_file, fourcc, src_rate, (video_cols, video_rows), isColor = True)

if SAVE_EVENT_OUTPUT:
    video_file                                                                      = file_in.split('.')[0] + '_out.mp4'
    src_rate                                                                        = 30


    fourcc                                                                          = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer                                                                    = cv2.VideoWriter(video_file, fourcc, src_rate, (cols, rows), isColor = True)


while cap.isOpened():
    ret, frame                                                                      = cap.read()
    ############### IF WANT IN GRAYSCALE UNCOMMENT THIS LINE
    #frame                                                                           = RGB_to_gray_disp(frame)

    frame                                                                           = Convert_to_oculi_size(frame)

    #print(np.shape(frame))
    ##############################################

    if frame is None or (CROP_FILE and frame_index >= LAST_INDEX):
        break

    frame_gray                                                                      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    full_frame_display                                                              = frame.copy()

    full_frame_bandwidth_header                                                     = np.zeros((100, cols, 3), dtype = np.uint8)



    cv2.putText(full_frame_bandwidth_header, 'Bandwidth: {0:.2f}%'.format(100), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
    frames_to_display.append(np.vstack((full_frame_header, full_frame_bandwidth_header, full_frame_display)))

    if frame_index <= FIRST_INDEX:
        prev_frame                                                                  = frame_gray.copy()
    if frame_index <= FIRST_INDEX + 3:
        background                                                                  = frame_gray.copy()
    
    elif (frame_index > FIRST_INDEX) and (not DOWNSAMPLE_TEST or (DOWNSAMPLE_TEST and not downsample_counter)):
        if DVS_TEST:
            dvs_frame, dvs_npix                                                     = dvs.difference(frame_gray, prev_frame, DVS_TH)
            if not bandwidth_counter:
                dvs_bandwidth                                                       = 100 * (dvs_npix / (np.shape(dvs_frame)[0] * np.shape(dvs_frame)[1]))
            dvs_frame_display                                                       = dvs.difference_to_bgr(dvs_frame)
            dvs_bandwidth_header                                                    = np.zeros((100, cols, 3), dtype = np.uint8)
            cv2.putText(dvs_bandwidth_header, '{0:.2f}%'.format(dvs_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
            frames_to_display.append(np.vstack((dvs_frame_header, dvs_bandwidth_header, dvs_frame_display)))
        
        if MOTION_TEST or MOTION_TRIG:
            motion_frame, motion_npix                                               = autoscan.difference(frame_gray, prev_frame, MOTION_TLO, MOTION_THI)

            
            if MOTION_TEST:
                if not bandwidth_counter:
                    motion_bandwidth                                                = 100 * (motion_npix / (np.shape(motion_frame)[0] * np.shape(motion_frame)[1]))
                
                motion_frame_display                                                = np.zeros((rows, cols, 3), dtype = np.uint8)
                
                motion_frame_display[~np.isnan(motion_frame)]                       = frame[~np.isnan(motion_frame)]

                if ret:
                    p1 = (int(bbox_motion[0]), int(bbox_motion[1]))
                    p2 = (int(bbox_motion[0] + bbox_motion[2]), int(bbox_motion[1] + bbox_motion[3]))
                    cv2.rectangle(motion_frame_display, p1, p2, (255,0,0), 2, 1)
                ##################################################################### TRACKER CODE
                if bbox_motion == None:
                    bbox_motion = cv2.selectROI(motion_frame_display,False)
                    prev_box = bbox_motion
                ret = tracker.init(motion_frame_display, bbox_motion)
                ret, bbox_motion = tracker.update(motion_frame_display)
                

                    



                
                motion_bandwidth_header                                             = np.zeros((100, cols, 3), dtype = np.uint8)
                cv2.putText(motion_bandwidth_header, '{0:.2f}%'.format(motion_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)


                 # deers = detectDeer(full_frame_display)
                 
                 # df = deers.pandas().xyxy[0]
                 # # print(df)
                 # if (len(df)> 0):
                 #     for i in range(len(df)):
                 #         row = np.array(df.iloc[i])
                 #         row[0] = int(row[0])
                 #         row[1] = int(row[1])
                 #         row[2] = int(row[2])
                 #         row[3] = int(row[3])
                 #         #frame = cv2.rectangle(frame,(row[0],row[1]),(row[2],row[3]),(255,0,0),2)
                 #         os.chdir("C:\\Users\\marou\\Desktop\\DeerV\\DeerImagesMotionvid23")
                 #         # Filename
                 #         filename = videoName + '['+str(imageCount)+'].jpg'
                 #         cv2.imwrite(filename, motion_frame_display[row[1]:row[3],row[0]:row[2]])
                         
                 #         imageCount = imageCount + 1
                 
                 
    
                     
                     
                     
                # actrow = np.count_nonzero(motion_frame_display[:,:,0], axis = 1)
                # actcol = np.count_nonzero(motion_frame_display[:,:,0], axis = 0)
                 
                # imageWidth = len(actrow)
                # imageLength = len(actcol)
                #Threshold of random pixels triggered in a row or column
                # t1 = 20
                                             
                # reglist = findRegions1(actrow,actcol,t1)
                
                '''
                reglist = findRegions2(cv2.cvtColor(motion_frame_display, cv2.COLOR_BGR2GRAY), 20)
                
                
                mask = np.zeros(motion_frame_display.shape, dtype=np.uint8)             
                if reglist != None:
                    for region1 in reglist:
                        region1.draw(motion_frame_display)
                        
                '''                 

                #if ret:
                frames_to_display.append(np.vstack((motion_frame_header, motion_bandwidth_header, motion_frame_display)))







        if COLORS_TEST or COLORS_TRIG:
            colors_frame, colors_npix                                               = autoscan.intensity(frame_gray, COLORS_TLO, COLORS_THI)

            if COLORS_TEST:
                if not bandwidth_counter:
                    colors_bandwidth                                                = 100 * (colors_npix / (np.shape(colors_frame)[0] * np.shape(colors_frame)[1]))
                colors_frame_display                                                = np.zeros((rows, cols, 3), dtype = np.uint8)
                colors_frame_display[~np.isnan(colors_frame)]                       = frame[~np.isnan(colors_frame)]
                mixed_frame                                                         = np.zeros((rows, cols, 3), dtype = np.uint8)

                if ret:
                        p1 = (int(bbox_motion[0]), int(bbox_motion[1]))
                        p2 = (int(bbox_motion[0] + bbox_motion[2]), int(bbox_motion[1] + bbox_motion[3]))
                        cv2.rectangle(motion_frame_display, p1, p2, (255,0,0), 2, 1)
                        prev_box = bbox_motion
                else:
                        cv2.putText(motion_frame_display, "Tracking failure detected", (20,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                        #Coordinates of bbox_motion
                        bbox_x, bbox_y, bbox_w, bbox_h = [int(v) for v in prev_box]
                        #Cropped color frame
                        cropped_frame = colors_frame_display[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w]
                        #Motion Cropped Frame
                        mixed_frame = motion_frame_display 
                        mixed_frame[bbox_y:bbox_y+bbox_h, bbox_x:bbox_x+bbox_w] = cropped_frame
                        #frames_to_display.append(np.vstack((motion_frame_header, motion_bandwidth_header, mixed_frame)))



                ##################################################################### TRACKER CODE
                # if bbox_motion == None:
                #     bbox_motion = cv2.selectROI(colors_frame_display,False)
                # ret = tracker.init(colors_frame_display, bbox_motion)
                # ret, bbox_motion = tracker.update(colors_frame_display)
                # if ret:
                #         p1 = (int(bbox_motion[0]), int(bbox_motion[1]))
                #         p2 = (int(bbox_motion[0] + bbox_motion[2]), int(bbox_motion[1] + bbox_motion[3]))
                #         cv2.rectangle(colors_frame_display, p1, p2, (255,0,0), 2, 1)
                # else:
                #         cv2.putText(colors_frame_display, "Tracking failure detected", (20,20), 
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                #         #cv2.putText(frame, tracker_type + " Tracker", (100,20), 
                #         #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
                #         #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), 
                #         #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
                #         #cv2.imshow("Tracking", frame)
                ################################################################################# END OF TRACKER CODE
                
                colors_bandwidth_header                                             = np.zeros((100, cols, 3), dtype = np.uint8)
                cv2.putText(colors_bandwidth_header, '{0:.2f}%'.format(colors_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
                frames_to_display.append(np.vstack((colors_frame_header, colors_bandwidth_header, mixed_frame)))

        if MOTION_AND_COLORS_TEST:
            motion_and_colors_frame                                                 = np.zeros((rows, cols, 3), dtype = np.float32)
            motion_and_colors_frame[~np.isnan(motion_frame) 
                                 & (~np.isnan(colors_frame))]                       = frame[~np.isnan(motion_frame) 
                                                                                         & (~np.isnan(colors_frame))]
            if not bandwidth_counter:
                motion_and_colors_bandwidth                                         = 100 * (len(np.argwhere(cv2.cvtColor(motion_and_colors_frame, cv2.COLOR_BGR2GRAY))) / (np.shape(motion_and_colors_frame)[0] * np.shape(motion_and_colors_frame)[1]))
            motion_and_colors_frame_display                                         = motion_and_colors_frame.copy()
            motion_and_colors_frame_display[motion_and_colors_frame == 0]           = np.nan
            motion_and_colors_frame_display                                         = motion_and_colors_frame_display.astype(np.uint8)
            motion_and_colors_bandwidth_header                                      = np.zeros((100, cols, 3), dtype = np.uint8)
            cv2.putText(motion_and_colors_bandwidth_header, '{0:.2f}%'.format(motion_and_colors_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
            frames_to_display.append(np.vstack((motion_and_colors_frame_header, motion_and_colors_bandwidth_header, motion_and_colors_frame_display)))

        if MOTION_OR_COLORS_TEST:
            motion_or_colors_frame                                                  = np.zeros((rows, cols, 3), dtype = np.float32)
            motion_or_colors_frame[~np.isnan(motion_frame) 
                                | (~np.isnan(colors_frame))]                        = frame[~np.isnan(motion_frame) 
                                                                                         | (~np.isnan(colors_frame))]
            if not bandwidth_counter:
                motion_or_colors_bandwidth                                          = 100 * (len(np.argwhere(cv2.cvtColor(motion_or_colors_frame, cv2.COLOR_BGR2GRAY))) / (np.shape(motion_or_colors_frame)[0] * np.shape(motion_or_colors_frame)[1]))
            motion_or_colors_frame_display                                          = motion_or_colors_frame.copy()
            motion_or_colors_frame_display[motion_or_colors_frame == 0]             = np.nan
            motion_or_colors_frame_display                                          = motion_or_colors_frame_display.astype(np.uint8)
            motion_or_colors_bandwidth_header                                       = np.zeros((100, cols, 3), dtype = np.uint8)
            cv2.putText(motion_or_colors_bandwidth_header, '{0:.2f}%'.format(motion_or_colors_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
            frames_to_display.append(np.vstack((motion_or_colors_frame_header, motion_or_colors_bandwidth_header, motion_or_colors_frame_display)))
            
        if BACKGROUND_SUBTRACTION_TEST:
            back_sub_frame, motion_npix                                             = autoscan.difference(frame_gray, background, BACK_SUB_TLO, BACK_SUB_THI)
            if not bandwidth_counter:
                back_sub_bandwidth                                                  = 100 * (motion_npix / (np.shape(back_sub_frame)[0] * np.shape(back_sub_frame)[1]))
            back_sub_frame_display                                                  = np.zeros((rows, cols, 3), dtype = np.uint8)
            back_sub_frame_display[~np.isnan(back_sub_frame)]                       = frame[~np.isnan(back_sub_frame)]
            back_sub_bandwidth_header                                               = np.zeros((100, cols, 3), dtype = np.uint8)
            ########################## For Actionable Signals
            '''
            actionable_signal_rows                                          =    np.count_nonzero(back_sub_frame.astype(np.uint8), axis = 1)
            actionable_signal_cols                                          =    np.count_nonzero(back_sub_frame.astype(np.uint8), axis = 0)
            print(actionable_signal_cols)
            print(actionable_signal_rows)               
            '''
            ##########################
            cv2.putText(back_sub_bandwidth_header, '{0:.2f}%'.format(back_sub_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
            frames_to_display.append(np.vstack((back_sub_frame_header, back_sub_bandwidth_header, back_sub_frame_display)))

        if SUM_ABS_DIFF_TEST:
            result                                                                  = sum_abs_diff_function.sum_abs_diff(frame_gray, int(sum_abs_diff_index) - 2, cum_frames, N, SUM_ABS_DIFF_TH)
            
            cum_frames                                                              = result[0]
            if (int(sum_abs_diff_index) - 2) >= N:       
                cum_frames                                                          = result[0]
                sum_abs_diff_frame                                                  = result[1]
                sum_abs_diff_npix                                                   = result[2]
                


                if not bandwidth_counter:
                    sum_abs_diff_bandwidth                                          = 100 * (sum_abs_diff_npix / (np.shape(sum_abs_diff_frame)[0] * np.shape(sum_abs_diff_frame)[1]))
                
                sum_abs_diff_frame_display                                          = np.zeros((rows, cols, 3), dtype = np.float32)
                sum_abs_diff_frame_display[sum_abs_diff_frame == 0]                 = np.nan
                sum_abs_diff_frame_display[~np.isnan(sum_abs_diff_frame_display)]   = frame[~np.isnan(sum_abs_diff_frame_display)]
                #print(((sum_abs_diff_frame_display)))
                

                sum_abs_diff_frame_display                                          = sum_abs_diff_frame_display.astype(np.uint8)
                #################################################################TRACKER CODE
                if bbox_sumAbsDiff == None:
                    print("NFNDSF")
                    bbox_sumAbsDiff = cv2.selectROI(sum_abs_diff_frame_display,False)
                    bbox_sumAbsDiff = bbox_motion
                ret = tracker.init(sum_abs_diff_frame_display, bbox_motion)
                ret, bbox_sumAbsDiff = tracker.update(sum_abs_diff_frame_display)
                if ret:
                        p1 = (int(bbox_sumAbsDiff[0]), int(bbox_sumAbsDiff[1]))
                        p2 = (int(bbox_sumAbsDiff[0] + bbox_sumAbsDiff[2]), int(bbox_sumAbsDiff[1] + bbox_sumAbsDiff[3]))
                        cv2.rectangle(sum_abs_diff_frame_display, p1, p2, (255,0,0), 2, 1)
                else:
                        cv2.putText(sum_abs_diff_frame_display, "Tracking failure detected", (20,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                        #cv2.putText(frame, tracker_type + " Tracker", (100,20), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
                        #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
                        #cv2.imshow("Tracking", frame)
                ################################################# END OF TRACKER CODE
                sum_abs_diff_bandwidth_header                                       = np.zeros((100, cols, 3), dtype = np.uint8)
                cv2.putText(sum_abs_diff_bandwidth_header, '{0:.2f}%'.format(sum_abs_diff_bandwidth), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (144, 0, 255), thickness = 5)
                
                
                '''
                #THRESHOLD FOR REGIONS
                threshold = sum_abs_diff_npix/180
                print("THRESHOLD = ", threshold)
                reglist = findRegions2(cv2.cvtColor(sum_abs_diff_frame_display, cv2.COLOR_BGR2GRAY), int(threshold))
                
                mask = np.zeros(sum_abs_diff_frame_display.shape, dtype=np.uint8)             
                if reglist != None:
                    for region1 in reglist:
                        region1.draw(sum_abs_diff_frame_display)
                        
                '''
                                 


                frames_to_display.append(np.vstack((sum_abs_diff_frame_header, sum_abs_diff_bandwidth_header, sum_abs_diff_frame_display)))
            else:
                sum_abs_diff_bandwidth_header                                       = np.zeros((100, cols, 3), dtype = np.uint8)
                sum_abs_diff_frame_display                                          = np.zeros((rows, cols, 3), dtype = np.uint8)
            #KARIM COMMENTED THIS OUT
            ############ frames_to_display.append(np.vstack((sum_abs_diff_frame_header, sum_abs_diff_bandwidth_header, sum_abs_diff_frame_display)))
            sum_abs_diff_index                                                     += 1
        



        if not SUM_ABS_DIFF_TEST or (SUM_ABS_DIFF_TEST and ((int(sum_abs_diff_index) - 2) >= N)):
            if len(frames_to_display) == 2:
                output                                                              = np.hstack((
                                                                                        frames_to_display[0], 
                                                                                        frames_to_display[1]
                                                                                    ))
            elif len(frames_to_display) == 3:
                output                                                              = np.hstack((
                                                                                        frames_to_display[0], 
                                                                                        frames_to_display[1], 
                                                                                        frames_to_display[2]
                                                                                    ))
            elif len(frames_to_display) == 4:
                output                                                              = np.hstack((
                                                                                        frames_to_display[0], 
                                                                                        frames_to_display[1], 
                                                                                        frames_to_display[2], 
                                                                                        frames_to_display[3]
                                                                                    ))
            elif len(frames_to_display) == 5:
                output                                                              = np.vstack((
                                                                                        np.hstack((
                                                                                            frames_to_display[0], 
                                                                                            frames_to_display[1], 
                                                                                            frames_to_display[2], 
                                                                                            frames_to_display[3]
                                                                                        )), 
                                                                                        np.hstack((
                                                                                            frames_to_display[4], 
                                                                                            empty_frame, 
                                                                                            empty_frame, 
                                                                                            empty_frame
                                                                                        ))
                                                                                    ))
            elif len(frames_to_display) == 6:
                output                                                              = np.vstack((
                                                                                        np.hstack((
                                                                                            frames_to_display[0], 
                                                                                            frames_to_display[1], 
                                                                                            frames_to_display[2], 
                                                                                            frames_to_display[3]
                                                                                        )), 
                                                                                        np.hstack((
                                                                                            frames_to_display[4], 
                                                                                            frames_to_display[5], 
                                                                                            empty_frame, 
                                                                                            empty_frame
                                                                                        ))
                                                                                    ))
            elif len(frames_to_display) == 7:
                output                                                              = np.vstack((
                                                                                        np.hstack((
                                                                                            frames_to_display[0], 
                                                                                            frames_to_display[1], 
                                                                                            frames_to_display[2], 
                                                                                            frames_to_display[3]
                                                                                        )), 
                                                                                        np.hstack((
                                                                                            frames_to_display[4], 
                                                                                            frames_to_display[5], 
                                                                                            frames_to_display[6], 
                                                                                            empty_frame
                                                                                        ))
                                                                                    ))
            elif len(frames_to_display) == 8:
                output                                                              = np.vstack((
                                                                                        np.hstack((
                                                                                            frames_to_display[0], 
                                                                                            frames_to_display[1], 
                                                                                            frames_to_display[2], 
                                                                                            frames_to_display[3]
                                                                                        )), 
                                                                                        np.hstack((
                                                                                            frames_to_display[4], 
                                                                                            frames_to_display[5], 
                                                                                            frames_to_display[6], 
                                                                                            frames_to_display[7]
                                                                                        ))
                                                                                    ))
            else:
                output                                                              = frames_to_display[0].copy()

            if DISPLAY:
                cv2.imshow('Oculi® BionicVision®', output)

                keyboard_listener                                                   = cv2.waitKey(wait_key) & 0xFF

                if keyboard_listener == 32 and wait_key != 0:
                    wait_key                                                        = 0
                elif keyboard_listener  == 32 and wait_key == 0:
                    wait_key                                                        = wait
                elif keyboard_listener == ord('d'):
                    wait_key                                                        = 0
                elif keyboard_listener == ord('+'):
                    wait_key                                                        = np.floor(wait_key / 2).astype(int)
                    if wait_key <= 0:
                        wait_key                                                    = 1
                elif keyboard_listener == 27:
                    break

            if SAVE:
                video_writer.write(output)
            if SAVE_EVENT_OUTPUT:
                video_writer.write(sum_abs_diff_frame_display)
                
        downsample_counter                                                          = (downsample_counter + 1) if (downsample_counter < DOWNSAMPLE_VALUE) else 0
        bandwidth_counter                                                           = (bandwidth_counter + 1) if ((bandwidth_counter < BANDWIDTH_VALUE) and (not SUM_ABS_DIFF_TEST or (SUM_ABS_DIFF_TEST and ((int(sum_abs_diff_index) - 2) > N)))) else 0

    print('frame index: ', frame_index)
    frame_index                                                                    += 1
    prev_frame                                                                      = frame_gray.copy()
    frames_to_display                                                               = list()

if DISPLAY:
    cv2.destroyAllWindows()

if SAVE:
    video_writer.release()
if SAVE_EVENT_OUTPUT:
    video_writer.release()

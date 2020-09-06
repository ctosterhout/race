import base64
from flask import Flask, render_template
from io import BytesIO
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
import csv, random
from keras.models import load_model
from keras.preprocessing.image import *
import tensorflow as tf
from keras import backend as k
import cv2



'''
run train_lanes, train_steering, and train_signs first.
stick the models they generate in a subfolder called "models"
'''


#global vars
steering_model = load_model( "./models/steering_two.h5")
lane_change_model = load_model( "./models/lane_change.h5")
lane_model = load_model( "./models/lanes.h5")
direction_model = load_model( "./models/direction.h5")
sign_model = load_model( "./models/sign.h5")

sign_detect_time= 0 #the last time we detected a sign
sign_detect_delay = 2 #how long, in seconds, to check for another sign after a detection
sign_check_time = 0 # the last time we check for a sign
sign_check_delay = .05 #how often, in seconds, to check for a sign

changed_lanes_start = -10 #last time we initiated lane change
changed_lanes_delay_steering = 3 #how many seconds to use one lane model while changing lanes
changed_lanes_delay_speed = .75 #how many seconds to slow down

lane_detect_time = -10 #last time we checked the lanes and direction
lane_detect_delay =.2 #how often to check lanes and direction
lane = "left" #the lane we're in
lane_width = "double" #single or double

single_lane_threshold = 3 #number of frames of double lane before switch to double
double_lane_counter= 0 #counter for the single_lane_threshold

direction = "correct" #"correct" or "wrong" the direction we're going
uturn_threshhold = 40 # less than this is executing uturn
uturn_counter = uturn_threshhold #counting up to threshhold
uturn_stage = 0 #stage1 back up, stage2 go forward

stuck_counter = 0 #counts how many frame we've been backing up
try_forward = 0  #counts how many frames we've been going forward
changed_lanes_direction = "none" #right, left, or none. which lane we're changing to

wrong_way_threshold = 10 #number of direction detections before resetting counters
wrong_counter = 0 #counts how many predictions for wrong direction
correct_counter =0 #counts how many predictions for correct direction

race_time=0

sio = socketio.Server()
app = Flask(__name__)



def log_it(log_string):
    global race_time
    print(str(race_time) + " " + log_string)
    try:
        with open("./Log/log.txt", "a+") as f:
            f.write(str(race_time) + " " + log_string + "\n")
    except:
        pass

def process_sign_image(image_bytes):
    image = load_img(image_bytes)
    image = image.crop((0,0,320,90))
    image = img_to_array(image)
    image = cv2.resize(image, (0,0), fx=0.7, fy=0.7)

    r, g, b = cv2.split(image)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 85) & (g < 70) & (b < 220)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    black_filter =  ((r < 40) & (g < 40) & (b < 40))
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    gray_filter = ((np.absolute(r-g) < 40) & (np.absolute(g-b) < 40) & (np.absolute(b-r) < 40)) & ((r>32) & 
        (g>32) & (b> 32)) & ((r<239) & (g<239) & (b<239))  

    b[gray_filter], b[np.invert(gray_filter)] = 0, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0
    g[g_filter], g[np.invert(g_filter)] = 0, 0
    
    
    #sign was too small, just erase it
    sign_proximity = np.count_nonzero(r) 
    #if sign_proximity < 140:
    #    r[r_filter] = 0
    
    
    masked_sign_image = cv2.merge((r, g, b)) 
    masked_sign_image = (masked_sign_image / 255. -.5).astype(np.float32)    
    masked_sign_image = np.rollaxis(masked_sign_image, -1)
    
    return masked_sign_image, sign_proximity

        

def process_lane_image(image_bytes):
    image = load_img(image_bytes)  
    image = img_to_array(image)
    image = cv2.resize(image, (64, 64))
    image = (image / 255. -.5).astype(np.float32)
    image = np.rollaxis(image, -1)
    return image
    
    
def process_direction_image(image_bytes):
    image = load_img(image_bytes)  
    image = image.crop((0,110,320,240))
    image = img_to_array(image)
    image = cv2.resize(image, (160, 65)) 
    
    r, g, b = cv2.split(image)
    white_filter = ((r > 200) & (g > 200) & (b > 200))
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 110) & (g < 70) & (b < 220)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    b[b_filter], b[np.invert(b_filter)] = 0, 0
    r[r_filter], r[np.invert(r_filter)] = 255, 0  
    g[g_filter], g[np.invert(g_filter)] = 0, 0
    image = cv2.merge((r, g, b))
    
    image = (image / 255. -.5).astype(np.float32)
    image = np.rollaxis(image, -1)
    return image
    
    
def process_steer_image(image_bytes, invert=False):
    image = load_img(image_bytes)
    image = image.crop((0,40,320,220)) #(0,120,320,240))
    image = img_to_array(image)
    #image = cv2.resize(image, (0,0), fx=0.35, fy=0.35)
    image = cv2.resize(image, (64, 64))
    if invert: 
        image= flip_axis(image, 1)
        steering_inverted = True
    else:
        steering_inverted= False

    
    image /= 255.
    image -= 0.5
    
    return image, steering_inverted
    
    
    
def process_front_wall_image(image_bytes ,sa):
    image = load_img(image_bytes) #240, 320
        
    #crop((left, top, right, bottom))
    left_clip = 159 + (sa/.25)
    right_clip=160 + (sa/.25)
    
        
    image = image.crop((left_clip,105,right_clip,240))

    image = img_to_array(image)
      
    r, g, b = cv2.split(image)
    r_filter = (r == np.maximum(np.maximum(r, g), b)) & (r >= 120) & (g < 150) & (b < 150)
    g_filter = (g == np.maximum(np.maximum(r, g), b)) & (g >= 120) & (r < 150) & (b < 150)
    b_filter = (b == np.maximum(np.maximum(r, g), b)) & (b >= 120) & (r < 150) & (g < 150)
    black_filter =  ((r < 40) & (g < 40) & (b < 40))
    y_filter = ((r >= 128) & (g >= 128) & (b < 100))
    gray_filter = ((np.absolute(r-g) < 40) & (np.absolute(g-b) < 40) & (np.absolute(b-r) < 40)) & ((r>32) & 
        (g>32) & (b> 32)) & ((r<239) & (g<239) & (b<239)) 
    
    r[ black_filter | gray_filter], r[np.invert( black_filter | gray_filter)] = 255, 0

    wall_proximity = np.count_nonzero(r)

    return wall_proximity

def process_side_wall_image(image):

    
    image = load_img(image)
    image = img_to_array(image)
    left_side = image[:,:1]
    right_side= image[:,319:]
    
    r_r, r_g, r_b = cv2.split(right_side)
    l_r, l_g, l_b = cv2.split(left_side)
    r_black_filter =  ((r_r < 40) & (r_g < 40) & (r_b < 40))
    r_y_filter = ((r_r >= 128) & (r_g >= 128) & (r_b< 100))
    l_black_filter =  ((l_r < 40) & (l_g < 40) & (l_b < 40))
    l_y_filter = ((l_r >= 128) & (l_g >= 128) & (l_b < 100))

    r_r[r_y_filter | r_black_filter ], r_r[np.invert(r_y_filter | r_black_filter )] = 255, 0  
    l_r[l_y_filter | l_black_filter ], l_r[np.invert(l_y_filter | l_black_filter )] = 255, 0

    r_wall_prox = np.count_nonzero(r_r)
    l_wall_prox =np.count_nonzero(l_r)  
    
    return l_wall_prox, r_wall_prox
    
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x   

def get_lane(image_bytes):
    global lane_model, lane, lane_width
    global double_lane_counter, single_lane_threshold
    
    image = process_lane_image(image_bytes)
    #get the lane, left/right
    lane_pred = lane_model.predict(np.array([image]))
    lane_pred_class = np.argmax(lane_pred) #0 left, 1 right
    lane_pred_conf = np.amax(lane_pred)
    if lane_pred_class ==0:
        double_lane_counter+=1
        if lane == "right":
            log_it("Driving in left lane.")
        in_lane="left"
        if double_lane_counter >= single_lane_threshold:
            if lane_width == "single":
                log_it("Driving in double lane.")
            lane_width = "double"
    elif lane_pred_class ==1:
        double_lane_counter+=1
        if lane == "left":
            log_it("Driving in right lane.")
        in_lane = "right"
        if double_lane_counter >= single_lane_threshold:
            if lane_width == "single":
                log_it("Driving in double lane.")
            lane_width = "double"
    elif lane_pred_class == 2:
        double_lane_counter = 0
        if lane_width == "double":
            log_it("Driving in single lane.")
        in_lane="left"
        lane_width = "single"

    if double_lane_counter > single_lane_threshold: double_lane_counter = single_lane_threshold
    return in_lane
    
def get_direction(image_bytes):
    global direction_model
    image = process_direction_image(image_bytes)

    direction_pred = direction_model.predict(np.array([image]))
    direction_pred_class = np.argmax(direction_pred) #0 correct, 1 wrong, 2 unknown
    direction_pred_conf = np.amax(direction_pred)
    if direction_pred_class ==1 and direction_pred_conf >= .9:
        going_direction="wrong"
        log_it("Wrong direction detected.")
    else: going_direction = "correct"
    
    return going_direction
    
def check_signs(image_bytes):
    global race_time
    global sign_detect_time, sign_check_time
    global lane, changed_lanes_direction, changed_lanes_start
    
    sign_image, sign_proximity = process_sign_image(image_bytes)
    sign_check_time = race_time
    if sign_proximity >= 100: #check if there even is a sign before trying to predict what it is
        sign_pred = sign_model.predict(np.array([sign_image]))
        sign_pred_class = np.argmax(sign_pred)
        sign_pred_conf = np.amax(sign_pred)
    
        if sign_pred_conf >= .93: 
            log_it("Sign prediction:" +str(sign_pred_class) + "- Confidence: " + str(sign_pred_conf))

            if sign_pred_class == 1:
                changed_lanes_direction = "right"
                changed_lanes_start = race_time
                log_it("Changing lanes")
                log_it("Going right")  

            elif sign_pred_class == 2:
                changed_lanes_direction = "left"
                changed_lanes_start = race_time
                log_it("Changing lanes")
                log_it("Going left")
            sign_detect_time = race_time


    
        
def change_lanes( image_bytes):
    global changed_lanes_direction
    global lane_change_model
    
    if changed_lanes_direction == "left":
        steering_image, steering_inverted = process_steer_image(image_bytes)
        
    else:# changed_lanes_direction == "right":
        steering_image, steering_inverted = process_steer_image(image_bytes, True)
                
    sa = (lane_change_model.predict(np.array([steering_image]))[0][0]) * 40 #x40 because we normalized between -1 to 1 in training
    if steering_inverted: sa = -sa
    
    return sa  
    
def get_steering_angle(image_bytes):
    global steering_model, lane_width

    steering_image, steering_inverted = process_steer_image(image_bytes)
    if lane_width == "double":
        sa = (steering_model.predict(np.array([steering_image]))[0][0]) * 40 #x40 because we normalized between -1 to 1 in training
    else:
        sa = (lane_change_model.predict(np.array([steering_image]))[0][0]) * 40 #x40 because we normalized between -1 to 1 in training
    if steering_inverted: sa = -sa
        
    return sa

def get_throttle(image_bytes, sa, speed):
    global lane_width
    
    wall_proximity = process_front_wall_image(image_bytes, sa)
    #if wall_proximity < 25 or wall_proximity > 60: wall_proximity = 0 
    if lane_width == "single":
        if wall_proximity < 26 or wall_proximity > 35: wall_proximity = 0
        speed_var = (wall_proximity**1.2) /135
    else:
        if wall_proximity < 30 or wall_proximity > 35: wall_proximity = 0 
        speed_var = (wall_proximity) /135 #135 is max wall infront
    target_speed = max(.3, (1 - (speed_var))*2)
    throttle= 1.2 - (speed / target_speed)

    return throttle
    
    
    
@sio.on('telemetry')
def telemetry(sid, data):
    global race_time
    global changed_lanes_start, changed_lanes_delay_steering
    global lane_detect_time, lane_detect_delay, lane
    global direction, uturn_direction, uturn_counter
    global stuck_counter, try_forward
    global wrong_counter, correct_counter, wrong_way_threshold
    global sign_check_delay, sign_detect_delay, sign_detect_time, sign_check_time

    # The current telemetry
    img_str = data["image"]
    speed = float(data["speed"])
    race_time = float(data["time"])
    sa = float(data["steering_angle"])
    throttle = float(data["throttle"]) 
    current_lap = int(data["lap"])

    # read and process image
    image_bytes = BytesIO(base64.b64decode(img_str))
    
    #find which lane we're in and the direction
    need_to_uturn = False
    if race_time - lane_detect_time > lane_detect_delay:
        lane = get_lane(image_bytes)
        direction = get_direction(image_bytes)
        lane_detect_time = race_time
        if direction == "correct":
            correct_counter +=1
            if correct_counter >= wrong_way_threshold:
                wrong_counter = 0
                correct_counter = 0
        else:
            wrong_counter +=1
            if wrong_counter >=2: #increase this if uturning when going right direction
                log_it("Performing a u-turn.")
                need_to_uturn = True
                wrong_counter = 0
                correct_counter = 0

    
    #check for fork signs
    if (race_time - sign_check_time > sign_check_delay) and (race_time - sign_detect_time > sign_detect_delay ):
        check_signs(image_bytes)
    
                
    #get sa and check if changing lanes
    if (race_time - changed_lanes_start) > changed_lanes_delay_steering:
        sa = get_steering_angle(image_bytes)
    else: #get sa for changing lanes
        sa = change_lanes(image_bytes)
        
    
    #get throttle, slow down for lane change
    if (race_time - changed_lanes_start) <= changed_lanes_delay_speed:
        target_speed = .8
        throttle= 1.2 - (speed / target_speed)
    else:
        throttle = get_throttle(image_bytes, sa, speed)

    #turn around if going wrong way
    if need_to_uturn:
        if lane == "right": uturn_direction = 1
        else: uturn_direction = 2
        uturn_counter = 0
    if uturn_counter < uturn_threshhold:
        uturn_counter +=1
        if uturn_direction == 1: sa = -45
        else: sa = 45
        target_speed = .8
        throttle = 1.2 - (speed / target_speed)
    else: uturn_counter = uturn_threshhold #prevent runaway
    
    #back up if we crash
    try_forward +=1 #count going foward before checking for another crash
    if try_forward > 100: try_forward = 11 #prevent it from going too high
    if speed >= 0 and speed <=.1 and race_time > 5 and stuck_counter < 1 and try_forward > 50:
        stuck_counter = 35 #go backward for this many frames
    if stuck_counter > 0:
        sa = get_steering_angle(image_bytes) #sa = 0
        sa = -sa #opposite sa of going forward, but about half
        target_speed = 1
        throttle = -1.2 + (speed / target_speed) 
        stuck_counter -= 1 #count how long we've been going backward
        try_forward = 0 #reset counter that counts going forward
     
    
    #log_it(sa, throttle)
    send_control(sa, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)



if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

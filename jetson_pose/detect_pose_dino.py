import jetson.inference
import jetson.utils

import numpy as np

from pynput.keyboard import Key, Controller
keyboard = Controller()

import matplotlib.pyplot as plt
import time

# load the pose estimation model. The first time this runs it will 
# Take some time to optimize the model for jetson TRT inference.
net = jetson.inference.poseNet("resnet18-body", argv=["--log-level=error", f"--threshold={.15}"])

# create video sources & outputs
cameraStream = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")

# process frames until the user exits
TRACKED = [0,1,2,3,4]
KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
track_y = []
prev_y = 0
prev_3_deriv = [0,0,20]
prev_3_loc_index = 0
THRESHOLD = 10 #px
while (1):
    # capture the next image
    img = cameraStream.Capture()

    # perform pose estimation (with point overlay)
    poses = net.Process(img, overlay='keypoints')

    #Each pose corresponds to one human detection
    #We only want to process the pose(s) if there's only one human
    if len(poses) > 0:
        pose = poses[0]

        #Lists to track the x and y locations of each point in the "mean face"
        xs = []
        ys = []

        #For each point in mean face, find the keypoint with that name and add it to xs and ys
        '''
        Note: Finding keypoints by getting the name and then finding idx of that name is
        roundabout, but it is protected against index out of bounds errors.
        '''
        for idx in TRACKED:
            name = KEYPOINT_NAMES[idx]
            if pose.FindKeypoint(name) >= 0:
                xs.append(pose.Keypoints[pose.FindKeypoint(name)].x)
                ys.append(pose.Keypoints[pose.FindKeypoint(name)].y)


        center_x = np.mean(xs)
        center_y = np.mean(ys)

        y_deriv = center_y - prev_y
        prev_y = center_y

        #Threshold on the current derivative with the extra condition
        if np.all([i < THRESHOLD for i in prev_3_deriv]) and y_deriv > THRESHOLD:
            print("DUCK")

            #Duck command needs to be held down
            keyboard.press(Key.down)
            time.sleep(0.01)

        elif np.all([i < THRESHOLD for i in prev_3_deriv]) and y_deriv < -1*THRESHOLD:
            print("JUMP")

            #Jump command needs to be pressed and released
            keyboard.release(Key.down)
            keyboard.press(Key.up)
            time.sleep(0.01)
            keyboard.release(Key.up)
        
        #Prev_3_deriv is unordered for tracking last three points.
        #(queue)
        prev_3_deriv[prev_3_loc_index] = np.abs(y_deriv)
        prev_3_loc_index = (prev_3_loc_index + 1)%3

    #Uncomment these lines to use the plotting tracker.
    #print(f"Tracked center = ({center_x},{center_y})")
    #track_y.append(center_y)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("Pose Detection", net.GetNetworkFPS()))

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

#Quick plotting to determine thresholding parameters
'''
plt.figure()
plt.plot(track_y)
plt.title('Position of Head Center')
plt.xlabel('Frames')
plt.ylabel('Position (px)')
plt.savefig('center_height.png')

y_deriv = [track_y[idx] - track_y[idx-1] for idx in range(1, len(track_y))]
plt.figure()
plt.plot(y_deriv)
plt.title('Derivative of Head Center')
plt.xlabel('Frames')
plt.ylabel('Derivative (px)')
plt.savefig('center_deriv.png')
'''

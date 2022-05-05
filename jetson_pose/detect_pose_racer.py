import jetson.inference
import jetson.utils

import numpy as np

#pynput library for simulating keypresses
#Controller() object for emulating keyboard
from pynput.keyboard import Key, Controller
keyboard = Controller()

# load the pose estimation model
net = jetson.inference.poseNet("resnet18-body", argv=["--log-level=error", f"--threshold={.15}"])

# create video sources & outputs
# /dev/video0 for USB Camera
videoStream = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")

#Experimentally tuned cuttoff
TURNING_CUTOFF = .20

# process frames until the user exits
while True:
    # capture the next image
    img = videoStream.Capture()

    # perform pose estimation (with keypoint overlay)
    poses = net.Process(img, overlay='keypoints')

    #Each pose is a human detected.
    for pose in poses:

        #Find the left and right wrist
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        right_wrist_idx = pose.FindKeypoint('right_wrist')

        #Unless one or both wrists dont exist, exit prematurely (do nothing)
        if left_wrist_idx < 0 or right_wrist_idx < 0:
            break
        
        #make a column vector for the positions
        left_wrist = np.array([[pose.Keypoints[left_wrist_idx].x],[pose.Keypoints[left_wrist_idx].y]])
        right_wrist = np.array([[pose.Keypoints[right_wrist_idx].x],[pose.Keypoints[right_wrist_idx].y]])

        #Python standard types version (no numpy)
        #left_wrist = [pose.Keypoints[left_wrist_idx].x, pose.Keypoints[left_wrist_idx].y]
        #right_wrist = [pose.Keypoints[right_wrist_idx].x, pose.Keypoints[right_wrist_idx].y]

        #Find the angle of the connecting line to the horizontal
        wrist_linkage = left_wrist - right_wrist
        linkage_angle = np.arctan2(wrist_linkage[1][0], wrist_linkage[0][0])

        #threshold on the angle
        #This racing game requires press-and-hold
        if linkage_angle > TURNING_CUTOFF:
            print("Turning Left")
            keyboard.release(Key.right)
            keyboard.press(Key.left)

        elif linkage_angle < -1*TURNING_CUTOFF:
            print("Turning Right")
            keyboard.release(Key.left)
            keyboard.press(Key.right)

        else:
            print("Doing Nothing")
            keyboard.release(Key.left)
            keyboard.release(Key.right)

        #Debugging block to find locations of left and right wrist
        #print(f"Left wrist is at {left_wrist.x}, {left_wrist.y}")
        #print(f"Right wrist found at {right_wrist.x}, {right_wrist.y}")
        #print(pose.Keypoints)
        #print(f"Left wrist is at {pose.Keypoints[9]}")
        #print(f"Right wrist is at {pose.Keypoints[10]}")

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("Pose Detection", net.GetNetworkFPS()))

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

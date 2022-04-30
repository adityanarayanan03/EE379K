#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys

import numpy as np

from pynput.keyboard import Key, Controller
keyboard = Controller()

import matplotlib.pyplot as plt
import time

# load the pose estimation model
net = jetson.inference.poseNet("resnet18-body", argv=["--log-level=error", f"--threshold={.15}"])

# create video sources & outputs
input = jetson.utils.videoSource("/dev/video0")
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
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay='keypoints')

    # print the pose results
    # print("detected {:d} objects in image".format(len(poses)))

    #print(poses)

    #Each pose corresponds to one human detection
    if len(poses) > 0:
        pose = poses[0]


        xs = []
        ys = []
        for idx in TRACKED:
            name = KEYPOINT_NAMES[idx]
            if pose.FindKeypoint(name) >= 0:
                xs.append(pose.Keypoints[pose.FindKeypoint(name)].x)
                ys.append(pose.Keypoints[pose.FindKeypoint(name)].y)


        center_x = np.mean(xs)
        center_y = np.mean(ys)

        y_deriv = center_y - prev_y
        prev_y = center_y

        if np.all([i < THRESHOLD for i in prev_3_deriv]) and y_deriv > THRESHOLD:
            print("DUCK")
            keyboard.press(Key.down)
            time.sleep(0.01)
            #keyboard.release(Key.down)
        elif np.all([i < THRESHOLD for i in prev_3_deriv]) and y_deriv < -1*THRESHOLD:
            print("JUMP")
            keyboard.release(Key.down)
            keyboard.press(Key.up)
            time.sleep(0.01)
            keyboard.release(Key.up)
        
        prev_3_deriv[prev_3_loc_index] = np.abs(y_deriv)
        prev_3_loc_index = (prev_3_loc_index + 1)%3


        

        

    #print(f"Tracked center = ({center_x},{center_y})")

    #track_y.append(center_y)

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("Pose Detection", net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

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

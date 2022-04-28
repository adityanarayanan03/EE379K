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

# load the pose estimation model
net = jetson.inference.poseNet("resnet18-body", argv=["--log-level=error", f"--threshold={.15}"])

# create video sources & outputs
input = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay='keypoints')

    # print the pose results
    # print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        left_wrist_idx = pose.FindKeypoint('left_wrist')
        right_wrist_idx = pose.FindKeypoint('right_wrist')

        if left_wrist_idx < 0 or right_wrist_idx < 0:
            break
        
        left_wrist = np.array([[pose.Keypoints[left_wrist_idx].x],[pose.Keypoints[left_wrist_idx].y]])
        #left_wrist = [pose.Keypoints[left_wrist_idx].x, pose.Keypoints[left_wrist_idx].y]
        right_wrist = np.array([[pose.Keypoints[right_wrist_idx].x],[pose.Keypoints[right_wrist_idx].y]])
        #right_wrist = [pose.Keypoints[right_wrist_idx].x, pose.Keypoints[right_wrist_idx].y]

        wrist_linkage = left_wrist - right_wrist
        linkage_angle = np.arctan2(wrist_linkage[1][0], wrist_linkage[0][0])

        TURNING_CUTOFF = .20
        if linkage_angle > TURNING_CUTOFF:
            print("Turning Left")
            keyboard.release(Key.right)
            keyboard.press(Key.left)
            #keyboard.release(Key.left)
        elif linkage_angle < -1*TURNING_CUTOFF:
            print("Turning Right")
            keyboard.release(Key.left)
            keyboard.press(Key.right)
            #keyboard.release(Key.right)
        else:
            print("Doing Nothing")
            keyboard.release(Key.left)
            keyboard.release(Key.right)



        #print(f"Left wrist is at {left_wrist.x}, {left_wrist.y}")
        #print(f"Right wrist found at {right_wrist.x}, {right_wrist.y}")
        #print(pose.Keypoints)
        #print(f"Left wrist is at {pose.Keypoints[9]}")
        #print(f"Right wrist is at {pose.Keypoints[10]}")

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format("Pose Detection", net.GetNetworkFPS()))

    # print out performance info
    #net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break

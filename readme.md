##Human Pose Estimation Based Controller for Video Games

Our project implements simple and intuitive motion controls for video games using two examples: a simple car racing game, and the Google Chrome No-Internet Dinosaur game. 

The system runs on a Jetson Nano, and performs real-time human pose estimation to derive a set of keypoints describing a users arbitrary motions. Based on these keypoints, game-specific classifiers are built in order to map the human's physically-intuitive motions into keypresses to play the game.

Directories and Files:
* jetson_pose: contains all the code intended to run on the jetson nano
* jetson_pose/detect_pose_dino.py and jetson_pose/detect_pose_racer.py: Scripts to run real-time keypoint detection and classify the keypoints into game-specific commands for the dino and racer game(s) respectively. 
* jetson_pose/images: contains a few graphs used to tune thresholding parameters.
* car_game: Forked from https://github.com/ashutoshtiwari13/MyMotoGP, a simple 2-D Car racing game played with left and right arrow keys.
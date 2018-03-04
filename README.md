# HotDogNotHotDog for MarioKart
image classifier to detect a sign in MarioKart 8

## Step 0 - Get the Pix(videoFrames folder)

- https://www.dropbox.com/s/yvkdq5kiait86b9/videoFrames.zip?dl=0

## Step 1 - Generate the lst files:

- `python im2rec.py --list --recursive --num-thread 4 --train-ratio 0.6 --test-ratio 0.2 mk videoFrames`

## Step 2 - Generate the Rec file

- `python im2rec.py --num-thread 4 mk videoFrames`

## Step 3 - Train the Model

- python notHotDog.py

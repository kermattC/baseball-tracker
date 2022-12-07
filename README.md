# baseball-tracker
Course project for CSCI 5520G

For each of these programs they stop frame-by-frame. You can progress to the next frame with any key press.
Instructions on running the code are given for relevant files. Depending on which version of python you use, you'll have to use either python or python3.

## project-pitch3-track.py
* Run via: ``` python3 project-pitch3-track.py ``` r ```python project-pitch3-track.py ```
* My best result. The ball is tracked for majority of its flight before it starts moving downwards, got 5 frames of tracking before it loses tracking, but regains another 2 or 3 frames before losing tracking again. I didn't show this in my presentation because the performance was worse then, but turns out I made a mistake and fixing it yielded better results.

## project-pitch2-track.py
* Run via ``` python3 project-pitch-track.py ``` or ```python project-pitch-track.py ```
* My second best result. The ball is tracked for 8 frames during its downwards movement.

# Other Files

## data
* The videos I used to test this project's method. 
* pitch.mp4: used to grab an image patch for template matching
* pitch2.mp4: Second best result, which is shown in project-pitch2-track.py
* pitch3.mp4: Best result, shown in project-pitch3-track.py
* pitch-fastball.mp4: Wanted to test this method in a different pitch, turns out it performs worse 
* hard-pitch.mp4, hard-pitch2.mp4, hard-pitch3.mp4, pitch-fastball.mp4: I wanted to test this project's method in different scenarios that would make it more difficult to perform. Different camera angles, clutter and occlusion are in these videos. pitch-fastball.mp4 is a different type of pitch that I wanted to see if tracking could be done on, which turned out to not perform very well.

## progress-programs
This file contains programs that I wrote while progressing to the final product. Details are as follows:
###### 1st-pass-template-matching.py
* First iteration of the project, just uses template matching. Originally wrote this as a starting point and just to see how it performs

###### 2nd-pass-gradient-template-matching.py
* Uses template matching after getting the image derivative of each frame. Able to find the ball in some videos. Works way better than the first iteration.

###### binary-threshold.py
* Just an interesting thing I tried to see if I can get something valuable out of it. I wondered if I can get the binary threshold of a frame to filter out things other than the baseball. Since the baseball is white, binary thresholding was able to filter out a lot. But I wasn't sure what to do with it afterwards so I reached a dead end.

###### dense-of.py
* First time using dense optical flow. It just shows the magnitude and direction of pixel movement for each frame. 

###### sparse-optical-flow.py
* Demonstrates why this method of optical flow won't work. Since the baseball has no edges, it can't be detected and tracked.

###### lucas-kanade.py
* Another demonstration of using lucas kanade. Looks prettier though. You can try running it via 
``` python lucas-kanade.py ../data/pitch.mp4 ``` or ``` python3 lucas-kanade.py ../data/pitch.mp4 ```

###### struct-similarity.py
* Uses scikit-image's struct similarity method to track the differences between each frame. Looked promising but I couldn't figure out a way to filter out all the other bounding boxes, so I reached a dead end. Plus the paper I was reading that used this method didn't have any source code available.

###### img-difference.py
* similar to struct-similarity.py I couldn't find a way to filter out all the other bounding boxes, so this is another dead end



## failure-cases
Programs I wrote but couldn't get things working
###### failed-4th-pass-gradient-template-matching.py
* First attempt at applying scale invariant template matching. I couldn't get good results out of it and ended up breaking the code

###### failed-5th-pass-scale-invariance-template-matching.py
* Second attempt at doing scale invariant template matching. I followed another method I found online (more details in the file comments), but couldn't get any good results out of it


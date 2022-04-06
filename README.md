My implementation of mediapipe to get images with hands as an input and to eventualy return a serial version of the hands positions or how clenched they are as an output.

test.py: Tests mediapipe and opencv, drawing various parts of the hand on the video that is created

coords.py: makes an array coordlist with the coordinates of each part of the hand (based on the hand positions below)

clenched.py: makes values thumbclencehd-pinkyclenched which shows how clenched each finger is. Depending on the implementation of the robotic hand, we may use this instead.

TODO:

Output to Serial

![posiitio](https://user-images.githubusercontent.com/92749562/159334727-7f1a1940-1332-46e3-88c4-b05c1d786aa9.jpeg)

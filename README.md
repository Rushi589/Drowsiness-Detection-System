Drowsiness Detection System  

Drowsy driving is a major cause of road accidents, leading to reduced attention and slower reaction times. Fatigue driven accidents can not be avoided using traditional road safety approaches. Hence, there is a need for a smart system that can spot the initial signs of drowsiness.  

Solution:- 

This project presents a real-time Drowsiness Detection System using Computer Vision and Deep Learning. The system tracks two primary variables:  

Yawn Detection: A CNN-LSTM model trained on the Detect-Yawning Dataset. The model classifies frames as Yawn or No Yawn. The CNN captures spatial features and the LSTM identifies temporal features in a sequence of frames. The trained model achieved a 99% accuracy on the test set.  

Eye Aspect Ratio (EAR): EAR is a measure of eye openness. It is determined using facial landmarks (Mediapipe FaceMesh). If EAR is less than 0.25 for 15 consecutive frames, it indicates prolonged eye closure.

       EAR=∣∣p2−p6∣∣+∣∣p3−p5∣∣ / 2×∣∣p1−p4∣∣

where,
p1 and p4 → the horizontal corners of the eye (leftmost and rightmost points).

p2 and p6 → the upper and lower eyelid points (near the middle of the eye).

p3 and p5 → another pair of upper and lower eyelid points (closer to the corners).


Yawn detection or prolonged eye closure triggers a Drowsiness Alert in the system.  

Tech Stack :-

Python, TensorFlow/Keras, OpenCV, Mediapipe, Scikit-learn  

Model - CNN-LSTM for temporal classification + EAR for blink detection  

Dataset-Link- https://www.kaggle.com/datasets/gauravduttakiit/detectyawning

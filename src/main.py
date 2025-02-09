import numpy as np
import mediapipe as mp

print("NumPy version:", np.__version__)
print("MediaPipe version:", mp.__version__)

# Example: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

print("MediaPipe Hands initialized!")

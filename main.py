import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

blur_coefficient = 100

cap = cv2.VideoCapture(0)

detect = cv2.CascadeClassifier('./haar_cascades/frontal_face.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

while cap.isOpened():

    i = 0

    ret, frame = cap.read()

    height = frame.shape[0]
    width = frame.shape[1]
    channels = frame.shape[2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(gray, 1.1, 4)

    height = frame.shape[0]
    width = frame.shape[1]

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Just works with RGB images so we should convert it to RGB
    results = face_mesh.process(RGB)

    # Prevent errors with if statement (or try: catch)
    if results.multi_face_landmarks:
        # Array for convex Fill
        facelandmarks = []

        # All detected landmarks
        for face_landmarks in results.multi_face_landmarks:

            for i in range(0, 468):

                face_landmark = face_landmarks.landmark[i]

                x = int(face_landmark.x * width)
                y = int(face_landmark.y * height)

                facelandmarks.append([x,y])

        np_array_of_landmarks = np.array(facelandmarks, np.int32)

        convexHull = cv2.convexHull(np_array_of_landmarks)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(mask, convexHull, (255,255,255))

        frame_blurred = cv2.blur(frame, (blur_coefficient,blur_coefficient))

        bitwise_and = cv2.bitwise_and(frame_blurred, frame_blurred, mask=mask)

        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        final = cv2.add(bitwise_and, background)

        cv2.imshow('final', final)
    else:
        cv2.putText(frame, 'Cannot detect faces', (10, 10), 3, cv2.FONT_HERSHEY_SIMPLEX, 2, cv2.LINE_AA)
        cv2.imshow('final', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

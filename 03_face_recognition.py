''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import numpy as np
import time
import os.path
import smtplib

recognizer = cv2.face.LBPHFaceRecognizer_create()
yaml_file = os.path.join(os.path.dirname(__file__), 'trainer/trainer.yml')
print(yaml_file)
recognizer.read(yaml_file)
cascadePath = os.path.join(os.path.dirname(__file__), "Cascades/", "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = {
    1: 'Eumie',
    2: 'Andrew'
}

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

authorized_person = False
id_seen = None
time_first_seen = None
not_auth_time = time.time()


def saw_person(id):
    global id_seen, time_first_seen, authorized_person, not_auth_time

    # First time being called
    if not id_seen:
        id_seen = id
        time_first_seen = time.time()
        return

    # If seeing a new person
    if id_seen != id:
        not_auth_time = time.time()
        id_seen = id
        time_first_seen = time.time()
        return


    # If seeing the same person
    if id_seen == id:
        # If 5 seconds has passed
        if time.time() - time_first_seen >= 5:
            authorized_person = (id_seen >= 0)
            return


while not authorized_person:
    if time.time() - not_auth_time >= 8:
        break

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
   )

    cv2.putText(img, str(8 - (time.time() - not_auth_time)), (5, 45), font, 1, (255,255,255), 3)

    faces_found = 0
    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            faces_found += 1
            saw_person(id)
            id = f'{names[id]} Recognized'
            confidence = "  {0}%".format(round(100 - confidence))
            # print('Found user', id)
            # found_person = True
        else:
            id = ""
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    if not faces_found:
        saw_person(-1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

if authorized_person:
    print(f'AUTHORIZED AS: {names[id_seen]}')
else:
    print('UNAUTHORIZED ACCESS. REPORTING')
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("bwibots@gmail.com", "bingbangbong")
    msg = "Subject: Unauthorized Attempt to Log in at BWILab\nDear Dr. Hart, \nThere was an unauthorized login attempt in the BWIbot just now."
    server.sendmail("bwibots@gmail.com", "bwibots@gmail.com", msg)


# Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


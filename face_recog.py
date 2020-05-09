import cv2
import numpy as np
import time
import os.path
import smtplib
import re
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
yaml_file = os.path.join(os.path.dirname(__file__), 'trainer/trainer.yml')
#print(yaml_file)
recognizer.read(yaml_file)
cascadePath = os.path.join(os.path.dirname(__file__), "Cascades/", "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# declare id counter
id = 0

print("\n Enter name to authorize.\n If you want to verify the user exists just press enter")
name_to_authorize = input('\n  ---> YOUR NAME: ')
name_valid = False
face_id_to_authorize = 0

#read all names and store in map
name_file = open('names.txt', 'r')
names = {0: "Unknown"}
lines = name_file.readlines()
for line in lines:
    m = re.search("(\d): (.*)", line.strip())
    names[int(m.group(1))] = m.group(2)
    if re.match(m.group(2), name_to_authorize):
        name_valid = True
        face_id_to_authorize = m.group(1)
name_file.close()

if re.match(name_to_authorize, ""):
    name_valid = True
    face_id_to_authorize = -1

if not name_valid:
    print("\n Name not recognized, terminating...")
    sys.exit()

if len(lines) == 0:
    print("\n No faces trained, terminating...")
    sys.exit()

print()

#counter for number of times we've seen a face
seen_cnt = [0] * (len(lines) + 1)

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# start face recoginition
start_time = 0
done = False
while (True):
    if done:
        break

    ret, img = cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    
    # display 8 second countdown timer
    if start_time != 0:
        if time.time() >= start_time + 8:
            break
        cv2.putText(img, "{:.2f}".format(8 - time.time() + start_time), (5, 45), font, 1.5, (255,255,255), 2)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    faces_found = 0
    for (x,y,w,h) in faces:
        # countdown timer starts when face is first seen
        if start_time == 0:
            start_time = time.time()
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 64):
            faces_found += 1
            
            seen_cnt[id] += 1
            
            # done if we've seen any one face more than 40 times
            if seen_cnt[id] > 40:
                done = True

            id = f'{names[id]}'
            confidence = "  {0}%".format(round(100 - confidence))
            # print('Found user', id)
            # found_person = True
        else:
            seen_cnt[0] += 1
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 1)
        #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()

authorized_person = True

# find the most seen face
most_seen = 0
for i in range(len(seen_cnt)):
    if (seen_cnt[i] > seen_cnt[most_seen]):
        most_seen = i

# not authorized if face is unknown
if most_seen == 0:
    authorized_person = False

# make sure the most seen face is considerably greater than all others
for i in range(len(seen_cnt)):
    if (i != most_seen and seen_cnt[i] > seen_cnt[most_seen] - 10):
        authorized_person = False
        
# make sure we have seen this face enough times for proper verification
if (seen_cnt[most_seen] < 5):
    authorized_person = False

if (face_id_to_authorize != -1):
    # if checking for specific face, make sure the face matches
    if int(face_id_to_authorize) != int(most_seen):
        authorized_person = False

if authorized_person:
    print(f' AUTHORIZED AS: {names[most_seen]}')
else:
    # report unauthorized access
    print(' UNAUTHORIZED ACCESS. REPORTING')
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("bwibots@gmail.com", "bingbangbong")
    msg = "Subject: Unauthorized Attempt to Log in at BWILab\nDear Dr. Hart, \nThere was an unauthorized login attempt in the BWIbot just now."
    server.sendmail("bwibots@gmail.com", "bwibots@gmail.com", msg)


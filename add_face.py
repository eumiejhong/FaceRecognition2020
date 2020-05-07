import cv2
import os
import re
import sys

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
name = input('\n Enter your name: ')
face_id = 0

if re.match(name, ""):
    print(" Invalid name, terminating...")
    sys.exit()

name_exists = False
name_list = []

# create names.txt if file does not exist
if not os.path.exists("./names.txt"):
    name_file = open("names.txt", 'w')
    name_file.close()

# open name file and see if person already exists
name_file = open('names.txt', 'r+')
lines = name_file.readlines()
for line in lines:
    m = re.search('(\d): (.*)', line.strip())
    current_id = m.group(1)
    current_name = m.group(2)
    if (re.match(name, current_name)):
        name_exists = True
        print(" You already have a face registered, we will re-scan your face")
        face_id = current_id
    name_list.append(line)

# generate new id if person doesn't already exist
if (not name_exists):
    face_id = len(lines) + 1
    name_list.append("%d: %s\n" % (face_id, name))

# restore file
name_file.seek(0)
name_file.writelines(name_list)
name_file.close()



print(" Initializing face capture. Look at the camera and wait...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

    cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
cam.release()
cv2.destroyAllWindows()

# Train faces
os.system("python3 02_face_training.py")

import face_recognition
import cv2 
import os
from datetime import datetime

import pickle

from attendance import markAttendance
print(cv2.__version__)

Encodings=[]
#learnings of the Known Faces
Names=[]
#Names of The Known Faces

# Checking if there is a new student(s) present 
while True:
    print("Class Id:202200001234")
    print("Is There Any New Students Added to The Class ?")
    UserInput=input(" If Present , Press 'y', If not 'n' ")
    
    if UserInput=="y":
        image_dir='Image Directory Comes Here'
        # the Folder which has all the Images
        # to Walk through all the files of the Folder(Known)
        for root,dirs,files in os.walk(image_dir):
            print(files)
            for file in files:
                path=os.path.join(root,file) # Joining the root found in the Image directory with The file name\
                #print(path)
                #Getting The name of the person from then file
                name =os.path.splitext(file)[0]
                print(name)
                person=face_recognition.load_image_file(path) # loading the person's Pic to the variable "person"
                encoding=face_recognition.face_encodings(person)[0] # Learning the "Person"'s Face into Variable "encoding"
                Encodings.append(encoding) #appending the learning of the person to the entire list


    elif UserInput=='n':
        break

# User Defined Function For Marking Attendance
def markAttendance(name) :


    with open(Attendance_file_path,'a+') as f:
            myDataList =f.readlines()
            f.writelines(f'\n')
            nameList=[]

            for line in myDataList:
                entry =line.split(",")
                nameList.append(entry[0])
                print(name)

          
            if name not in nameList:
                now=datetime.now()
                dtString=now.strftime('%H:%M:%S ,%m/%d/%Y')
                f.writelines(f'\n')
                f.writelines(f'{name},{dtString}')
                
                 
                      


with open('train.pkl','rb') as f:
        Names=pickle.load(f)
        Encodings=pickle.load(f)   


names=set()
time_stamp=set()
# Path For The CSV File to be Saved
save_path = 'Saved Path Comes Here'
classid="202200001234" # your Class Id Comes Here

now=datetime.now()
dtString=now.strftime('(%d-%m-%Y)')
#concatanating the Class ID, dt String , .csv File Extension
file_name= classid +"_" + dtString +"_" +"Attendance"+"." +"csv"
    
#joining The File Path to that of File name so that it can be saved There.    
completeFileName = os.path.join(save_path, file_name)

Attendance_file_path= completeFileName

print("Check today's Attendance at:")
print(completeFileName)
file1 = open(completeFileName, "x")
 
# Live Camera Attendance Mode                
cam =cv2.VideoCapture(0)
while True:
    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=0.33,fy=0.33)

    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    #locating the Person's Face in the Live camera Feed Frame
    facePositions=face_recognition.face_locations(frameRGB)
     #Encoding the Person's Face in the Live camera Feed Frame
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    
    # for all the faces present in that frame, it is checking for some match
    for(top,right,bottom,left),face_encoding in zip (facePositions,allEncodings):
        name="Unknown Person"
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        
        if True in matches:
            first_match_index = matches.index(True)
            name=Names[first_match_index]
        top=top*3
        bottom=bottom*3
        right=right*3
        left=left*3
        font=cv2.FONT_HERSHEY_SIMPLEX
        # drawing a rectangle over the Face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        # Writing a Name over the Face
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
        now=datetime.now()
        
        time_stamp_student=now.strftime("%H:%M:%S")
        # to avoid Duplicates If The Name is already marked attendance it doesnt mark it twice.
        while name not in names:
            names.add(name)
            if name != "Unknown Person":
                    markAttendance(name)
            

        #time_stamp.add(time_stamp_student)
        cv2.putText(frame,"Person Identified as:",(left,top-25),font,.75,(0,0,255),2)
        current_name=name
     # to Show The Frame Output
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    # to Quit The Frame Output
    if cv2.waitKey(1)==ord('q'):
        break   
cam.release()

cv2.destroyAllWindows()       
        


from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from sklearn.cluster import KMeans

global rec_frame, switch, rec, out,counter,camera,live
switch = 1
rec = 0
counter = 0
live = 0
camera = cv2.VideoCapture(0)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pos_x = []
pos_y = pd.DataFrame()
pos_z = pd.DataFrame()
pos_wrist = []
pos_elbow = []
pos_shoulder = []
pos_nose = []
stage = None


# instatiate flask app
app = Flask(__name__, template_folder='./templates')






def gen_frames():  # generate frame by frame from camera
   global out, rec_frame,live_frame,pos_wrist,pos_shoulder,pos_nose,pos_elbow,counter,test_shoulder,test_nose,test_wrist_1,test_wrist2 
   while True: 
    success, frame = camera.read()
    if success:         
         if (rec):
                print("In Recording Stage")  
                rec_frame = frame
                
                #Setup mediapipe instance
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    
                        # Recolor image to RGB
                        image = cv2.cvtColor(rec_frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        # Make detection
                        results = pose.process(image)

                        # Recolor back to BGR
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Extract landmarks
                        try:
                            print("Recording Pattern")
                            landmarks = results.pose_landmarks.landmark

                            # Get coordinates
                            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]


                                
                            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                            nose=[landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                                     landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                            pos_wrist.append([wrist[0],wrist[1]])
                            pos_elbow.append([elbow[0],elbow[1]])
                            pos_shoulder.append([shoulder[0],shoulder[1]])
                            pos_nose.append([nose[0],nose[1]])
                          
                        except:
                            pass

                      
                        #frame = cv2.flip(frame, 1)
                        frame = cv2.putText(frame, "Recognizing Pattern...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 4)  
                
         if(live):
             live_frame = frame
            
             with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:       

                  # Recolor image to RGB
                  image = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
                  image.flags.writeable = False

                  # Make detection
                  results = pose.process(image)

                  # Recolor back to BGR
                  image.flags.writeable = True
                  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                  # Extract landmarks
                  try:
                      landmarks = results.pose_landmarks.landmark

                      # Get coordinates
                      shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                      elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                      wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                      nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                              landmarks[mp_pose.PoseLandmark.NOSE.value].y]

                      pos_wrist.append([wrist[0], wrist[1]])
                      pos_elbow.append([elbow[0], elbow[1]])
                      pos_shoulder.append([shoulder[0], shoulder[1]])
                      pos_nose.append([nose[0], nose[1]])                 
                     

                     
                      #New Origin
                      test_shoulder=[new_centroid_shoulder[0]-pos_nose[0][0],new_centroid_shoulder[1]-pos_nose[0][1]]
                      test_elbow=[new_centroid_elbow[0] - pos_nose[0][0], new_centroid_elbow[1] - pos_nose[0][1]]

                      #Wrist
                      test_wrist_1=[new_centroid_wrist[0,0]-pos_nose[0][0],new_centroid_wrist[0,1]-pos_nose[0][1]]
                      test_wrist_2=[new_centroid_wrist[1,0]-pos_nose[0][0],new_centroid_wrist[1,1]-pos_nose[0][1]]

                     
                      dist_shoulder=((test_shoulder[0]-shoulder[0])**2+(test_shoulder[1]-shoulder[1])**2)**0.5
                      dist_elbow = ((test_elbow[0] - elbow[0]) ** 2 + (test_elbow[1] - elbow[1]) ** 2) ** 0.5

                      dist_wrist_1 = ((test_wrist_1[0] - wrist[0]) ** 2 + (test_wrist_1[1] - wrist[1]) ** 2) ** 0.5
                      dist_wrist_2 = ((test_wrist_2[0] - wrist[0]) ** 2 + (test_wrist_2[1] - wrist[1]) ** 2) ** 0.5
                      #Logic for Wrist curl without angle calculation
                      if dist_wrist_1<0.25:
                          stage = "down"

                      if dist_wrist_2<0.25 and stage == 'down':
                          stage="Up"
                          counter += 1
                          print(counter)
                  except:
                        pass

             frame = cv2.putText(frame, stage + counter, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

         try:
             ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
             #frame = cv2.flip(frame, 1)
             frame = buffer.tobytes()
             yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
         except Exception as e:
                        pass             
   else:
     pass
            


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera,rec,live
    if request.method == 'POST':

        if request.form.get('End') == 'End':
                camera.release()
                cv2.destroyAllWindows()

        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec
            rec = not rec
            if rec ==1:
                print("Recording")
            elif rec==0 and live==0:
                print("Recognizing Pattern")
                pattern_rec()
        elif request.form.get('Live') == 'Start Exercise' and rec==0: 
            
            live = not live
            if live==1:                      
               print('Live Started')
              
            elif live==0:
               print("Exercise Stopped") 
         
              



    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

def pattern_rec():
    global new_centroid_wrist,new_centroid_nose,new_centroid_elbow,new_centroid_shoulder
    #Finding Centroids of Data
    if np.any(pos_nose) and  np.any(pos_wrist) and np.any(pos_elbow) and np.any(pos_shoulder):

        kmeans=KMeans(n_clusters=1)
        kmeans.fit(pos_nose)
        #pos_wrist['KMean']=kmeans.labels_
        centroids_nose = kmeans.cluster_centers_


        kmeans=KMeans(n_clusters=2)
        kmeans.fit(pos_wrist)
        #pos_wrist['KMean']=kmeans.labels_
        centroids_wrist = kmeans.cluster_centers_


        kmeans=KMeans(n_clusters=2)
        kmeans.fit(pos_elbow)
        #pos_wrist['KMean']=kmeans.labels_
        centroids_elbow = kmeans.cluster_centers_


        kmeans=KMeans(n_clusters=1)
        kmeans.fit(pos_shoulder)
        #pos_wrist['KMean']=kmeans.labels_
        centroids_shoulder = kmeans.cluster_centers_

   
        #Shifting the trained dataset
        new_centroid_wrist=np.zeros([2,2])
        new_centroid_nose=[0,0]
        new_centroid_elbow=[centroids_nose[0,0]+centroids_elbow[0,0],centroids_nose[0,1]+centroids_elbow[0,1]]
        new_centroid_shoulder=[centroids_nose[0,0]+centroids_shoulder[0,0],centroids_nose[0,1]+centroids_shoulder[0,1]]
        #New centroid wrist
        new_centroid_wrist[0,0]=centroids_nose[0,0]+centroids_wrist[0,0]
        new_centroid_wrist[0,1]=centroids_nose[0,1]+centroids_wrist[0,1]
        new_centroid_wrist[1,0]=centroids_nose[0,0]+centroids_wrist[1,0]
        new_centroid_wrist[1,1]=centroids_nose[0,1]+centroids_wrist[1,1]
    
        print('Preprocessed the data')
    else:
        print("No Data")



if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()






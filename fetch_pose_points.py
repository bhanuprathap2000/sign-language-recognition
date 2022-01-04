import cv2
import numpy as np 
import mediapipe as mp 
import pickle


cap = cv2.VideoCapture('Video-1.mp4')
print(cap.get(cv2.CAP_PROP_FPS))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

draw=False

final_data=[]
with mp_holistic.Holistic(static_image_mode=True, model_complexity=2, enable_segmentation=True) as holistic:
	while(cap.isOpened()):
		ret, frame = cap.read()
		image_height, image_width, _ = frame.shape
		if ret == True:
			results = holistic.process(frame)
			if results.pose_landmarks:
				final_data.append(results.pose_landmarks)
				#print(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.THUMB_CMC])
				#print(f'Nose coordinates: ('f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, 'f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})')
				if(draw):
					annotated_image = frame.copy()
					mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
					mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

					cv2.imshow('Output', annotated_image)
					# Press Q on keyboard to  exit
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
		else: 
			break


cap.release()
cv2.destroyAllWindows()

dbfile = open('Video-1_MPHolistic', 'ab')
pickle.dump(final_data, dbfile)                     
dbfile.close()

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def cal_yawn(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))

	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))

	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)

	distance = dist.euclidean(top_mean,low_mean)
	return distance

# cam = cv2.VideoCapture('http://192.168.1.50:4747/video')

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

#-------Models---------#
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor(args["shape_predictor"])

print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
#--------Variables-------#

COUNTER = 0
TOTAL = 0


yawn_thresh = 20
yawn_frame = 5
ptime = 0
while True :
	frame = vs.read()


	#---------FPS------------#
	# ctime = time.time()
	# fps= int(1/(ctime-ptime))
	# ptime = ctime
	# cv2.putText(frame,f'FPS:{fps}',(frame.shape[1]-120,frame.shape[0]-20),cv2.FONT_HERSHEY_PLAIN,2,(0,200,0),3)

	#------Detecting face------#
	img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_model(img_gray)
	for face in faces:
		# #------Uncomment the following lines if you also want to detect the face ----------#
		# x1 = face.left()
		# y1 = face.top()
		# x2 = face.right()
		# y2 = face.bottom()
		# # print(face.top())
		# cv2.rectangle(frame,(x1,y1),(x2,y2),(200,0,00),2)


		#----------Detect Landmarks-----------#
		shapes = landmark_model(img_gray,face)
		shape = face_utils.shape_to_np(shapes)

		#-------Detecting/Marking the lower and upper lip--------#
		lip = shape[48:60]
		cv2.drawContours(frame,[lip],-1,(0, 165, 255),thickness=3)

		#-------Calculating the lip distance-----#
		lip_dist = cal_yawn(shape)
		# print(lip_dist)
		if lip_dist >= yawn_thresh :
			# cv2.putText(frame, f'User Yawning!',(frame.shape[1]//2 - 170 ,frame.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),2)
			COUNTER += 1

		else:
			if COUNTER >= yawn_frame:
				TOTAL += 1

			# reset the eye frame counter
			COUNTER = 0
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Yawn: {}".format(TOTAL), (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "EAR: {:.2f}".format(lip_dist), (300, 30),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


	cv2.imshow('Webcam' , frame)
	if cv2.waitKey(1) & 0xFF == ord('q') :
		break

cam.release()
cv2.destroyAllWindows()
vs.stop()
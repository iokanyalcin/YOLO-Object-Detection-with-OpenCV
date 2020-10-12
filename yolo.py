#import neccesary packages
import numpy as np
import argparse
import time
import cv2
import os

#construct the argumnet parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,
                help="path to input image")
ap.add_argument("-y","--yolo",required=True,
                help="base path to YOLO directory")
ap.add_argument("-c","--confidence",type=float, default=0.5,
                help="minimum probability to filter weak detections.")
ap.add_argument("-t","--threshold",type=float,default=0.3,
                help="threshold when applying non-maxima suppression")

args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"],"coco.names"]) #path.../coco.names
LABELS = open(labelsPath).read().strip().split('\n')
print("Objects can be detect by yolo:")
print(LABELS)
# initialize a list of colors to represent each possible class label
np.random.seed(7)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
#derive paths tot the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


#load our object detector trained on COCO dateset (80 Classes)
print("Loading YOLO Network...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Load input image and get dimensions
image = cv2.imread(args["image"])
(n_H,n_W) = image.shape[:2]

#determine output layer names from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#forward pass
blob = cv2.dnn.blobFromImage(image, 1/255,(416,416),swapRB=True,crop=False) #Convert input image 416x416 and BRG to RGB. Scaling 1/255
net.setInput(blob)

#Measure the time of computation (Forward Pass)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

#intialize bounding_boxes confidencees and classIDs
boxes = list() #Bounding boxes
confidences = list() #Confidences scores, intialized 0.5.
classIDs = list() #Object classes, initialzied 0.3.



#Draw bounding box on image
#ensure that at least one detection exist
# loop over each of the layer outputs
for output in layerOutputs:
	# loop over each of the detections
	for detection in output:
		# extract the class ID and confidence (i.e., probability) of
		# the current object detection
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
			# scale the bounding box coordinates back relative to the
			# size of the image, keeping in mind that YOLO actually
			# returns the center (x, y)-coordinates of the bounding
			# box followed by the boxes' width and height
			box = detection[0:4] * np.array([n_W, n_H, n_W, n_H])
			(centerX, centerY, width, height) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top and
			# and left corner of the bounding box
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			# update our list of bounding box coordinates, confidences,
			# and class IDs
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)


# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)


def control():
    print("Check is completed.")

control()
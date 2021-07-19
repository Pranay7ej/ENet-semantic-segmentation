# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt class labels")
ap.add_argument("-v", "--video", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-s", "--show", type=int, default=1,
	help="whether or not to display frame")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="width (pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")

else:
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# initialize the legend
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)   
cv2.imshow("Legend", legend)

# load our model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])

# initialize the video stream to output video file
vs = cv2.VideoCapture(args["video"])
writer = None

# try to determine the total number of frames in the video file
try:
	prop =  cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
# blobfromimage and enet model
	frame = imutils.resize(frame, width=args["width"])
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	output = net.forward()
	end = time.time()
	(numClasses, height, width) = output.shape[1:4]
	classMap = np.argmax(output[0], axis=0)
	mask = COLORS[classMap]

	mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
		interpolation=cv2.INTER_NEAREST)
	output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")

	# check if the video writer is None
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(output.shape[1], output.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(output)

	# check to see if we should display the output frame to our screen
	if args["show"] > 0:
		cv2.imshow("Frame", output)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# clean by releasing pointers
writer.release()
vs.release()


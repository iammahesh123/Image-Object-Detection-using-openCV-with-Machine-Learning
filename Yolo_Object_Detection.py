import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Colours
colors = []
colors.append((255, 0, 0))
colors.append((0, 0, 255))
colors.append((0, 255, 0))
colors.append((255, 255, 0))
colors.append((255, 0, 255))
colors.append((0, 255, 255))
colors.append((102, 0, 0))
colors.append((105, 105, 105))
colors.append((106, 90, 205))
colors.append((75, 0, 130))
colors.append((0, 0, 128))
colors.append((128, 0, 0))
colors.append((255, 215, 0))
colors.append((255, 69, 0))

# Loading image
img = cv2.imread("test1.jpg")

# if image is test 2
# img = cv2.resize(img, None, fx=0.3, fy=0.3)

height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[np.random.randint(0, len(colors))], 2)
        cv2.putText(img, label, (x, y + 30), font, 2, colors[np.random.randint(0, len(colors))], 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

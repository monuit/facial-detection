import cv2
import numpy as np
import imutils
import time
import argparse
from config import prototxtPath, caffemodelPath, conf, thickness, blue, white, font, meanValues
from imutils.video import VideoStream


class FaceDetection:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(prototxtPath, caffemodelPath)
        
    def drawRectangle(self, image, color, t):
        (x, y, x1, y1) = t
        h = y1 - y
        w = x1 - x
        barLength = int(h / 8)
        cv2.rectangle(image, (x, y-barLength), (x+w, y), color, -1)
        cv2.rectangle(image, (x, y-barLength), (x+w, y), color, thickness)
        cv2.rectangle(image, (x, y), (x1, y1), color, thickness)
        return image

    def changeFontScale(self, h, fontScale):
        fontScale = h/108 * fontScale
        return fontScale
    
    def detectFaces(self, image):
        h, w, _ = image.shape
        resizedImage = cv2.resize(image, (300, 300))
        blob = cv2.dnn.blobFromImage(resizedImage, 1.0, (300, 300), meanValues)

        self.net.setInput(blob)
        faces = self.net.forward()

        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > conf:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                fontScale = self.changeFontScale(y1-y, 0.4)
                image = self.drawRectangle(image, blue, (x, y, x1, y1))
                text = "{:0.2f}%".format(confidence * 100)
                textY = y - 2
                if (textY - 2 < 20): textY = y + 20 
                cv2.putText(image, text, (x, textY), font, fontScale, white, 1)

        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to the image file")
    parser.add_argument("--src", type=int, default=0, help="Source for the webcam. Default is 0.")
    args = parser.parse_args()

    face_detection = FaceDetection()

    if args.image:
        image = cv2.imread(args.image)
        image = face_detection.detectFaces(image)
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
    else:
        vs = VideoStream(src=args.src).start()  # Import now included
        time.sleep(2.0)
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            frame = face_detection.detectFaces(frame)
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) > 0:
                break
        cv2.destroyAllWindows()
        vs.stop()

if __name__ == "__main__":
    main()

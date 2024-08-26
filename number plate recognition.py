import cv2
import os

# Load the Haar Cascade for Russian plate number detection
harcascade = "D:/higher study/project/car number plate recognition/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Initialize video capture for the webcam
cap = cv2.VideoCapture(0)

# Set the width and height of the video feed
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Minimum area for detecting plates
min_area = 500
count = 0

# Ensure the directory exists
if not os.path.exists("plates"):
    os.makedirs("plates")

while True:
    success, img = cap.read()

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect plates in the image
    plates = plate_cascade.detectMultiScale(img_gray, 1.2, 10)

    img_roi = None  # Initialize img_roi to avoid errors if no plates are detected
    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Draw a rectangle around the detected plate
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label the detected area as "Number Plate"
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Extract the region of interest (ROI)
            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)

    # Display the result
    cv2.imshow("Result", img)

    # Break the loop when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and img_roi is not None:
        cv2.imwrite("plates/scaned_img_" + str(count) + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count += 1
    elif key == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

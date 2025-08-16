import cv2
import numpy as np
import time
import os
import random
# -------------------------
# Load Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Filter paths and names
# -------------------------
filter_paths = [
    r"C:/Users/LENOVO/Desktop/AI/C5/mask1.png",
    r"C:/Users/LENOVO/Desktop/AI/C5/mask2.png",
    r"C:/Users/LENOVO/Desktop/AI/C5/mask3.png",
    r"C:/Users/LENOVO/Desktop/AI/C5/crown.png",
]
filter_names = ["Bernoa Mask", "Kabuki Mask", "Carnival Mask"]

# -------------------------
# Load Filters
# -------------------------
filters = []
for path in filter_paths:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    filters.append(img)

# -------------------------
# Helper to overlay filter
# -------------------------
def add_filter(frame, overlay_img, x, y, w, h):
    overlay_img = cv2.resize(overlay_img, (w, h))
    b, g, r, a = cv2.split(overlay_img)
    mask = a / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (1 - mask) * frame[y:y+h, x:x+w, c] + mask * overlay_img[:, :, c]
    return frame

# -------------------------
# Setup camera
# -------------------------
cap = cv2.VideoCapture(0)

current_filter = 0




print("Controls:")
print("N - Next filter")
print("P - Previous filter")
print("S - Save snapshot")
print("Q - Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 7)

    show_adjust_msg = False  # Flag for warning

    for (x, y, w, h) in faces:
        name = filter_names[current_filter]

        if name == "Bernoa Mask":
            fw, fh = w, int(h * 0.5)
            fx, fy = x, y + int(h * 0.15)
        elif name in "Kabuki Mask":
            fw, fh = int(w * 1.2), int(h * 1.5)
            fx, fy = x-int(w*0.09) , y - int(h * 0.45)
        else:
            fw, fh = int(w * 1.05), int(h * 1.3)
            fx, fy = x-int(w*0.01) , y - int(h * 0.2)
        
        # Check if filter goes out of frame
        if fx < 0 or fy < 0 or fx + fw > frame.shape[1] or fy + fh > frame.shape[0]:
            show_adjust_msg = True
        else:
            frame = add_filter(frame, filters[current_filter], fx, fy, fw, fh)


    # Show filter name
    cv2.putText(frame, filter_names[current_filter], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)

    # Show warning if needed
    if show_adjust_msg:
        cv2.putText(frame, "Adjust face position", (50, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
     

    cv2.imshow("Fun Filters", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)
    elif key == ord('p'):
        current_filter = (current_filter - 1) % len(filters)
    elif key == ord('s'):
        img_filename = f"{random.randint(10000,999999)}"+".png"
        cv2.imwrite(img_filename, frame)
        print(f"Snapshot saved: {img_filename}")

cap.release()
cv2.destroyAllWindows()

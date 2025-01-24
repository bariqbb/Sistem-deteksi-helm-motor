import cv2
from ultralytics import YOLO
import pandas as pd
import time
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import os
from datetime import datetime
import requests

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)  
# Setup LCD 
lcd = CharLCD('PCF8574', 0x27)  
lcd.clear()

# Direktori simpan gambar
if not os.path.exists('captured_images'):
    os.makedirs('captured_images')

cam = cv2.VideoCapture(0)
model = YOLO('640_ncnn_model')  # Load model
label = open('label.txt', 'r')
labels = label.read().split('\n')

counter, fps = 0, 0
fps_avg_frame_count = 10
start_time = time.time()
last_capture_time = 0
capture_cooldown = 5  # Cooldown capture
#url = "http://192.168.100.36:8000/ta-gui/public/api/upload"
url = "https://j0g0g2mh-80.asse.devtunnels.ms/ta-gui/public/api/upload"
while True:
    ret, img = cam.read()
    result = model.predict(img, conf=0.5)  # Inference
    a = result[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    detected_label_0 = False
    detected_label_1 = False

    # Hitung FPS
    counter += 1
    if counter % fps_avg_frame_count == 0:
        endtime = time.time()
        fps = fps_avg_frame_count / (endtime - start_time)
        start_time = time.time()
    
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(img, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    # Visualisasi Bounding Box
    for index, row in px.iterrows():
        x1 = int(row[0]) # x1
        y1 = int(row[1]) # y1
        x2 = int(row[2]) # x2
        y2 = int(row[3]) # y2
        conf_score = row[4] # confidence score
        label_index = int(row[5]) # index hasil
        label_name = labels[label_index]

        # Set flag deteksi
        if label_index == 0:
            detected_label_0 = True
        elif label_index == 1:
            detected_label_1 = True

        # label deteksi dan confidence score
        label_text = f'{label_name} {conf_score:.2f}'

        # Dgambar bounding box
        if label_index == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 255, 0), -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y1 - 20), (0, 0, 255), -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Mengatur LCD & LED
    current_time = time.time()
    if detected_label_1 and not detected_label_0:
        GPIO.output(26, GPIO.HIGH)  
        lcd.clear()
        lcd.write_string("Pelanggar")  
        lcd.cursor_pos = (1,0)
        lcd.write_string("Terdeteksi")
        
        # Capture gambar 
        if current_time - last_capture_time > capture_cooldown:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            filename = f"captured_images/violation_{timestamp}.jpg"
            cv2.imwrite(filename, img)
            print(f"Image captured: {filename}")
            path = filename
            with open(path,'rb') as image_file:
                files = {'image' : image_file}
                data = {'timestamp' : timestamp}
                response = requests.post(url, files=files, data=data)
                if response.status_code == 201 : 
                    print ("upload berhasil")
                    print("Response : ", response.json())
                else : 
                    print(f"Upload gagal, status code : {response.status_code}")
                    print("Response : ",response.text)
            last_capture_time = current_time
    else:
        GPIO.output(26, GPIO.LOW)   # Turn off the lamp
        lcd.clear()  # Clear the LCD display
        lcd.write_string("Aman")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup GPIO and LCD
GPIO.cleanup()
lcd.clear()
cam.release()
cv2.destroyAllWindows()

import os
import sys
import pickle
import cv2
import test_config
import test_face
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import time
import select
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

BUTTON_PIN = 17

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

# Ask for email and course code
lcd.clear()
lcd.write_string("Enter email:")
email = input("Enter email to send attendance to: ").strip()

lcd.clear()
lcd.write_string("Enter course:")
course_code = input("Course code for current class: ").strip()

# Load course registration
registered_ids = []
course_dir = test_config.COURSE_LIST
for filename in os.listdir(course_dir):
    if filename.startswith("course."):
        student_id = filename.split('.')[1]
        with open(os.path.join(course_dir, filename), 'rb') as f:
            course_list = pickle.load(f)
            if course_code in course_list:
                registered_ids.append(int(student_id))

# Load model and names
with open('names.p', 'rb') as f:
    names = pickle.load(f)
with open('labels.p', 'rb') as f:
    labels = pickle.load(f)

try:
    model = cv2.face.EigenFaceRecognizer_create()
except:
    model = cv2.face.LBPHFaceRecognizer_create()
model.read(test_config.TRAINING_FILE)

camera = test_config.get_camera()

present_ids = []
present_names = []

lcd.clear()
lcd.write_string("Press btn to")
lcd.cursor_pos = (1, 0)
lcd.write_string("capture face")

try:
    exit_loop = False
    while not exit_loop:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1).strip().lower()
            if key == 'x':
                exit_loop = True
                break

        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            lcd.clear()
            lcd.write_string("Capturing...")
            image = camera.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = test_face.detect_single(gray)

            if result is None:
                lcd.clear()
                lcd.write_string("Face Not Found")
                time.sleep(2)
                lcd.clear()
                lcd.write_string("Press btn to")
                lcd.cursor_pos = (1, 0)
                lcd.write_string("capture face")
                continue

            x, y, w, h = result
            crop_img = test_face.crop(gray, x, y, w, h)
            resized_img = test_face.resize(crop_img)
            label_pred, confidence = model.predict(resized_img)

            lcd.clear()
            if confidence < test_config.POSITIVE_THRESHOLD:
                if label_pred in registered_ids and label_pred not in present_ids:
                    present_ids.append(label_pred)
                    present_names.append(names[labels.index(label_pred)])
                    lcd.write_string("Marked: " + names[labels.index(label_pred)][:16])
                else:
                    lcd.write_string("Already marked")
            else:
                lcd.write_string("Not Recognized")

            time.sleep(2)
            lcd.clear()
            lcd.write_string("Press btn to")
            lcd.cursor_pos = (1, 0)
            lcd.write_string("capture face")

except KeyboardInterrupt:
    pass
finally:
    camera.release()
    lcd.clear()
    GPIO.cleanup()

# Print attendance
attendance_text = f"\nAttendance for course {course_code}:\n"
for sid, sname in zip(present_ids, present_names):
    attendance_text += f"ID: {sid}, Name: {sname}\n"
print(attendance_text)

# Email the attendance
try:
    sender_email = "attendancedesignlabiitp@gmail.com"  # Replace with your Gmail
    sender_password = "zoia yrww hslq wrtq"      # Replace with your app password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = f"Attendance for {course_code}"

    msg.attach(MIMEText(attendance_text, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()
    print(f"Attendance sent to {email}")
except Exception as e:
    print("Failed to send email:", e)

from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os
import json
import tesserocr
from tesserocr import PyTessBaseAPI
import ctypes
from image_processing import capture_window
from socket_connection import create_socket_connection
from PIL import Image
from datetime import datetime
import cv2
from matplotlib import pyplot as plt
import platform
import random
import math
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
from queue import Queue
import select

shot_q = Queue()
putter_in_use = False

class Handler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        length = int(self.headers.get('content-length'))
        if length > 0 and putter_in_use:
            response_code = 200
            message = '{"result" : "OK"}'
            res = json.loads(self.rfile.read(length))
            #print(res)

            putt = {
                "DeviceID": "Rapsodo MLM2PRO",
                "Units": METRIC,
                "ShotNumber": 99,
                "APIversion": "1",
                "ShotDataOptions": {
                    "ContainsBallData": True,
                    "ContainsClubData": True,
                    "LaunchMonitorIsReady": True,
                    "LaunchMonitorBallDetected": True,
                    "IsHeartBeat": False
                }
            }
            putt['BallData'] = {}
            putt['BallData']['Speed'] = float(res['ballData']['BallSpeed'])
            putt['BallData']['TotalSpin'] = float(res['ballData']['TotalSpin'])
            putt['BallData']['SpinAxis'] = 0
            putt['BallData']['HLA'] = float(res['ballData']['LaunchDirection'])
            putt['BallData']['VLA'] = 0
            putt['ClubData'] = {}
            putt['ClubData']['Speed'] = float(res['ballData']['BallSpeed'])
            shot_q.put(putt)

        else:
            if not putter_in_use:
                print("Ignored sporadic putter shot")
            response_code = 500
            message = '{"result" : "ERROR"}'
        self.send_response_only(response_code) # how to quiet this console message?
        self.end_headers()
        #print(json.loads(message))
        self.wfile.write(str.encode(message))


class MyServer(threading.Thread):
    def run(self):
        self.server = ThreadingHTTPServer(('0.0.0.0', 8888), Handler)
        print("starting httpserver")
        self.server.serve_forever()
        print("stopped")
    def stop(self):
        print("stopping httpserver... ", end='')
        self.server.shutdown()

from pygame import mixer
mixer.init()

class Sounds:
    all_dashes=mixer.Sound("deedoo.wav") # Rapsodo range shows all dashes for a 'no-read'
    bad_capture=mixer.Sound("3tone.wav") # One or more data fields was interpreted incorrectly

class TestModes :
    none = 0
    auto_shot = 1 # allows debugging without having to hit shots
    
test_mode = TestModes.none
#test_mode = TestModes.auto_shot

# Set the path to the Tesseract OCR executable and library
tesseract_path = os.path.join(os.getcwd(), 'Tesseract-OCR')
tessdata_path = os.path.join(tesseract_path, 'tessdata')
tesseract_library = os.path.join(tesseract_path, 'libtesseract-5.dll')

# Set the Tesseract OCR path for tesserocr
tesserocr.tesseract_cmd = tessdata_path
ctypes.cdll.LoadLibrary(tesseract_library)

# Loading settings
def load_settings():
    fname = "settings.json"
    if len(sys.argv) > 1 :
        fname = sys.argv[1]
        if os.path.exists(fname):    
            print(f"Using settings from: {fname}")
        else:
            print(f"Can't locate specified settings file: {sys.argv[1]}")
            sys.exit(1)
            
    with open(os.path.join(os.getcwd(), fname), "r") as file:
        lines = file.readlines()
        cleaned_lines = [line.split("//")[0].strip() for line in lines if not line.strip().startswith("//")]
        cleaned_json = "\n".join(cleaned_lines)
        settings = json.loads(cleaned_json)
    return settings

settings = load_settings()
HOST = settings.get("HOST")
PORT = settings.get("PORT")
WINDOW_NAME = settings.get("WINDOW_NAME")
TARGET_WIDTH = settings.get("TARGET_WIDTH")
TARGET_HEIGHT = settings.get("TARGET_HEIGHT")
METRIC = settings.get("METRIC")

rois = []
# Fill rois array from the json.  If ROI1 is present, assume they all are
if settings.get("ROI1") :
    rois.append(list(map(int,settings.get("ROI1").split(','))))
    rois.append(list(map(int,settings.get("ROI2").split(','))))
    rois.append(list(map(int,settings.get("ROI3").split(','))))
    rois.append(list(map(int,settings.get("ROI4").split(','))))
    rois.append(list(map(int,settings.get("ROI5").split(','))))
    rois.append(list(map(int,settings.get("ROI6").split(','))))
    print("Imported ROIs from JSON")
 
if not PORT:
    PORT=921
if not HOST:
    HOST="127.0.0.1"
if not METRIC:
    METRIC="Yards"
    
class Color:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'

def print_colored_prefix(color, prefix, message):
    print(f"{color}{prefix}{Color.RESET}", message)

# Initialize tesseract API once and reuse
api = tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.SINGLE_WORD, lang='train', path=tesserocr.tesseract_cmd)

def select_roi(screenshot):
    plt.imshow(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    print("Please select the region of interest (ROI).")
    roi = plt.ginput(n=2)
    plt.close()
    x1, y1 = map(int, roi[0])
    x2, y2 = map(int, roi[1])
    return (x1, y1, x2 - x1, y2 - y1)

def recognize_roi(screenshot, roi):
    # crop the roi from screenshot
    cropped_img = screenshot[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    # use tesseract to recognize the text
    api.SetImage(Image.fromarray(cropped_img))
    result = api.GetUTF8Text()

    # strip any trailing periods, and keep only one decimal place
    cleaned_result = re.findall(r"[-]?(?:\d*\.*\d)", result)

    if len(cleaned_result) == 0:
        return '-' # didn't find a valid number
    else :
        return cleaned_result[0]

def send_shots():
    global putter_in_use
    try:
        if send_shots.create_socket:
            send_shots.sock = create_socket_connection(HOST, PORT)
            send_shots.create_socket = False

        # Check if we recevied any unsollicited messages from GSPRO (e.g. change of club)
        read_ready, _, _ = select.select([send_shots.sock], [], [], .1)
        while read_ready:
            data = send_shots.sock.recv(1024)
            #print(data)
            jsons = re.split('(\{.*?\})(?= *\{)', data.decode("utf-8"))
            for this_json in jsons:
                if len(this_json) > 0 :
                    msg = json.loads(this_json)
                    if msg['Code'] == 201 :
                        #print(msg)
                        if msg['Player']['Club'] == "PT" and not putter_in_use:
                            print("Putting mode")
                            putter_in_use = True
                        else:
                            if msg['Player']['Club'] != "PT":
                                print(f"Club is {msg['Player']['Club']}")
                                putter_in_use = False
                            
            read_ready, _, _ = select.select([send_shots.sock], [], [], .1)
        # Check if we have a shot to send.  If not, we can return
        try:
            message = shot_q.get_nowait()
        except:
            # No shot to send
            return
        
        message['ShotNumber'] = send_shots.shot_count
        send_shots.sock.sendall(json.dumps(message).encode())
        ball_speed = message['BallData']['Speed']
        total_spin = message['BallData']['TotalSpin']
        spin_axis = message['BallData']['SpinAxis']
        hla= message['BallData']['HLA']
        vla= message['BallData']['VLA']
        club_speed= message['ClubData']['Speed']
         
        print_colored_prefix(Color.GREEN,"MLM2PRO Connector ||", f"Shot {send_shots.shot_count} - Ball Speed: {ball_speed} MPH, Total Spin: {total_spin} RPM, Spin Axis: {spin_axis}°, HLA: {hla}°, VLA: {vla}°, Club Speed: {club_speed} MPH")
        send_shots.shot_count += 1

        # Poll politely until there is a message received on the socket
        #print(message)
        read_ready = False
        while not read_ready:
            read_ready, _, _ = select.select([send_shots.sock], [], [], .5)
                
        # Here we know the response is due to the shot we sent
        data = send_shots.sock.recv(1024) # Note, we know there's a response now
        #print(data)
        if len(data) > 0 and json.loads(data)['Code'] == 200:
            print_colored_prefix(Color.BLUE, "MLM2PRO Connector ||", "Shot data has been sent successfully...")
            send_shots.gspro_connection_notified = False;
            send_shots.create_socket = False

    except Exception as e:
        print(e)
        print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "No response from GSPRO. Retrying")
        if not send_shots.gspro_connection_notified:
            Sounds.all_dashes.play()
            send_shots.gspro_connection_notified = True;
        send_shots.create_socket = True

    return
# Initialize function 'send_shots' static varibles
send_shots.gspro_connection_notified = False
send_shots.shot_count = 1
send_shots.create_socket = True

def main():
    try:
        input("- Press enter after you've hit your first shot. -")

        ball_speed_last = total_spin_last = spin_axis_last = hla_last = vla_last = club_speed = None
        screenshot_attempts = 0
        incomplete_data_displayed = False
        ready_message_displayed = False

        # Create a ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=3)

        # Set the path where you want to save the screenshots
        screenshots_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Screenshots')

        print_colored_prefix(Color.GREEN, "GSPro ||", "Connecting to OpenConnect API ({}:{})...".format(HOST, PORT))

        # Run capture_window function in a separate thread
        if test_mode != TestModes.auto_shot:
            while True :
                try:
                    future_screenshot = executor.submit(capture_window, WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
                    screenshot = future_screenshot.result()
                    break
                except Exception as e:
                    print(f"{e}. Retrying")
                time.sleep(1)
        
        values = ["Ball Speed", "Spin Rate", "Spin Axis", "Launch Direction (HLA)", "Launch Angle (VLA)", "Club Speed"]

        # Ask user to select ROIs for each value, if they weren't found in the json
        if len(rois) == 0 :
            for value in values:
                print(f"Please select the ROI for {value}.")
                roi = select_roi(screenshot)
                rois.append(roi)
            print("You can paste these 6 lines into JSON")
            i = 1
            for roi in rois:
                print(f" \"ROI{i}\" : \"", roi[0],",",roi[1],",",roi[2],",",roi[3],"\",",end='')
                print(f"\t// {values[i-1]}")
                i=i+1
            print()

        #create_socket = True
        last_auto_time = 0
        graceful_shut = False
        while not graceful_shut:

            # send any pending shots from the queue.  Will block while awaiting response
            send_shots()
            
            # Run capture_window function in a separate thread
            if test_mode != TestModes.auto_shot:
                while True :
                    try:
                        future_screenshot = executor.submit(capture_window, WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
                        screenshot = future_screenshot.result()
                        break
                    except Exception as e:
                        print(f"{e}. Retrying")
                    time.sleep(1)

                result = []
                for roi in rois:
                    result.append(recognize_roi(screenshot, roi))
            else:
                if time.time() - last_auto_time > 6:
                    result = [random.randint(30,100), random.randint(1000,2000), random.randint(-10,10),0,10,80] # fake shot data
                    last_auto_time = time.time()
                #time.sleep(6)

            ball_speed, total_spin, spin_axis, hla, vla, club_speed = map(str, result)

            # Check if any values are incomplete/incorrect            
            try:
                sound_to_play = Sounds.bad_capture # default error sound
                if ball_speed == '-' and total_spin == '-' and spin_axis == '-' and hla == '-' and vla == '-' and club_speed == '-':
                    sound_to_play = Sounds.all_dashes
                    raise ValueError

                # Convert strings to floats
                ball_speed = float(ball_speed)
                total_spin = float(total_spin)
                spin_axis = float(spin_axis)
                hla = float(hla)
                vla = float(vla)
                club_speed = float(club_speed)

                # HLA and spin axis could well be 0.0
                if ball_speed == 0.0 or total_spin == 0.0 or vla == 0.0 or club_speed == 0.0:
                    raise ValueError("A value is 0")

                incomplete_data_displayed = False
                shot_ready = True
            except ValueError:
                if not incomplete_data_displayed:
                    screenshot_attempts += 1
                    sound_to_play.play()
                    print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "Invalid or incomplete data detected:")
                    print_colored_prefix(Color.RED,"MLM2PRO Connector ||", f"* Ball Speed: {ball_speed} MPH, Total Spin: {total_spin} RPM, Spin Axis: {spin_axis}°, HLA: {hla}°, VLA: {vla}°, Club Speed: {club_speed} MPH")
                    incomplete_data_displayed = True
                shot_ready = False
                continue

            # check if values are the same as previous
            if shot_ready and (ball_speed == ball_speed_last and total_spin == total_spin_last and
                spin_axis == spin_axis_last and hla == hla_last and vla == vla_last and club_speed == club_speed_last):
                if not ready_message_displayed:
                    print_colored_prefix(Color.BLUE, "MLM2PRO Connector ||", "System ready, take a shot...")
                    ready_message_displayed = True
                time.sleep(1)
                continue

            if (ball_speed != ball_speed_last or total_spin != total_spin_last or
                    spin_axis != spin_axis_last or hla != hla_last or vla != vla_last or club_speed != club_speed_last):
                #shot_count += 1
                screenshot_attempts = 0  # Reset the attempt count when valid data is obtained
                ready_message_displayed = False  # Reset the flag when data changes

                #print_colored_prefix(Color.GREEN,"MLM2PRO Connector ||", f"Shot {send_shots.shot_count} - Ball Speed: {ball_speed} MPH, Total Spin: {total_spin} RPM, Spin Axis: {spin_axis}°, HLA: {hla}°, VLA: {vla}°, Club Speed: {club_speed} MPH")

                message = {
                    "DeviceID": "Rapsodo MLM2PRO",
                    "Units": METRIC,
                    "ShotNumber": 999,
                    "APIversion": "1",
                    "BallData": {
                        "Speed": float(ball_speed),
                        "SpinAxis": float(spin_axis),
                        "TotalSpin": float(total_spin),
                        "BackSpin": round(total_spin * math.cos(math.radians(spin_axis))),
                        "SideSpin": round(total_spin * math.sin(math.radians(spin_axis))),
                        "HLA": float(hla),
                        "VLA": float(vla)
                    },
                    "ClubData": {
                        "Speed": float(club_speed)
                    },
                    "ShotDataOptions": {
                        "ContainsBallData": True,
                        "ContainsClubData": True,
                        "LaunchMonitorIsReady": True,
                        "LaunchMonitorBallDetected": True,
                        "IsHeartBeat": False
                    }
                }
                shot_q.put(message)

                ball_speed_last = ball_speed
                total_spin_last = total_spin
                spin_axis_last = spin_axis
                hla_last = hla
                vla_last = vla
                club_speed_last = club_speed
            time.sleep(1)

    except Exception as e:
        print_colored_prefix(Color.RED, "MLM2PRO Connector ||","An error occurred: {}".format(e))
    except KeyboardInterrupt:
        print("Ctrl-C pressed")
        graceful_shut = True
    finally:
        if api is not None:
            api.End()
            print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "Tesseract API ended...")
        # Believe it or not, if you close this socket, it will hang the GSPConnect.exe
        # and hog the CPU.
        # Leaving this open allows this connector to be repeatedly relaunched.  Weird
        #send_shots.sock.close()
        #print_colored_prefix(Color.RED, "GSPro ||", "Socket connection closed...")
        s.stop()
        sys.exit()

if __name__ == "__main__":
    s = MyServer()
    s.start()
    time.sleep(1)
    plt.ion()  # Turn interactive mode on.
    main()

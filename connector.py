from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os
import json
import tesserocr
from tesserocr import PyTessBaseAPI
import ctypes
from image_processing import capture_window, process_screenshot
from socket_connection import create_socket_connection
from PIL import Image
from datetime import datetime
import cv2
from matplotlib import pyplot as plt

# Set the path to the Tesseract OCR executable and library
tesseract_path = os.path.join(os.getcwd(), 'Tesseract-OCR')
tessdata_path = os.path.join(tesseract_path, 'tessdata')
tesseract_library = os.path.join(tesseract_path, 'libtesseract-5.dll')

# Set the Tesseract OCR path for tesserocr
tesserocr.tesseract_cmd = tessdata_path
ctypes.cdll.LoadLibrary(tesseract_library)

# Loading settings
def load_settings():
    with open(os.path.join(os.getcwd(), 'settings.json'), "r") as file:
        lines = file.readlines()
        cleaned_lines = [line.split("//")[0].strip() for line in lines if not line.strip().startswith("//")]
        cleaned_json = "\n".join(cleaned_lines)
        settings = json.loads(cleaned_json)
    return settings

settings = load_settings()
HOST = settings["HOST"]
PORT = settings["PORT"]
WINDOW_NAME = settings["WINDOW_NAME"]
TARGET_WIDTH = settings["TARGET_WIDTH"]
TARGET_HEIGHT = settings["TARGET_HEIGHT"]
METRIC = settings["METRIC"]

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
    cleaned_result = ''.join(c for c in result if c.isdigit() or c == '.' or c == '-' or c == '_' or c == '~')
    return cleaned_result.strip()


def main():
    try:
        input("- Press enter after you've hit your first shot. -")
        sock = create_socket_connection(HOST, PORT)
        ball_speed_last = total_spin_last = spin_axis_last = hla_last = vla_last = club_speed = None
        shot_count = 0
        screenshot_attempts = 0
        incomplete_data_displayed = False
        ready_message_displayed = False

        # Create a ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=3)

        # Set the path where you want to save the screenshots
        screenshots_path = os.path.join(os.path.expanduser('~'), 'Downloads', 'Screenshots')

        print_colored_prefix(Color.GREEN, "GSPro ||", "Connecting to OpenConnect API ({}:{})...".format(HOST, PORT))

        # Run capture_window function in a separate thread
        future_screenshot = executor.submit(capture_window, WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
        screenshot = future_screenshot.result()

        # Ask user to select ROIs for each value
        rois = []
        values = ["Ball Speed", "Spin Rate", "Spin Axis", "Launch Direction (VLA)", "Launch Angle (HLA)", "Club Speed"]
        for value in values:
            print(f"Please select the ROI for {value}.")
            roi = select_roi(screenshot)
            rois.append(roi)

        while True:
            # Run capture_window function in a separate thread
            future_screenshot = executor.submit(capture_window, WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
            screenshot = future_screenshot.result()

            result = []
            for roi in rois:
                result.append(recognize_roi(screenshot, roi))

            ball_speed, total_spin, spin_axis, hla, vla, club_speed = map(str, result)

            # Check if any values are incomplete/incorrect            
            try:
                # Convert strings to floats
                ball_speed = float(ball_speed)
                total_spin = float(total_spin)
                spin_axis = float(spin_axis)
                hla = float(hla)
                vla = float(vla)
                club_speed = float(club_speed)

                if ball_speed == 0.0 or spin_axis == 0.0 or total_spin == 0.0 or hla == 0.0 or vla == 0.0 or club_speed == 0.0:
                    raise ValueError("A value is 0")

                incomplete_data_displayed = False
                error_occurred = False  # Reset the flag when valid data is obtained
                shot_ready = True

            except ValueError:
                if not incomplete_data_displayed:
                    screenshot_attempts += 1
                    print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "Invalid or incomplete data detected, retaking screenshot...")
                    incomplete_data_displayed = True
                error_occurred = True  # Set the flag when an error occurs
                shot_ready = False
                continue

            # check if values are the same as previous
            if shot_ready and (ball_speed == ball_speed_last and total_spin == total_spin_last and
                spin_axis == spin_axis_last and hla == hla_last and vla == vla_last and club_speed == club_speed_last):
                if not ready_message_displayed or error_occurred:  # Include the error_occurred flag here
                    print_colored_prefix(Color.BLUE, "MLM2PRO Connector ||", "System ready, take a shot...")
                    ready_message_displayed = True
                    error_occurred = False  # Reset the flag after the message is printed
                time.sleep(1)
                continue

            if (ball_speed != ball_speed_last or total_spin != total_spin_last or
                    spin_axis != spin_axis_last or hla != hla_last or vla != vla_last or club_speed != club_speed_last):
                shot_count += 1
                screenshot_attempts = 0  # Reset the attempt count when valid data is obtained
                ready_message_displayed = False  # Reset the flag when data changes

                print_colored_prefix(Color.GREEN,"MLM2PRO Connector ||", f"Shot {shot_count} - Ball Speed: {ball_speed} MPH, Total Spin: {total_spin} RPM, Spin Axis: {spin_axis}°, HLA: {hla}°, VLA: {vla}°, Club Speed: {club_speed} MPH")

                message = {
                    "DeviceID": "Rapsodo MLM2PRO",
                    "Units": METRIC,
                    "ShotNumber": shot_count,
                    "APIversion": "1",
                    "BallData": {
                        "Speed": float(ball_speed),
                        "SpinAxis": float(spin_axis),
                        "TotalSpin": float(total_spin),
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
                try:
                    sock.sendall(json.dumps(message).encode())
                    print_colored_prefix(Color.BLUE, "MLM2PRO Connector ||", "Shot data has been sent successfully...")
                except Exception as e:
                    print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "Failed to send shot data: {}".format(e))


                ball_speed_last = ball_speed
                total_spin_last = total_spin
                spin_axis_last = spin_axis
                hla_last = hla
                vla_last = vla
                club_speed_last = club_speed

            time.sleep(1)
    except Exception as e:
        print_colored_prefix(Color.RED, "MLM2PRO Connector ||","An error occurred: ", e)

    finally:
        if api is not None:
            api.End()
            print_colored_prefix(Color.RED, "MLM2PRO Connector ||", "Tesseract API ended...")
        sock.close()
        print_colored_prefix(Color.RED, "GSPro ||", "Socket connection closed...")

if __name__ == "__main__":
    plt.ion()  # Turn interactive mode on.
    main()
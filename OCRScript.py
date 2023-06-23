import cv2
import numpy as np
from ctypes import windll
import win32gui
import win32ui
import os
import time
from PIL import Image
import json
import socket
import pytesseract

# Configurable parameters
TESSERACT_PATH = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
HOST, PORT = "127.0.0.1", 921
WINDOW_NAME = "AirPlay"
TARGET_WIDTH = 1638
TARGET_HEIGHT = 752

# Setting the tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def capture_window(window_name: str, target_width: int, target_height: int) -> np.array:
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, window_name)

    rect = win32gui.GetClientRect(hwnd) 
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]

    # getting the window's position
    rect_pos = win32gui.GetWindowRect(hwnd)
    left = rect_pos[0]
    top = rect_pos[1]

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bitmap)

    # Using PrintWindow for the screenshot, this includes the window frame
    result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
    img = np.ascontiguousarray(img)[..., :-1]  # make image C_CONTIGUOUS and drop alpha channel

    if not result:  # result should be 1
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")

    # Resize the image to the target dimensions
    img = cv2.resize(img, (target_width, target_height))

    return img


def process_screenshot(file_path):
    example = Image.open(file_path).convert('L')
    columns_px = (1638 - 230) / 5
    row_px = (752 - 270) / 3
    margin = 5
    keep = [{'x': 1, 'y': 3}, {'x': 1, 'y': 4}, {'x': 2, 'y': 1}, {'x': 2, 'y': 4}, {'x': 3, 'y': 1}]
    metric = [""] * len(keep)
    value = [0] * len(keep)

    for i in range(len(keep)):
        left = 230 + margin + (keep[i]['y'] - 1) * columns_px
        upper = 270 + margin + (keep[i]['x'] - 1) * row_px
        right = left + columns_px - margin
        lower = upper + 47 - margin
        geomMetric = (left, upper, right, lower)

        cropMetric = example.crop(geomMetric)
        metric[i] = pytesseract.image_to_string(cropMetric)

        left = 230 + margin + (keep[i]['y'] - 1) * columns_px
        upper = 270 + 47 + margin + (keep[i]['x'] - 1) * row_px
        right = left + columns_px - margin
        lower = upper + row_px - 47 - margin
        geomValue = (left, upper, right, lower)

        cropValue = example.crop(geomValue)

        # Convert PIL image to OpenCV format
        cropValue_cv = np.array(cropValue) 
        cropValue_cv = cv2.cvtColor(cropValue_cv, cv2.COLOR_RGB2BGR)

        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(cropValue_cv, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Convert back to PIL format for pytesseract
        thresholded_pil = Image.fromarray(thresholded)

        raw_value = pytesseract.image_to_string(thresholded_pil, config='--psm 7 --oem 3 -c tessedit_char_whitelist=$.0123456789-')

        # Ignore non-numeric characters
        cleaned_value = ''.join(c for c in raw_value if c.isdigit() or c == '.' or c == '-' or c == '_' or c == '~')
        value[i] = cleaned_value if cleaned_value else '0'

    result = [{"metric": metric[i].strip().upper(), "value": value[i]} for i in range(len(keep))]
    return result


def create_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except socket.error as e:
        print(f"Could not establish connection: {e}")
        raise
    return sock


def update_last_values(ball_speed, total_spin, sa, hla, vla):
    return ball_speed, total_spin, sa, hla, vla


def main():
    input("Press enter after you've hit your first shot.")
    sock = create_socket_connection(HOST, PORT)
    downloads_folder = os.path.expanduser("~\Downloads")
    file_path = os.path.join(downloads_folder, "results.png")

    ball_speed_last = total_spin_last = sa_last = hla_last = vla_last = None
    shot_count = 0

    while True:
        screenshot = capture_window(WINDOW_NAME, TARGET_WIDTH, TARGET_HEIGHT)
        cv2.imwrite(file_path, screenshot)

        result = process_screenshot(file_path)

        if any(r['metric'].strip() == "-" or r['value'] == "-" for r in result):
            print("Incomplete data, retaking screenshot...")
            continue

        ball_speed = float(next((r['value'] for r in result if r['metric'].strip() == "BALL SPEED"), 0))
        total_spin = float(next((r['value'] for r in result if r['metric'].strip() == "SPIN RATE"), 0))
        sa = float(next((r['value'] for r in result if r['metric'].strip() == "SPIN AXIS"), 0))
        hla = float(next((r['value'] for r in result if r['metric'].strip() == "LAUNCH DIRECTION"), 0))
        vla = float(next((r['value'] for r in result if r['metric'].strip() == "LAUNCH ANGLE"), 0))

        if sum([
            ball_speed != ball_speed_last,
            total_spin != total_spin_last,
            sa != sa_last,
            hla != hla_last,
            vla != vla_last
        ]) >= 2:
            ball_speed_last, total_spin_last, sa_last, hla_last, vla_last = update_last_values(ball_speed, total_spin, sa, hla, vla)

            shot_count += 1

            print(f"- Shot {shot_count} -")
            print(f"Ball Speed: {ball_speed} MPH")
            print(f"Total Spin: {total_spin} RPM")
            print(f"Spin Axis: {sa}°")
            print(f"HLA: {hla}°")
            print(f"VLA: {vla}°")

            data_to_send = {
                "DeviceID": "MLM2PRO",
                "Units": "Yards",
                "ShotNumber": shot_count,
                "APIversion": "1",
                "BallData": {
                    "Speed": ball_speed,
                    "SpinAxis": sa,
                    "TotalSpin": total_spin,
                    "HLA": hla,
                    "VLA": vla
                },
                "ShotDataOptions": {
                    "ContainsBallData": "true",
                    "ContainsClubData": "false"
                }
            }

            print(json.dumps(data_to_send))
            sock.sendall(json.dumps(data_to_send).encode("utf-8"))

        time.sleep(0.5)

    sock.close()

if __name__ == '__main__':
    main()

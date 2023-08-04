import cv2
import numpy as np
from PIL import Image
from ctypes import windll
import win32gui
import win32ui
import matplotlib.pyplot as plt

w = -1
h = -1

shown_window_failure = False
shown_size_failure = False
shown_target_failure = False
def capture_window(window_name: str, target_width: int, target_height: int) -> np.array:
    global w, h
    global shown_window_failure
    global shown_size_failure
    global shown_target_failure
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, window_name)
    if not hwnd:
        if not shown_window_failure:
            shown_window_failure = True
            print(f"Can't find window called '{window_name}'")
        raise RuntimeError(f"Looking for window '{window_name}'")
    if shown_window_failure:
        print(f"Found window called '{window_name}'")
        shown_window_failure = False
    
    rect = win32gui.GetClientRect(hwnd)
    if w == -1 :
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
    else :
        if not (w == rect[2] - rect[0] and h == rect[3] - rect[0]) :
           if not shown_size_failure:
                shown_size_failure = True
                print(f"Target window ({window_name}) size has changed to {rect[2] - rect[0]}x{rect[3] - rect[1]}, not {w}x{h}")
           raise RuntimeError(f"Please fix size of window {window_name}")
    if shown_size_failure:
        print(f"Window '{window_name}' is now the correct size")
        shown_size_failure = False

    if not (w==target_width and h==target_height) :
        if not shown_target_failure:
            shown_target_failure = True
            print(f"Warning: dimensions seem wrong {w}x{h} vs json:{target_width}x{target_height}")
        raise RuntimeError("Try fixing screen mirror dimensions")
    else :
        if shown_target_failure:
            shown_target_failure = False
            print(f"Dimensions look good now")

    try:
        bitmap = None
        save_dc = None
        mfg_dc = None
        hwmd_dc = None
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)

        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4))
        img = np.ascontiguousarray(img)[..., :-1] 

    except Exception as e:
        raise RuntimeError(f"{e}. Exception in capture_window")

    finally:
        # Clean up what we can, then return with exception
        if bitmap is not None:
            win32gui.DeleteObject(bitmap.GetHandle())
        if save_dc is not None:
            save_dc.DeleteDC()
        if mfg_dc is not None:
            mfc_dc.DeleteDC()
        if hwnd_dc is not None:
            win32gui.ReleaseDC(hwnd, hwnd_dc)
    
    return img

def process_screenshot(screenshot, api, rois):
    example = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)).convert('L')
    result = {}

    for name, roi in rois.items():
        cropValue = example.crop(roi)
        
        # Convert to binary
        cropValue_np = np.array(cropValue)
        _, binary_cropValue = cv2.threshold(cropValue_np, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Morphological operation to remove smaller text elements
        kernel = np.ones((5,5),np.uint8)
        binary_cropValue = cv2.morphologyEx(binary_cropValue, cv2.MORPH_OPEN, kernel)

        # Convert binary image back to PIL Image
        binary_cropValue_pil = Image.fromarray(binary_cropValue)

        api.SetImage(binary_cropValue_pil)
        raw_value = api.GetUTF8Text()

        # Ignore non-numeric characters
        cleaned_value = ''.join(c for c in raw_value if c.isdigit() or c == '.' or c == '-')

        # strip any trailing periods, which can sometimes come from degree symbols
        value = cleaned_value.strip('.') if cleaned_value else '0'

        result[name] = value

    return result

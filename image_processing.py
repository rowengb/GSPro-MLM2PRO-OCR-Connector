import cv2
import numpy as np
from PIL import Image
from ctypes import windll
import win32gui
import win32ui

def capture_window(window_name: str, target_width: int, target_height: int) -> np.array:
    windll.user32.SetProcessDPIAware()
    hwnd = win32gui.FindWindow(None, window_name)

    rect = win32gui.GetClientRect(hwnd) 
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]

    rect_pos = win32gui.GetWindowRect(hwnd)
    left = rect_pos[0]
    top = rect_pos[1]

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

    if not result:  
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        raise RuntimeError(f"Unable to acquire screenshot! Result: {result}")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for auto_crop function
    img_cropped = auto_crop(img_gray)
    img_resized = cv2.resize(img_cropped, (target_width, target_height))

    return img_resized

def auto_crop(image):

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize bounding rectangle coordinates
    x_min, y_min, x_max, y_max = None, None, None, None

    # Calculate the bounding box coordinates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x) if x_min is not None else x
        y_min = min(y_min, y) if y_min is not None else y
        x_max = max(x_max, x + w) if x_max is not None else x + w
        y_max = max(y_max, y + h) if y_max is not None else y + h

    # If no contours are detected, return the original image
    if x_min is None or y_min is None or x_max is None or y_max is None:

        return image

    # Add padding around the bounding box
    padding = 5
    x_min = max(int(x_min) - padding, 0)
    y_min = max(int(y_min) - padding, 0)
    x_max = min(int(x_max) + padding, image.shape[1])
    y_max = min(int(y_max) + padding, image.shape[0])

    # Crop the image using calculated coordinates
    cropped = image[y_min:y_max, x_min:x_max]

    return cropped

def process_screenshot(screenshot, api, coords):
    example = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)).convert('L')
    result = []

    for (x_min, y_min, x_max, y_max) in (coord['value'] for coord in coords):
        cropValue = example.crop((x_min, y_min, x_max, y_max))
        
        # Convert to binary
        # The conversion process is done on a numpy array, so we convert the PIL image to a numpy array first.
        cropValue_np = np.array(cropValue)
        _, binary_cropValue = cv2.threshold(cropValue_np, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Morphological operation to remove smaller text elements
        kernel = np.ones((5,5),np.uint8)
        binary_cropValue = cv2.morphologyEx(binary_cropValue, cv2.MORPH_OPEN, kernel)

        # Autocrop the preprocessed image
        binary_cropValue = auto_crop(binary_cropValue)

        # Convert binary image back to PIL Image
        binary_cropValue_pil = Image.fromarray(binary_cropValue)

        api.SetImage(binary_cropValue_pil)
        raw_value = api.GetUTF8Text()

        # Ignore non-numeric characters
        cleaned_value = ''.join(c for c in raw_value if c.isdigit() or c == '.' or c == '-' or c == '_' or c == '~')
        value = cleaned_value if cleaned_value else '0'

        result.append(value)

    return result

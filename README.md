# GSPro-MLM2PRO-OCR
A Python script that interfaces the Rapsodo MLM2PRO golf launch monitor with the GSPro golf simulation software using OpenCV and Tesseract OCR.

Required:

- AirPlay app from Windows Store - https://www.microsoft.com/store/productId/9P6QSQ5PH9KR
- Rapsodo MLM2PRO app on iPhone - https://apps.apple.com/us/app/rapsodo-mlm2pro/id1659665092
- Callaway® RPT™ Chrome Soft X® Golf Balls (3 Included in MLM2PRO box), these are necessary as we need to calculate Spin Axis and Spin Rate - https://rapsodo.com/products/callaway-rpt-chrome-soft-x-golf-balls

Steps:

1. Open Rapsodo MLM2PRO app, connect your LM, and go to the Rapsodo Range
2. Click on the little arrow next to "Ball Speed" on the right to show all the metrics
3. Mirror your iPhone screen to the AirPlay windows app
4. Adjust the AirPlay window size so that the Rapsodo MLM2Pro App fills it out completely with no black borders/bars (Example: https://ibb.co/DMHx12S)
5. Minimize the AirPlay application window
6. Open GSPRO and GSPRO Connect Open API window
7. Either Run the PY script in CMD (you will need to have all necessary modules installed) or download the ZIP from v0.1.0 and run the MLM2PROConnector.exe file (https://github.com/rowengb/GSPro-MLM2PRO-OCR/releases/tag/v0.1.0)
8. Take your first shot, wait for the numbers to populate on the iPhone app and then press the Enter key
9. Done

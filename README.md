# GSPro-MLM2PRO-OCR-Connector
A Python app that interfaces the Rapsodo MLM2PRO golf launch monitor with the GSPro golf simulation software using OpenCV and TesserOCR.

Required:

1. Screen Mirroring App
  - iOS/iPhone - AirPlay app from Windows Store - https://www.microsoft.com/store/productId/9P6QSQ5PH9KR
  - Android - EasyCast app from Windows Store - https://www.microsoft.com/store/productId/9P8BH9SMXQMC (also requires Android app installation)

2. Rapsodo MLM2PRO App
  - iPhone - https://apps.apple.com/us/app/rapsodo-mlm2pro/id1659665092
  - Android - https://play.google.com/store/apps/details?id=com.rapsodo.MLM&hl=en_US&gl=US

3. Golf Balls
- Callaway® RPT™ Chrome Soft X® Golf Balls (3 Included in MLM2PRO box), these are necessary to accurately calculate Spin Axis and Spin Rate - https://rapsodo.com/products/callaway-rpt-chrome-soft-x-golf-balls

Steps:

1. Download the ZIP from v1.0, unzip it, and open the Settings.json file (https://github.com/rowengb/GSPro-MLM2PRO-OCR/releases/tag/v1.0).
2. By default, the WINDOW_NAME is set to "AirPlay" (for iPhone), if you are using Android, change it to "EasyCast". Once done, save the file (Ctrl+S) and close it.
3. Open the Rapsodo MLM2PRO phone app, connect your Launch Monitor, and go to Simulation > Rapsodo Range.
4. Click on the little arrow next to "Ball Speed" on the right to show all the metrics.
5. Mirror your phone screen to the AirPlay/EasyCast windows app (Depending on your phone OS).
6. Adjust the AirPlay/EasyCast window size so that the Rapsodo MLM2Pro App fills it out with little to no black borders/bars (Doesn't have to be perfect, the connector app will still work with black borders/bars but may not be as accurate) - (Example: https://ibb.co/DMHx12S).
7. Minimize the AirPlay/EasyCast application window (Important!).
8. Open GSPRO and GSPRO Connect Open API window (Go to Range or Local Match).
10. Run the MLM2PROConnectorV1.exe app file as ADMINISTRATOR (located in the previously downloaded/unziped ZIP file) and wait for the "Press enter after you've hit your first shot" line to show.
11. Take your first shot, wait for the numbers to populate on the Rapsodo Range in the MLM2PRO app and then press the Enter key.
12. Done!

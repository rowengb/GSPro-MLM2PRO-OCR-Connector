# GSPro-MLM2PRO-OCR-Connector
A Python app that interfaces the Rapsodo MLM2PRO golf launch monitor with the GSPro golf simulation software using OpenCV and TesserOCR.

Required:

1. Screen Mirroring App
- iPhone/iPad one of:
	- AirPlay app from Windows Store - https://www.microsoft.com/store/productId/9P6QSQ5PH9KR
	- 5KPlayer - https://www.5kplayer.com/
- Android one of:
	- SCRCPY - 'Screen Copy' via USB - https://github.com/Genymobile/scrcpy (also requires enabling of USB debugging mode in Developer options.  See https://developer.android.com/studio/debug/dev-options)

2. Rapsodo MLM2PRO App
  - iPhone/iPad - https://apps.apple.com/us/app/rapsodo-mlm2pro/id1659665092
  - Android - https://play.google.com/store/apps/details?id=com.rapsodo.MLM&hl=en_US&gl=US

3. Golf Balls
- Callaway® RPT™ Chrome Soft X® Golf Balls (3 Included in MLM2PRO box), these are necessary to accurately calculate Spin Axis and Spin Rate - https://rapsodo.com/products/callaway-rpt-chrome-soft-x-golf-balls
- or any other ball with dot stickers from 
	- justapotamus - https://justapotamusdesigns.com/collections/rapsodo-mlm2pro-ball-dots
	- rapsodoballdots - https://rapsodoballdots.com/

Optional:

4. Putting
- This connector will interface with the utility found here, which was created for the R10 https://github.com/alleexx/cam-putting-py/releases
- See the readme at the above location for setup details
- The putting server will start by default, but if you want to disable it, you can include "DISABLE_PUTTING" : 1 in the settings.json
- Any putts detected while the putter isn't selected in GSPRO will be ignored

Steps:

1. Download the ZIP from v2.0, unzip it, and open the Settings.json file (https://github.com/rowengb/GSPro-MLM2PRO-OCR/releases/tag/v2.0).
2. By default, the WINDOW_NAME is set to "AirPlay" (for iOS).  If you are using 5KPlayer, you may need to change your device name to remove the apostrophe.  If you are using Android, change it to the name of the SCRCPY window (your device name). Also update the TARGET_HEIGHT and TARGET_WIDTH accordingly. Once done, save the file (Ctrl+S) and close it.
3. Open the Rapsodo MLM2PRO app, connect your Launch Monitor, and go to Simulation > Rapsodo Range.
4. If using a phone, click on the little arrow next to "Ball Speed" on the right to show all the metrics.
5. Mirror your device screen using
	- iPhone/iPad: AirPlay or 5KPlayer apps
	- Android: EasyCast app or scrcpy via USB debugging mode
6. Adjust the AirPlay/EasyCast window size so that the Rapsodo MLM2Pro App fills it out with little to no black borders/bars (Doesn't have to be perfect, the connector app will still work with black borders/bars but may not be as accurate) - (Example: https://ibb.co/DMHx12S).
7. Open GSPRO and GSPRO Connect Open API window (Go to Range or Local Match to test).
8. Run the MLM2PROConnectorV2.exe app file as ADMINISTRATOR (located in the previously downloaded/unzipped ZIP file) and wait for the "Press enter after you've hit your first shot" line to show.
9. Take your first shot, wait for the numbers to populate on the Rapsodo Range in the MLM2PRO app and then press the Enter key.
10. Set the ROIs for each shot metric one by one by creating rectangles around the desired value (See tutorial/example here - https://www.youtube.com/watch?v=zLptVv8umaU).  You can copy and paste the ROI values printed into your settings.json to avoid the need to select the boundaries each time.  If you get a misread shot, read the console to determine which one, then edit the ROI in your JSON.  The first two numbers are the X,Y location of the upper left corner, and the next two are width and height.  You can Ctrl-C to exit the connector and relaunch it to test your new ROI dimesions.
11. Optional for putting: start the ball_tracking.exe from its installation directory.  Don't forget to specify your ball color preference on the command line (with -c <color>).  Note, -c calibrate is a handy utility to determine which color setting is best for your setup
11. Done!

NOTE: Make sure to have GSPro, GSPro Connect and the AirPlay receiver app open and running before opening the MLM2PROConnectorV2.exe app otherwise it will just close instantly after pressing enter to confirm your first shot.

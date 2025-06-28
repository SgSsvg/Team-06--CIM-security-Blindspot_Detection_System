import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import csv
import sys
import cv2
from ultralytics import YOLO

# Optional: Uncomment the following lines to enable audio alerts with pygame
# import pygame
# pygame.mixer.init()
# alert_sound = pygame.mixer.Sound('alert.wav')

# Initialize CSV file for radar points
with open('radar_points.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'FrameNumber', 'X (m)', 'Y (m)', 'Range (m)', 'Doppler (m/s)', 'PeakVal', 'Camera_Dist (m)', 'Fused_Dist (m)'])

# Calibration parameters
pixel_per_meter = 100  # Adjust based on camera FOV (calibrate with a known object)
offset_x = 640  # Center of 1280x720 image (calibrate)
offset_y = 360  # Center of 1280x720 image (calibrate)
focal_length = 1000  # Focal length in pixels (calibrate with camera specs)
known_width = 0.5  # Average width of a person in meters (adjust for humans)

# Blind spot zones (in meters, adjust as needed)
blind_spots = {
    'left': {'x_min': -5, 'x_max': -2, 'y_min': 1, 'y_max': 5},
    'right': {'x_min': 2, 'x_max': 5, 'y_min': 1, 'y_max': 5}
}

# Load YOLOv8 pre-trained model from Ultralytics
yolo_model = YOLO('yolov8n.pt')  # Automatically downloads if not present
yolo_conf_threshold = 0.5  # Adjustable confidence threshold

# Serial configuration
configFileName = 'configuration_file'
byteBuffer = np.zeros(2**15, dtype='uint8')
byteBufferLength = 0

def serialConfig(configFileName):
    CLIport = serial.Serial('COM4', 115200)
    Dataport = serial.Serial('COM3', 921600)
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)
    return CLIport, Dataport

def parseConfigFile(configFileName):
    configParameters = {}
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        splitWords = i.split(" ")
        numRxAnt = 4
        numTxAnt = 2
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 *= 2
            digOutSampleRate = int(splitWords[11])
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(float(splitWords[5]))
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    scaling_factor = 0.78125
    configParameters["rangeIdxToMeters"] *= scaling_factor
    return configParameters

def readAndParseData16xx(Dataport, configParameters):
    global byteBuffer, byteBufferLength
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2**15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    maxBufferSize = 2**15
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    magicOK = 0
    dataOK = 0
    frameNumber = 0
    detObj = {}
    if Dataport.in_waiting > 0:
        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)
        if (byteBufferLength + byteCount) < maxBufferSize:
            byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
            byteBufferLength += byteCount
        if byteBufferLength > 16:
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc+8]
                if np.all(check == magicWord):
                    startIdx.append(loc)
            if startIdx:
                if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                    byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype='uint8')
                    byteBufferLength -= startIdx[0]
                if byteBufferLength < 0:
                    byteBufferLength = 0
                word = [1, 2**8, 2**16, 2**24]
                totalPacketLen = np.matmul(byteBuffer[12:12+4], word)
                if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                    magicOK = 1
        if magicOK:
            idX = 0
            magicNumber = byteBuffer[idX:idX+8]
            idX += 8
            version = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            subFrameNumber = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            for tlvIdx in range(numTLVs):
                word = [1, 2**8, 2**16, 2**24]
                try:
                    tlv_type = np.matmul(byteBuffer[idX:idX+4], word)
                    idX += 4
                    tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
                    idX += 4
                except:
                    pass
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    word = [1, 2**8]
                    tlv_numObj = np.matmul(byteBuffer[idX:idX+2], word)
                    idX += 2
                    tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX+2], word)
                    idX += 2
                    rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                    peakVal = np.zeros(tlv_numObj, dtype='int16')
                    x = np.zeros(tlv_numObj, dtype='int16')
                    y = np.zeros(tlv_numObj, dtype='int16')
                    z = np.zeros(tlv_numObj, dtype='int16')
                    for objectNum in range(tlv_numObj):
                        rangeIdx[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        x[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        y[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        z[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                    rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                    dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] = dopplerIdx[dopplerIdx > (configParameters["numDopplerBins"]/2 - 1)] - 65535
                    dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat
                    detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx,
                              "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                    dataOK = 1
            if idX > 0 and byteBufferLength > idX:
                shiftSize = totalPacketLen
                byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]), dtype='uint8')
                byteBufferLength -= shiftSize
                if byteBufferLength < 0:
                    byteBufferLength = 0
    print(f"Radar Frame: {frameNumber}, Objects: {detObj.get('numObj', 0)}")  # Debug radar data
    return dataOK, frameNumber, detObj

class ApplicationWindow(pg.GraphicsLayoutWidget):
    def __init__(self, CLIport, Dataport, configParameters):
        super().__init__()
        self.setWindowTitle("Radar-Camera Fusion BSD with Distance")
        self.plot = self.addPlot()
        self.plot.setLabel('left', 'Y (meters)')
        self.plot.setLabel('bottom', 'X (meters)')
        self.plot.setXRange(-10, 10)
        self.plot.setYRange(0, 10)
        self.CLIport = CLIport
        self.dataport = Dataport
        self.configParameters = configParameters
        self.frameData = {}
        self.currentIndex = 0
        self.buffer = []
        self.last_save = time.time()
        self.last_frame_time = time.time()  # For synchronization
        self.cap = cv2.VideoCapture(0)  # Initialize webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        # Initialize moving average buffer for fused distances
        self.distance_buffer = []
        self.max_buffer_size = 5  # Number of frames for moving average

    def project_radar_to_image(self, x, y, ranges):
        # Project radar coordinates to image plane using range-based scaling
        img_x = offset_x + (x / ranges) * (pixel_per_meter * ranges.max())
        img_y = offset_y - (y / ranges) * (pixel_per_meter * ranges.max())
        return np.clip(img_x, 0, 1279), np.clip(img_y, 0, 719)

    def calculate_camera_distance(self, width_pixels):
        # Distance estimation using pinhole camera model: D = (known_width * focal_length) / width_pixels
        return (known_width * focal_length) / width_pixels if width_pixels > 0 else float('inf')

    def calculate_dynamic_fused_distance(self, radar_range, camera_dist, radar_confidence, camera_confidence):
        # Normalize confidences to [0, 1]
        radar_confidence = np.clip(radar_confidence / 1000.0, 0, 1)  # Assuming peakVal max ~1000
        camera_confidence = np.clip(camera_confidence, 0, 1)
        
        # Dynamic weights based on confidences
        total_confidence = radar_confidence + camera_confidence
        weight_radar = radar_confidence / total_confidence if total_confidence > 0 else 0.5
        weight_camera = camera_confidence / total_confidence if total_confidence > 0 else 0.5
        
        # Fused distance with dynamic weighting
        if radar_range is not None and camera_dist > 0 and camera_dist < float('inf'):
            return (weight_radar * radar_range + weight_camera * camera_dist)
        return radar_range if radar_range is not None else camera_dist

    def apply_moving_average(self, new_distance):
        # Add new distance to buffer and maintain max size
        self.distance_buffer.append(new_distance)
        if len(self.distance_buffer) > self.max_buffer_size:
            self.distance_buffer.pop(0)
        # Compute moving average if buffer has data
        return np.mean(self.distance_buffer) if self.distance_buffer else new_distance

    def find_nearest_radar_point(self, img_x, img_y, radar_x, radar_y, ranges, peak_vals):
        if len(radar_x) == 0:
            return None, None, None, None
        distances = np.sqrt((img_x - radar_x)**2 + (img_y - radar_y)**2)
        min_idx = np.argmin(distances)
        if distances[min_idx] < 100:  # Threshold for association (in pixels)
            return radar_x[min_idx], radar_y[min_idx], ranges[min_idx], peak_vals[min_idx]
        return None, None, None, None

    def is_in_blind_spot(self, x, y):
        for zone_name, zone in blind_spots.items():
            if (zone['x_min'] <= x <= zone['x_max'] and zone['y_min'] <= y <= zone['y_max']):
                return zone_name
        return None

    def update(self):
        # Read radar data with timestamp
        radar_start_time = time.time()
        dataOK, frameNumber, detObj = readAndParseData16xx(self.dataport, self.configParameters)
        radar_end_time = time.time()

        # Read camera frame with timestamp
        camera_start_time = time.time()
        ret, frame = self.cap.read()
        camera_end_time = time.time()
        if not ret:
            print("Failed to read camera frame")
            return

        # Run YOLO detection
        yolo_start_time = time.time()
        results = yolo_model(frame, stream=True)
        yolo_detections = []
        humans_detected = 0
        vehicles_detected = 0
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confidences, classes):
                if conf >= yolo_conf_threshold:
                    if cls == 0:  # Person (human)
                        yolo_detections.append((box, True, conf))  # Include confidence
                        humans_detected += 1
                    elif cls in [2, 3, 5, 7]:  # Car, motorcycle, bus, truck
                        yolo_detections.append((box, False, conf))  # Include confidence
                        vehicles_detected += 1
        yolo_end_time = time.time()
        print(f"YOLO Frame: Detected {humans_detected} humans, {vehicles_detected} vehicles, Total Time: {yolo_end_time - yolo_start_time:.3f}s")

        # Process radar data and fusion
        radar_x, radar_y, radar_ranges, peak_vals = [], [], [], []
        if dataOK and detObj.get("numObj", 0) > 0:
            x = -detObj["x"]
            y = detObj["y"]
            ranges = detObj["range"]
            doppler = detObj["doppler"]
            peakVal = detObj["peakVal"]
            timestamp = (radar_start_time + radar_end_time) / 2  # Average radar timestamp
            radar_x, radar_y, radar_ranges, peak_vals = x, y, ranges, peakVal

            # Project radar points to image plane
            img_x, img_y = self.project_radar_to_image(radar_x, radar_y, radar_ranges)

            # Synchronization check
            time_diff = abs(timestamp - camera_end_time)
            print(f"Sync Check: Radar Time {timestamp:.3f}s, Camera Time {camera_end_time:.3f}s, Diff {time_diff:.3f}s")

        # Associate radar points with YOLO detections
        for box, is_human, conf in yolo_detections:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            camera_dist = self.calculate_camera_distance(width)
            radar_x_match, radar_y_match, radar_range, radar_conf = self.find_nearest_radar_point(center_x, center_y, img_x, img_y, radar_ranges, peak_vals) if len(radar_x) > 0 else (None, None, None, None)
            fused_dist = self.calculate_dynamic_fused_distance(radar_range, camera_dist, radar_conf if radar_conf is not None else 0, conf)
            smoothed_dist = self.apply_moving_average(fused_dist)

            # Draw bounding box and distance
            color = (0, 0, 255) if is_human else (0, 255, 0)  # Red for humans, Green for vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = 'Human' if is_human else 'Vehicle'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f'F: {smoothed_dist:.2f}m', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Log to buffer if radar match exists
            if radar_x_match is not None:
                doppler_idx = np.argmin(np.sqrt((img_x - center_x)**2 + (img_y - center_y)**2))
                self.buffer.append([timestamp, frameNumber, radar_x_match, radar_y_match, radar_range, doppler[doppler_idx], peakVal[doppler_idx], camera_dist, smoothed_dist])

        # Check for blind spot detections and trigger alerts
        alerts = []
        for box, is_human, conf in yolo_detections:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radar_x_match, radar_y_match, radar_range, radar_conf = self.find_nearest_radar_point(center_x, center_y, img_x, img_y, radar_ranges, peak_vals) if len(radar_x) > 0 else (None, None, None, None)
            if radar_x_match is not None:
                zone = self.is_in_blind_spot(radar_x_match, radar_y_match)
                if zone:
                    width = x2 - x1
                    camera_dist = self.calculate_camera_distance(width)
                    fused_dist = self.calculate_dynamic_fused_distance(radar_range, camera_dist, radar_conf if radar_conf is not None else 0, conf)
                    smoothed_dist = self.apply_moving_average(fused_dist)
                    alerts.append((radar_x_match, radar_y_match, zone, radar_range, camera_dist, smoothed_dist))
                    print(f"Alert: Object in {zone} blind spot at ({radar_x_match:.2f}, {radar_y_match:.2f})m, Radar: {radar_range:.2f}m, Camera: {camera_dist:.2f}m, Fused: {smoothed_dist:.2f}m")
                    # Optional: Uncomment for audio alerts with pygame
                    # try:
                    #     alert_sound.play()
                    # except:
                    #     print("Alert sound failed. Ensure alert.wav exists.")

        # Save radar data to CSV
        if timestamp - self.last_save > 0.5 and len(self.buffer) > 0:
            with open('radar_points.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                for row in self.buffer:
                    writer.writerow(row)
            self.buffer = []
            self.last_save = timestamp

        # Visualize radar points and camera feed
        self.plot.clear()
        self.image_item.setImage(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(radar_x) > 0:
            self.plot.plot(radar_x, radar_y, pen=None, symbol='o', symbolPen='w', symbolBrush='r')
            for i in range(len(radar_x)):
                if radar_ranges[i] < 5:
                    label = pg.TextItem(f"R:{radar_ranges[i]:.2f}m", anchor=(0.5, -0.5))
                    label.setPos(radar_x[i], radar_y[i])
                    self.plot.addItem(label)
            for rx, ry, zone, rrange, camera_dist, smoothed_dist in alerts:
                label = pg.TextItem(f"Alert: {zone}", color='r', anchor=(0.5, -1.0))
                label.setPos(rx, ry)
                self.plot.addItem(label)

        cv2.imshow('Camera Feed with YOLO', frame)

    def closeEvent(self, event):
        self.CLIport.write(('sensorStop\n').encode())
        self.CLIport.close()
        self.dataport.close()
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    CLIport, Dataport = serialConfig(configFileName)
    configParameters = parseConfigFile(configFileName)
    app = QtGui.QApplication([])
    win = ApplicationWindow(CLIport, Dataport, configParameters)
    win.show()
    timer = QtCore.QTimer()
    timer.timeout.connect(win.update)
    timer.start(33)  # ~30 fps
    sys.exit(app.exec_())
# Radar-Camera Fusion Blind Spot Detection System

## Overview
This project implements a **Radar-Camera Fusion Blind Spot Detection (BSD)** system using a combination of radar data and camera-based object detection with YOLOv8. The system processes radar data from a mmWave radar sensor, performs object detection on camera frames using the Ultralytics YOLOv8 model, and fuses the data to estimate distances and detect objects in predefined blind spot zones. The results are visualized in real-time using PyQtGraph, and radar points are logged to a CSV file for analysis.

## Features
- **Radar Data Processing**: Reads and parses mmWave radar data via serial communication.
- **Camera Object Detection**: Uses YOLOv8 for real-time detection of humans and vehicles.
- **Data Fusion**: Combines radar and camera distance estimates using dynamic weighting based on confidence scores.
- **Blind Spot Detection**: Identifies objects in predefined left and right blind spot zones.
- **Visualization**: Displays radar points and camera feed with bounding boxes and distance annotations.
- **Data Logging**: Saves radar and fused data to a CSV file (`radar_points.csv`).
- **Optional Audio Alerts**: Supports audio alerts for blind spot detections (requires `pygame` and an `alert.wav` file, currently commented out).

## Prerequisites
### Hardware
- **mmWave Radar Sensor**: Configured to output data via UART (e.g., TI AWR1642 or similar).
- **Webcam**: Compatible with OpenCV (tested with 1280x720 resolution).
- **Serial Ports**: Two COM ports (e.g., COM3 for data, COM4 for CLI) for radar communication.

### Software
- **Python 3.8+**
- **Dependencies**:
  ```bash
  pip install numpy pyqtgraph opencv-python ultralytics serial
  ```
  - Optional: `pygame` for audio alerts (`pip install pygame`).
- **YOLOv8 Model**: The `yolov8n.pt` model is automatically downloaded by the Ultralytics library if not present.
- **Configuration File**: A radar configuration file (e.g., `tes_config_1.cfg`) specifying radar parameters.

### Operating System
- Tested on Windows. Adjust serial port names (e.g., `COM3`, `COM4`) for other operating systems (e.g., `/dev/ttyUSB0` on Linux).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/radar-camera-fusion-bsd.git
   cd radar-camera-fusion-bsd
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Configuration File**:
   - Place your radar configuration file (e.g., `tes_config_1.cfg`) in the project directory or update the `configFileName` path in the script.

4. **Optional Audio Alerts**:
   - Uncomment the `pygame` code in the script.
   - Ensure an `alert.wav` file is present in the project directory.

## Usage
1. **Update Serial Port Settings**:
   - Modify the `serialConfig` function in `main.py` to match your radar's COM ports:
     ```python
     CLIport = serial.Serial('COM4', 115200)
     Dataport = serial.Serial('COM3', 921600)
     ```

2. **Run the Script**:
   ```bash
   python main.py
   ```

3. **Operation**:
   - A PyQtGraph window displays the radar points and blind spot alerts.
   - A separate OpenCV window shows the camera feed with YOLO bounding boxes and fused distance annotations.
   - Radar and fused data are logged to `radar_points.csv`.

4. **Calibration**:
   - Adjust the following parameters in the script based on your camera and radar setup:
     - `pixel_per_meter`: Pixels per meter for radar-to-image projection.
     - `offset_x`, `offset_y`: Image center for projection.
     - `focal_length`: Camera focal length in pixels.
     - `known_width`: Average width of detected objects (e.g., 0.5m for humans).
     - `blind_spots`: Coordinates of left and right blind spot zones.

## File Structure
- `main.py`: Main script implementing the radar-camera fusion system.
- `tes_config_1.cfg`: Example radar configuration file (update path as needed).
- `radar_points.csv`: Output CSV file for logged radar and fused data.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.
- (Optional) `alert.wav`: Audio file for blind spot alerts.

## Output
- **Visualization**:
  - PyQtGraph window: Radar points as red dots with distance labels, blind spot alerts in red text.
  - OpenCV window: Camera feed with red bounding boxes for humans, green for vehicles, and fused distance labels.
- **CSV Log** (`radar_points.csv`):
  - Columns: `Timestamp`, `FrameNumber`, `X (m)`, `Y (m)`, `Range (m)`, `Doppler (m/s)`, `PeakVal`, `Camera_Dist (m)`, `Fused_Dist (m)`.

## Notes
- **Synchronization**: The script logs the time difference between radar and camera frames to ensure synchronization. Adjust the `timer.start(33)` value if needed to match your radar frame rate.
- **Performance**: YOLOv8 detection time is printed to the console for monitoring. Use a GPU for faster inference if available.
- **Calibration**: Accurate calibration of `pixel_per_meter`, `focal_length`, and `known_width` is critical for precise distance estimation.
- **Blind Spot Zones**: Modify `blind_spots` dictionary to match your vehicle's blind spot regions.

## Troubleshooting
- **Serial Port Errors**: Verify COM port numbers and ensure no other applications are using them.
- **Camera Issues**: Check webcam compatibility and resolution settings.
- **YOLO Model Errors**: Ensure internet access for initial model download or place `yolov8n.pt` in the project directory.
- **Performance Lag**: Reduce camera resolution or increase `timer.start()` interval.

## Future Improvements
- Add support for multiple cameras.
- Implement advanced data fusion algorithms (e.g., Kalman filter).
- Enhance blind spot zone configuration with dynamic adjustments.
- Integrate with a vehicle CAN bus for real-time alerts.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **Ultralytics YOLOv8**: For object detection capabilities.
- **PyQtGraph**: For real-time visualization.
- **Texas Instruments**: For mmWave radar SDK and configuration examples.
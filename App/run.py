"""
Synth.Eye - Vision-Based Industrial AI Application
Main UI Application

This application provides a user interface for camera-based industrial inspection.
Layout is optimized for 4K monitors with 1920×1200px input image resolution.
"""

import sys
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
import os
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer, QRectF, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGraphicsView, QGraphicsScene, QSizePolicy
)
import os
import cv2
import numpy as np
import time
import yaml
from ultralytics import YOLO
import torch
# Custom Lib.:
#   ../Utilities/Image_Processing
import Utilities.Image_Processing
#   ../Utilities/General
import Utilities.General
#   ../Basler/Camera
from Basler.Camera import Basler_Cls
#   ../Calibration/Parameters
from Calibration.Parameters import Basler_Calib_Param_Str
# #   ../Parameters/Scene
# import Parameters.Scene
#   ../Measurement/Core
from Measurement.Core import Measure_Object_Cls
#   ../Measurement/Parameters
from Measurement.Parameters import Reference_Obj_Dimensions

# ============================================================================
# Color Constants
# ============================================================================

# Primary Colors
COLOR_PRIMARY = "#533bff"  # Main brand color (buttons, logo, accents)

# Text Colors
COLOR_TEXT_DARK = "#333"    # Main text color
COLOR_TEXT_MEDIUM = "#666"  # Secondary text color
COLOR_TEXT_LIGHT = "#999"   # Tertiary text color

# Background Colors
COLOR_BG_WHITE = "#ffffff"      # White background
COLOR_BG_LIGHT = "#f0f0f0"      # Light gray background
COLOR_BG_DISABLED = "#e0e0e0"   # Disabled button background

# Border Colors
COLOR_BORDER_LIGHT = "#ddd"     # Light border
COLOR_BORDER_MEDIUM = "#ccc"    # Medium border
COLOR_BORDER_GRID = "#e0e0e0"   # Grid lines

# Graph Colors
COLOR_GRAPH_TOTAL = "#2ecc71"   # Green - Total line
COLOR_GRAPH_OK = "#27ae60"      # Dark green - OK line
COLOR_GRAPH_NOK = "#e74c3c"     # Red - NOK line

# Status Colors
COLOR_STATUS_SUCCESS_BG = "#e8f5e9"  # Light green background
COLOR_STATUS_SUCCESS_TEXT = "#2e7d32" # Green text
COLOR_STATUS_SUCCESS_BORDER = "#4caf50" # Green border

"""
Description:
    Initialization of constants.
"""
# The name of the dataset, model, and color of the object bounding boxes.
CONST_CONFIG_MODEL_OBJ = {'Model': 'yolov8m_object_detection', 'Color': [(255, 165, 0), (0, 165, 255)]}
CONST_CONFIG_MODEL_DEFECT = {'Model': 'yolov8m_defect_detection', 'Color': [(80, 0, 255)]}
# The boundaries of an object (bounding box) determined using gen_obj_boundaries.py script.
CONST_OBJECT_BB_AREA = {'Min': 0.1, 'Max': 0.15}

# Repository root (parent of App/); avoids brittle os.getcwd() + 'Synth_Eye' splits on Synth_Eye_GAN.
project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Detect and assign the training device (GPU if available).
device_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device_id}")
if device_id.type == 'cuda':
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")

# Load hyperparameters and dataset configuration files.
with open(os.path.join(project_folder, 'Training', 'Args_Model_1.yaml'), 'r') as f:
    meta_args = yaml.safe_load(f)

# Load a pre-trained custom YOLO model.
model_object = YOLO(f"{project_folder}/YOLO/Model/Dataset_v2/{CONST_CONFIG_MODEL_OBJ['Model']}.pt")
model_defect = YOLO(f"{project_folder}/YOLO/Model/Dataset_v3/{CONST_CONFIG_MODEL_DEFECT['Model']}.pt")

# Custom camera configuration.
custom_cfg = {
    'exposure_time': 10000,
    'gain': 10,
    'balance_ratios': {'Red': 0.95, 'Green': 0.9, 'Blue': 1.2},
    'pixel_format': 'BayerRG8'
}

# ============================================================================
# Mock Camera Interface (for disabled camera option)
# ============================================================================

class MockCamera:
    """Mock camera class for testing UI without physical camera"""

    def __init__(self):
        self.connected = False
        self.resolution = None

    def connect(self):
        """Simulate camera connection"""
        self.connected = True
        self.resolution = (1920, 1200)
        return True

    def disconnect(self):
        """Simulate camera disconnection"""
        self.connected = False
        self.resolution = None

    def capture(self):
        """Simulate image capture - returns None for now"""
        if not self.connected:
            return None
        # Return a placeholder image (will be implemented later)
        return None

    def get_resolution(self):
        """Get camera resolution"""
        return self.resolution

# ============================================================================
# Aspect Ratio Label Widget
# ============================================================================

class AspectRatioLabel(QLabel):
    """QLabel that maintains a fixed aspect ratio"""

    def __init__(self, aspect_ratio=1.6, preferred_width=None, parent=None):  # 1920/1200 = 1.6
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.preferred_width = preferred_width
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        """Return size hint that maintains aspect ratio"""
        if self.width() > 0:
            width = self.width()
        elif self.preferred_width is not None:
            width = self.preferred_width
        else:
            width = 400  # Default fallback
        height = int(width / self.aspect_ratio)
        return QSize(width, height)

    def resizeEvent(self, event):
        """Maintain aspect ratio on resize"""
        super().resizeEvent(event)
        width = event.size().width()
        height = int(width / self.aspect_ratio)
        if self.height() != height:
            self.setFixedHeight(height)

# ============================================================================
# Productivity Graph Widget
# ============================================================================

class ProductivityGraph(QWidget):
    """Line graph widget showing Total, OK, and NOK counts over iterations"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use minimum size policy to allow responsive sizing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {COLOR_BG_WHITE}; border: 1px solid {COLOR_BORDER_MEDIUM};")

        # Data storage
        self.iterations = []
        self.total_data = []
        self.ok_data = []
        self.nok_data = []

    def add_data_point(self, total, ok, nok):
        """Add a new data point to the graph"""
        iteration = len(self.iterations) + 1
        self.iterations.append(iteration)
        self.total_data.append(total)
        self.ok_data.append(ok)
        self.nok_data.append(nok)
        self.update()  # Trigger repaint
        self.repaint()  # Force immediate repaint

    def clear_data(self):
        """Clear all graph data"""
        self.iterations = []
        self.total_data = []
        self.ok_data = []
        self.nok_data = []
        self.update()

    def paintEvent(self, event):
        """Draw the line graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()

        # Increased margins for better label readability
        left_margin = 60  # More space for Y-axis labels and tick values
        right_margin = 30
        top_margin = 30
        bottom_margin = 50  # More space for X-axis label

        graph_width = width - left_margin - right_margin
        graph_height = height - top_margin - bottom_margin
        graph_x = left_margin
        graph_y = top_margin

        # Draw background
        painter.fillRect(self.rect(), QColor(COLOR_BG_WHITE))

        if not self.iterations:
            # Draw placeholder text
            painter.setPen(QColor(COLOR_TEXT_LIGHT))
            # Use responsive font size based on widget height
            font_size = max(12, int(self.height() * 0.04))
            painter.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, font_size))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data available")
            return

        # Find max value for scaling
        max_val = max(max(self.total_data) if self.total_data else 1,
                     max(self.ok_data) if self.ok_data else 1,
                     max(self.nok_data) if self.nok_data else 1, 1)

        # Draw axes
        painter.setPen(QPen(QColor(COLOR_TEXT_DARK), 2))
        # X-axis
        painter.drawLine(graph_x, height - bottom_margin, graph_x + graph_width, height - bottom_margin)
        # Y-axis
        painter.drawLine(graph_x, graph_y, graph_x, height - bottom_margin)

        # Draw axis labels with larger, more readable font
        axis_font_size = min(max(14, int(self.height() * 0.04)), 20)  # Limit max size to 28pt
        label_font = QFont(SynthEyeApp.EUROSTYLE_FONT, axis_font_size, QFont.Bold)
        painter.setFont(label_font)
        painter.setPen(QColor(COLOR_TEXT_DARK))

        # X-axis label (larger and more space)
        x_label_rect = QRectF(graph_x, height - bottom_margin + 10, graph_width, 30)
        painter.drawText(x_label_rect, Qt.AlignCenter, "Iteration")

        # Y-axis label (larger and more space)
        painter.save()
        painter.translate(15, height / 2)
        painter.rotate(-90)
        y_label_rect = QRectF(-60, 0, 120, 30)
        painter.drawText(y_label_rect, Qt.AlignCenter, "Total")
        painter.restore()

        # Draw grid lines
        painter.setPen(QPen(QColor(COLOR_BORDER_GRID), 1))
        for i in range(5):
            y = int(round(graph_y + (graph_height * i / 4)))
            painter.drawLine(graph_x, y, graph_x + graph_width, y)

        # Draw data lines and points
        if len(self.iterations) >= 1:
            # Draw lines if we have more than one point
            if len(self.iterations) > 1:
                # OK line (darker green)
                painter.setPen(QPen(QColor(COLOR_GRAPH_OK), 2))
                for i in range(len(self.iterations) - 1):
                    x1 = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations))
                    y1 = height - bottom_margin - (graph_height * self.ok_data[i] / max_val)
                    x2 = graph_x + (graph_width * (self.iterations[i+1] - 1) / max(self.iterations))
                    y2 = height - bottom_margin - (graph_height * self.ok_data[i+1] / max_val)
                    painter.drawLine(x1, y1, x2, y2)

                # NOK line (red)
                painter.setPen(QPen(QColor(COLOR_GRAPH_NOK), 2))
                for i in range(len(self.iterations) - 1):
                    x1 = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations))
                    y1 = height - bottom_margin - (graph_height * self.nok_data[i] / max_val)
                    x2 = graph_x + (graph_width * (self.iterations[i+1] - 1) / max(self.iterations))
                    y2 = height - bottom_margin - (graph_height * self.nok_data[i+1] / max_val)
                    painter.drawLine(x1, y1, x2, y2)

            # Draw points for all data points (including single point)
            point_radius = 4
            # OK point (darker green)
            for i in range(len(self.iterations)):
                x = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations) if len(self.iterations) > 1 else 0)
                y = height - bottom_margin - (graph_height * self.ok_data[i] / max_val)
                painter.setPen(QPen(QColor(COLOR_GRAPH_OK), 2))
                painter.setBrush(QBrush(QColor(COLOR_GRAPH_OK)))
                painter.drawEllipse(x - point_radius, y - point_radius, point_radius * 2, point_radius * 2)

            # NOK point (red)
            for i in range(len(self.iterations)):
                x = graph_x + (graph_width * (self.iterations[i] - 1) / max(self.iterations) if len(self.iterations) > 1 else 0)
                y = height - bottom_margin - (graph_height * self.nok_data[i] / max_val)
                painter.setPen(QPen(QColor(COLOR_GRAPH_NOK), 2))
                painter.setBrush(QBrush(QColor(COLOR_GRAPH_NOK)))
                painter.drawEllipse(x - point_radius, y - point_radius, point_radius * 2, point_radius * 2)

        # Draw legend - responsive font size (if needed in future)
        legend_y = top_margin + 10
        legend_x = width - right_margin - 150
        legend_font_size = max(9, int(self.height() * 0.025))
        painter.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, legend_font_size))


# ============================================================================
# Main Application Window
# ============================================================================

class SynthEyeApp(QMainWindow):
    """Main application window for Synth.Eye"""

    EUROSTYLE_FONT = "Arial"  # Default, will be set in main()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synth.Eye - Vision-Based Industrial AI")

        # Camera state
        self.camera = MockCamera()
        self.captured_image = None
        self.measurement_image = None  # Clean pre-annotation image for measurement
        self.analysis_result = None
        self.detected_object_id = 0    # 0=front, 1=back; updated by ANALYZE

        self.Basler_Cam_Id_1 = None

        # Statistics
        self.total_scans = 0
        self.ok_count = 0
        self.nok_count = 0

        # Store responsive font sizes for use in dynamic updates
        self.title_font_size = None
        self.body_font_size = None

        # Setup UI
        self.init_ui()
        self.update_button_states()

    def init_ui(self):
        """Initialize the user interface"""
        # Set full-screen for 4K monitor
        self.showFullScreen()

        # Get screen size for responsive scaling
        screen = QApplication.primaryScreen().geometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Calculate responsive sizes based on screen dimensions
        # Use ~1% of screen width/height for margins and spacing
        margin = max(10, int(screen_width * 0.01))
        spacing = max(10, int(screen_width * 0.01))

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setStyleSheet("background-color: white;")

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(spacing)
        main_layout.setContentsMargins(margin, margin, margin, margin)

        # Left side: Camera View and Controls
        left_layout = QVBoxLayout()
        left_layout.setSpacing(spacing)

        # Calculate responsive font sizes (base on screen height)
        title_font_size = max(16, int(screen_height * 0.025))
        body_font_size = max(12, int(screen_height * 0.010))
        button_font_size = max(14, int(screen_height * 0.010))

        # Store for use in dynamic updates
        self.title_font_size = title_font_size
        self.body_font_size = body_font_size

        # Camera View
        camera_label = QLabel("Camera View")
        camera_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        camera_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        left_layout.addWidget(camera_label)

        # Calculate camera view size based on screen width (smaller, ~40% of available width)
        # Maintain 1920×1200 aspect ratio (1.6:1)
        camera_view_width = int(screen_width * 0.49)  # 40% of screen width

        # camera_view_width = int(screen_width * 0.8)
        camera_view_height = int(camera_view_width / 1.6)  # Maintain 1920×1200 aspect ratio

        # Store camera view dimensions for image resizing
        self.camera_view_width = camera_view_width
        self.camera_view_height = camera_view_height
        self.camera_view = AspectRatioLabel(aspect_ratio=1.6, preferred_width=camera_view_width)  # 1920/1200 = 1.6
        self.camera_view.setText("Camera Feed")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(f"""
            background-color: {COLOR_BG_LIGHT};
            color: {COLOR_TEXT_LIGHT};
            border: 2px solid {COLOR_BORDER_LIGHT};
            font-size: {body_font_size}pt;
        """)
        self.camera_view.setMaximumSize(camera_view_width, camera_view_height)
        left_layout.addWidget(self.camera_view)

        # Control Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(spacing)

        self.btn_connect = self.create_button("CONNECT", self.on_connect_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_capture = self.create_button("CAPTURE", self.on_capture_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_analyze = self.create_button("ANALYZE", self.on_analyze_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_measure = self.create_button("MEASURE", self.on_measure_clicked, COLOR_PRIMARY, button_font_size, screen_height)
        self.btn_clear = self.create_button("CLEAR", self.on_clear_clicked, COLOR_PRIMARY, button_font_size, screen_height)

        button_layout.addWidget(self.btn_connect)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_capture)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_measure)
        button_layout.addWidget(self.btn_clear)


        left_layout.addLayout(button_layout)

        left_layout.addStretch()

        # Bottom section: Logo and Status side by side
        bottom_layout = QHBoxLayout()
        bottom_layout.setAlignment(Qt.AlignBottom)

        # Branding section (left side)
        branding_layout = QVBoxLayout()
        branding_layout.setAlignment(Qt.AlignLeft | Qt.AlignBottom)

        # Logo image
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "Logo.png")
        logo_label = QLabel()
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo responsively (max 10% of screen height, maintain aspect ratio)
            max_logo_height = int(screen_height * 0.20)
            if pixmap.height() > max_logo_height:
                pixmap = pixmap.scaledToHeight(max_logo_height, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        branding_layout.addWidget(logo_label)

        bottom_layout.addLayout(branding_layout)
        # bottom_layout.addStretch()

        # Status section (right side)
        status_layout = QVBoxLayout()
        # status_layout.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        status_layout.setAlignment(Qt.AlignLeft| Qt.AlignTop)

        status_label = QLabel("Status")
        status_label.setAlignment(Qt.AlignLeft)
        status_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        status_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(status_label)

        self.status_camera = QLabel("Camera: Disconnected")
        self.status_camera.setAlignment(Qt.AlignLeft)
        self.status_camera.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.status_camera.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(self.status_camera)

        self.status_resolution = QLabel("Resolution: None")
        self.status_resolution.setAlignment(Qt.AlignLeft)
        self.status_resolution.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.status_resolution.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        status_layout.addWidget(self.status_resolution)

        bottom_layout.addLayout(status_layout)
        left_layout.addLayout(bottom_layout)

        # Right side: Logger and Graph
        right_layout = QVBoxLayout()
        right_layout.setSpacing(spacing)

        # System Logger
        logger_label = QLabel("System Logger")
        logger_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        logger_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        right_layout.addWidget(logger_label)

        self.logger = QTextEdit()
        self.logger.setReadOnly(True)
        self.logger.setStyleSheet(f"""
            background-color: {COLOR_BG_WHITE};
            color: {COLOR_TEXT_DARK};
            border: 2px solid {COLOR_BORDER_LIGHT};
            font-size: {body_font_size}pt;
            font-family: '{SynthEyeApp.EUROSTYLE_FONT}', monospace;
        """)
        right_layout.addWidget(self.logger, stretch=1)

        # Productivity Graph
        graph_label = QLabel("Productivity Graph")
        graph_label.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, title_font_size, QFont.Bold))
        graph_label.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        right_layout.addWidget(graph_label)

        self.productivity_graph = ProductivityGraph()
        right_layout.addWidget(self.productivity_graph, stretch=1)

        # Graph statistics text
        self.graph_stats = QLabel()
        self.graph_stats.setAlignment(Qt.AlignCenter)
        self.graph_stats.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, body_font_size))
        self.graph_stats.setStyleSheet(f"color: {COLOR_TEXT_DARK};")
        self.update_graph_stats()
        right_layout.addWidget(self.graph_stats)

        right_layout.addStretch()

        # Add layouts to main layout
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)

        # Log initial message
        self.log('Application Synt.Eye successfully started without any errors.')

    def create_button(self, text, callback, color=COLOR_PRIMARY, font_size=18, screen_height=1080):
        """Create a styled button"""
        btn = QPushButton(text)
        # Calculate responsive button height (about 6% of screen height, min 50px)
        btn_height = max(50, int(screen_height * 0.06))
        # Calculate responsive padding
        padding_v = max(10, int(screen_height * 0.015))
        padding_h = max(20, int(screen_height * 0.02))

        btn.setFont(QFont(SynthEyeApp.EUROSTYLE_FONT, font_size, QFont.Bold))
        btn.setMinimumHeight(btn_height)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: {padding_v}px {padding_h}px;
                font-size: {font_size}pt;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: {COLOR_BG_DISABLED};
                color: {COLOR_TEXT_LIGHT};
                opacity: 0.1;
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def darken_color(self, color):
        """Darken a hex color for hover effect"""
        # Simple darkening - convert hex to RGB, reduce values
        color = color.lstrip('#')
        r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r, g, b = max(0, r - 30), max(0, g - 30), max(0, b - 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def update_button_states(self):
        """Update button enabled/disabled states based on camera connection"""
        if not self.camera.connected:
            # Disconnected: only CONNECT enabled
            self.btn_connect.setEnabled(True)
            self.btn_capture.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_measure.setEnabled(False)
            self.btn_clear.setEnabled(False)
        else:
            # Connected: CONNECT becomes DISCONNECT, CAPTURE and CLEAR enabled
            self.btn_connect.setEnabled(True)
            self.btn_capture.setEnabled(True)
            self.btn_analyze.setEnabled(False)  # Only enabled after capture
            self.btn_measure.setEnabled(False)  # Only enabled after capture
            self.btn_clear.setEnabled(True)

    def on_connect_clicked(self):
        """Handle CONNECT/DISCONNECT button click"""
        if not self.camera.connected:
            # Connect
            if self.camera.connect():
                self.log(f'Scanning for available camera devices...')

                # Initialize and configure the Basler camera.
                self.Basler_Cam_Id_1 = Basler_Cls(config=custom_cfg)

                if self.Basler_Cam_Id_1.camera == None:
                    self.log('Failed to connect to camera. No device detected or connection attempt unsuccessful.')
                else:
                    self.btn_connect.setText("DISCONNECT")
                    self.log(f'Camera {self.Basler_Cam_Id_1.camera.GetDeviceInfo().GetModelName()} detected and connected at IP {self.Basler_Cam_Id_1.camera.GetDeviceInfo().GetIpAddress()}.')

                    self.update_status()
                    self.update_button_states()
        else:
            self.log(f'Disconnecting camera at IP {self.Basler_Cam_Id_1.camera.GetDeviceInfo().GetIpAddress()}...')

            # Disconnect
            self.camera.disconnect()
            self.btn_connect.setText("CONNECT")
            self.captured_image = None
            self.analysis_result = None
            self.camera_view.setText("Camera Feed")
            font_size = self.body_font_size if self.body_font_size else 16
            self.camera_view.setStyleSheet(f"""
                background-color: {COLOR_BG_LIGHT};
                color: {COLOR_TEXT_LIGHT};
                border: 2px solid {COLOR_BORDER_LIGHT};
                font-size: {font_size}pt;
            """)
            self.log(f'Camera {self.Basler_Cam_Id_1.camera.GetDeviceInfo().GetModelName()} at IP {self.Basler_Cam_Id_1.camera.GetDeviceInfo().GetIpAddress()} has been successfully disconnected.')

            # Release the classes.
            del self.Basler_Cam_Id_1

            self.update_status()
            self.update_button_states()

    def on_capture_clicked(self):
        """Handle CAPTURE button click"""
        if not self.camera.connected:
            return

        # Capture a single image.
        img_raw = self.Basler_Cam_Id_1.Capture()
        if img_raw is None:
            raise ValueError('[ERROR] No image captured!')

        # Initialize the class for custom image processing.
        Process_Image_Cls = Utilities.Image_Processing.Process_Image_Cls('real')

        # Apply the image processing pipeline.
        img_raw_processed = Process_Image_Cls.Apply(img_raw)

        # Undistort the image using camera calibration parameters.
        h, w = img_raw_processed.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(Basler_Calib_Param_Str.K, np.array(list(Basler_Calib_Param_Str.Coefficients.values()), dtype=np.float64),
                                                             (w, h), 1, (w, h))
        img_undistorted = cv2.undistort(img_raw_processed, Basler_Calib_Param_Str.K, np.array(list(Basler_Calib_Param_Str.Coefficients.values()), dtype=np.float64),
                                        None, new_camera_matrix)

        self.log('Capture button pressed. Performing camera scan...')
        self.captured_image = img_undistorted.copy()
        self.measurement_image = img_undistorted.copy()  # Preserve clean image for measurement

        # Get actual QLabel size and resize image to match
        label_width = self.camera_view.width()
        label_height = self.camera_view.height()
        
        # Resize image to match actual QLabel size
        img_resized = cv2.resize(self.captured_image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

        # Convert the resized image to QImage format
        if len(img_resized.shape) == 2:  # grayscale
            qimg = QImage(img_resized.data, label_width, label_height, label_width, QImage.Format.Format_Grayscale8)
        else:  # color image
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, label_width, label_height, 3 * label_width, QImage.Format.Format_RGB888)

        # Display the image in the QLabel
        pixmap = QPixmap.fromImage(qimg)
        self.camera_view.setPixmap(pixmap)
        self.camera_view.setScaledContents(False)  # Don't scale - image is already at correct size

        # Enable ANALYZE and MEASURE buttons
        self.btn_analyze.setEnabled(True)
        self.btn_measure.setEnabled(True)
        self.log('Image successfully captured with resolution 1920x1200 in RGB format.')

    def on_analyze_clicked(self):
        """Handle ANALYZE button click"""
        if self.measurement_image is None:
             return

        count = self.ok_count + self.nok_count

        self.log('Analyze button pressed. Performing Synth.Eye AI analysis of the RGB image...')

        # Perform prediction of the object on the test image set.
        results_object = model_object.predict(source=self.measurement_image, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

        # Initialize the variable to hold the processed image.
        processed_image = self.measurement_image.copy()

        # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
        if results_object[0].boxes.shape[0] >= 1:
            # Express the data from the prediction:
            #   ID name of the class, Bounding box in the YOLO format and Confidence.
            class_id = results_object[0].boxes.cls.cpu().numpy(); b_box = results_object[0].boxes.xywhn.cpu().numpy()
            conf = results_object[0].boxes.conf.cpu().numpy()

            for _, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
                # Get the area of the rectangle.
                A = b_box_i[2] * b_box_i[3]

                # If the calculated area of the object's bounding box is outside the limits, do not predict
                # the object.
                if A < CONST_OBJECT_BB_AREA['Min'] or A > CONST_OBJECT_BB_AREA['Max']:
                    continue

                # If the confidence of the prediction is less than 90%, do not predict the object.
                if conf_i < 0.9:
                    continue

                # Create a bounding box from the label data.
                Bounding_Box_Properties = {'Name': f'{int(class_id_i)}', 'Precision': f'{str(conf_i)[0:5]}',
                                           'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}

                # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in
                # the raw image.
                processed_image = Utilities.Image_Processing.Draw_Bounding_Box(processed_image, Bounding_Box_Properties, 'YOLO', CONST_CONFIG_MODEL_OBJ['Color'][int(class_id_i)],
                                                                            True, False)
                # Determine resolution of the processed image.
                img_h, img_w = self.measurement_image.shape[:2]
                Resolution = {'x': img_w, 'y': img_h}

                # Converts bounding box coordinates from YOLO format to absolute pixel coordinates.
                abs_coordinates_obj = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': b_box_i[0], 'y_c': b_box_i[1],
                                                                                    'width': b_box_i[2], 'height': b_box_i[3]},
                                                                                    Resolution)

                # Calculate object bounding box edges from center-based coordinates.
                obj_left = int(abs_coordinates_obj['x'] - abs_coordinates_obj['width'] / 2)
                obj_top = int(abs_coordinates_obj['y'] - abs_coordinates_obj['height'] / 2)
                obj_right = int(abs_coordinates_obj['x'] + abs_coordinates_obj['width'] / 2)
                obj_bottom = int(abs_coordinates_obj['y'] + abs_coordinates_obj['height'] / 2)

                # Crop the object region from the original image.
                cropped_image = self.measurement_image[obj_top:obj_bottom, obj_left:obj_right]

                self.analysis_result = 'OK'
                self.detected_object_id = int(class_id_i)

                if class_id_i == 0:
                    self.log(f'Detected front side of the metallic object on the image. Confidence: {str(conf_i*100.0)[0:5]} %.')
                else:
                    self.log(f'Detected back side of the metallic object on the image. Confidence: {str(conf_i*100.0)[0:5]} %.')

                # Perform defect detection only on specific object classes.
                #   Class ID (0) - Front side of the metalic object.
                if class_id_i == 0:
                    # Perform prediction of the defect on the test image set.
                    results_defect = model_defect.predict(source=cropped_image, device=device_id, imgsz=meta_args['imgsz'], conf=0.25, iou=0.5)

                    if results_defect[0].boxes.shape[0] >= 1:
                        # Express the data from the prediction of the defect.
                        defect_cls = results_defect[0].boxes.cls.cpu().numpy(); defect_b_box = results_defect[0].boxes.xywhn.cpu().numpy(); defect_conf = results_defect[0].boxes.conf.cpu().numpy()

                        for _, (d_class_i, d_b_box_i, d_conf_i) in enumerate(zip(defect_cls, defect_b_box, defect_conf)):
                            # If the confidence of the prediction is less than 80%, do not predict the object.
                            if d_conf_i < 0.8:
                                continue
                            self.analysis_result = 'NOK'

                            # Convert bounding box of the defect to absolute coordinates within cropped object.
                            abs_coordinates_defect = Utilities.General.YOLO_To_Absolute_Coordinates({'x_c': d_b_box_i[0], 'y_c': d_b_box_i[1],
                                                                                                     'width': d_b_box_i[2], 'height': d_b_box_i[3]},
                                                                                                    {'x': cropped_image.shape[1], 'y': cropped_image.shape[0]})

                            # Shift to original image.
                            abs_coordinates_defect['x'] += obj_left; abs_coordinates_defect['y'] += obj_top

                            # Generate YOLO-format label for original image.
                            yolo_coordinates_defect = Utilities.General.Absolute_Coordinates_To_YOLO(abs_coordinates_defect, Resolution)

                            # Create a bounding box from the label data of the defect.
                            Bounding_Box_Defect_Properties = {'Name': f'{int(d_class_i)}', 'Precision': f'{str(d_conf_i)[0:5]}',
                                                                'Data': yolo_coordinates_defect}

                            # Draw the bounding box of the defect with additional dependencies (name, precision, etc.) in
                            # the raw image.
                            processed_image = Utilities.Image_Processing.Draw_Bounding_Box(processed_image, Bounding_Box_Defect_Properties, 'YOLO', CONST_CONFIG_MODEL_DEFECT['Color'][int(class_id_i)],
                                                                                        True, False)

                            self.log(f'Detected defect in the form of fingerprint on the front side of the metalic object. Confidence: {str(d_conf_i*100.0)[0:5]} %.')

                if self.analysis_result == 'OK':
                    self.ok_count += 1
                    self.log(f'The result of the Synth.Eye AI analysis is {self.analysis_result}.')
                else:
                    self.nok_count += 1
                    self.log(f'The result of the Synth.Eye AI analysis is {self.analysis_result}. A defect has been detected.')

        if count == (self.ok_count + self.nok_count):
            # Determine resolution of the processed image.
            img_h, img_w = self.measurement_image.shape[:2]
            Resolution = {'x': img_w, 'y': img_h}

            self.log(f'The Synth.Eye AI analysis has been completed, but no object matching the rules was detected.')
        else:
            self.log(f'The Synth.Eye AI analysis has been completed.')

        self.captured_image = processed_image.copy()
        self.total_scans += 1

        # Get actual QLabel size and resize image to match
        label_width = self.camera_view.width()
        label_height = self.camera_view.height()
        
        # Resize image to match actual QLabel size
        img_resized = cv2.resize(self.captured_image, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

        # Convert to QImage and display in QLabel
        if len(img_resized.shape) == 2:
            qimg = QImage(img_resized.data, label_width, label_height, label_width, QImage.Format.Format_Grayscale8)
        else:  # color
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, label_width, label_height, 3 * label_width, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.camera_view.setPixmap(pixmap)
        self.camera_view.setScaledContents(False)  # Don't scale - image is already at correct size

        if self.measurement_image is not None:
            self.btn_measure.setEnabled(True)
        self.btn_analyze.setEnabled(False)

        # Update graph
        self.productivity_graph.add_data_point(
            self.total_scans, self.ok_count, self.nok_count
        )

        self.update_graph_stats()

    def on_clear_clicked(self):
        """Handle CLEAR button click"""
        self.log('Clear button pressed. Performing clear of all data...')

        # Clear graph data
        self.productivity_graph.clear_data()

        # Clear logger
        self.logger.clear()

        # Reset statistics
        self.total_scans = 0
        self.ok_count = 0
        self.nok_count = 0

        # Reset measurement state
        self.measurement_image = None
        self.detected_object_id = 0
        self.btn_measure.setEnabled(False)
        self.btn_analyze.setEnabled(False)

        # Update display
        self.update_graph_stats()
        self.log('All data successfully cleared.')

    def on_measure_clicked(self):
        """Handle MEASURE button click — runs object dimension measurement on the captured image"""
        if self.measurement_image is None:
            return

        side = 'front' if self.detected_object_id == 0 else 'back'
        self.log(f'Measure button pressed. Running dimension measurement on {side} side...')

        try:
            measurement = Measure_Object_Cls(
                image=self.measurement_image,
                object_id=self.detected_object_id,
                ref_obj_dim=Reference_Obj_Dimensions,
                tolerance=3.0,
                conversion_factor=Basler_Calib_Param_Str.Conversion_Factor,
                min_hole_diameter_mm=4.0,
                max_hole_diameter_mm=15.0
            )

            status, dimensions, angle, vis_img = measurement.Solve(draw_result=True)

            self.log(f'Measured height of the metallic object: {dimensions.Height:.2f} mm.')
            self.log(f'Measured width of the metallic object: {dimensions.Width:.2f} mm.')
            if self.detected_object_id == 0:
                self.log(f'Measured hole diameter of the front side: {dimensions.Hole_Diameter_Front:.2f} mm.')
            else:
                self.log(f'Measured hole diameter of the back side: {dimensions.Hole_Diameter_Back:.2f} mm.')
            self.log(f'Measured distance between hole centers: {dimensions.Hole_Center_Distance:.2f} mm.')
            self.log(f'Measured rotation angle of the object: {angle:.1f} deg.')
            result_str = 'PASS' if status else 'FAIL'
            tolerance_mm = 3.0
            if status:
                self.log(f'Measurement result: {result_str}. All dimensions are within the allowed tolerance of {tolerance_mm} mm.')
            else:
                self.log(f'Measurement result: {result_str}. One or more dimensions exceed the allowed tolerance of {tolerance_mm} mm.')

            display_img = vis_img if vis_img is not None else self.measurement_image.copy()
            self.captured_image = display_img.copy()

            # Display annotated measurement image
            label_width = self.camera_view.width()
            label_height = self.camera_view.height()
            img_resized = cv2.resize(display_img, (label_width, label_height), interpolation=cv2.INTER_LINEAR)

            if len(img_resized.shape) == 2:
                qimg = QImage(img_resized.data, label_width, label_height, label_width, QImage.Format.Format_Grayscale8)
            else:
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                qimg = QImage(img_rgb.data, label_width, label_height, 3 * label_width, QImage.Format.Format_RGB888)

            self.camera_view.setPixmap(QPixmap.fromImage(qimg))
            self.camera_view.setScaledContents(False)

        except Exception as e:
            self.log(f'[ERROR] Measurement failed: {e}')

        self.btn_measure.setEnabled(False)
        self.btn_analyze.setEnabled(True)

    def update_status(self):
        """Update status display"""
        if self.camera.connected:
            resolution = self.camera.get_resolution()
            if resolution:
                self.status_camera.setText("Camera: Connected")
                self.status_resolution.setText(f"Resolution: {resolution[0]}x{resolution[1]}")
            else:
                self.status_camera.setText("Camera: Connected")
                self.status_resolution.setText("Resolution: None")
        else:
            self.status_camera.setText("Camera: Disconnected")
            self.status_resolution.setText("Resolution: None")

    def update_graph_stats(self):
        """Update the statistics text below the graph"""
        text = (
            f"The total number of images analyzed is <b>{self.total_scans}</b>, "
            f"of which <span style='color: green;'>{self.ok_count}</span> were found to be OK "
            f"and <span style='color: red;'>{self.nok_count}</span> were found to be NOK."
        )
        self.graph_stats.setText(text)

    def log(self, message):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.logger.append(f"{timestamp} – {message}")

# ============================================================================
# Font Loading
# ============================================================================

def load_eurostyle_font():
    """Load Eurostyle font from App/fonts directory"""
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")

    # Try to load EuroStyle Normal.ttf first, then eurostile.TTF
    font_files = [
        os.path.join(font_dir, "EuroStyle Normal.ttf"),
        os.path.join(font_dir, "eurostile.TTF")
    ]

    font_family = None
    for font_file in font_files:
        if os.path.exists(font_file):
            font_id = QFontDatabase.addApplicationFont(font_file)
            if font_id != -1:
                font_families = QFontDatabase.applicationFontFamilies(font_id)
                if font_families:
                    font_family = font_families[0]
                    print(f"[INFO] Loaded font: {font_family} from {font_file}")
                    break

    if font_family is None:
        print("[WARNING] Could not load Eurostyle font, using default font")
        return "Arial"

    return font_family

# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Load Eurostyle font
    eurostyle_font = load_eurostyle_font()

    # Store font name globally for use in the application
    SynthEyeApp.EUROSTYLE_FONT = eurostyle_font

    window = SynthEyeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
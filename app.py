import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from huggingface_hub import from_pretrained_keras
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
                              QHBoxLayout, QSlider, QFileDialog, QFrame, QMessageBox, 
                              QTabWidget, QProgressBar, QGraphicsView, QGraphicsScene, 
                              QGraphicsEllipseItem, QGraphicsTextItem)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QPen
import pyqtgraph as pg
from pds4_tools import pds4_read

# Load the pre-trained low-light enhancement model
model = from_pretrained_keras("keras-io/low-light-image-enhancement")

def post_process(image, output):
    residuals = tf.split(output, num_or_size_splits=8, axis=-1)
    x = image
    for r in residuals:
        x = x + r * (tf.square(x) - x)
    return x

def low_light_enhancement(img, blend_ratio=0.5, illumination_angle=None, surface_temp=None):
    """
    Enhance the image considering low-light conditions. Additionally, adjusts based on illumination metadata.
    """
    img_rgb = Image.fromarray(img).convert('RGB')
    image = keras.preprocessing.image.img_to_array(img_rgb)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Adjust enhancement parameters based on illumination angle if available
    if illumination_angle is not None:
        print(f"Illumination Angle: {illumination_angle}")
        if illumination_angle > 80:  # Very close to shadow regions
            blend_ratio = 0.7  # Increase the blend ratio for better enhancement
        elif illumination_angle < 30:  # Strong light
            blend_ratio = 0.3  # Reduce blend ratio

    # Adjust based on surface temperature (if needed for future features)
    if surface_temp is not None:
        print(f"Surface Temperature: {surface_temp}")
        if surface_temp < 100:  # Cold region
            blend_ratio *= 1.1  # Slightly increase enhancement in cold regions

    output_image = model(image)
    enhanced_img = post_process(image, output_image)
    enhanced_img = tf.cast((enhanced_img[0] * 255), dtype=tf.uint8)
    enhanced_img_pil = Image.fromarray(enhanced_img.numpy())
    
    final_image = Image.blend(img_rgb, enhanced_img_pil, alpha=blend_ratio)
    return np.array(final_image)

def denoise_image(img):
    return cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(img)

    # Apply CLAHE to each channel
    clahe_channels = [clahe.apply(channel) for channel in channels]
    return cv2.merge(clahe_channels)

def load_pds4_image(pds4_file):
    data = pds4_read(pds4_file)
    print(data)
    img = data[0].data  # Access the image data
    metadata = data.label  # Access the associated metadata
    small_img = img[:1000, :]  # Extract a smaller portion for processing

    # Extract metadata details for illumination and surface properties
    illumination_angle = metadata.get('Observation_Angle', {}).get('Incidence_Angle', None)
    surface_temp = metadata.get('Surface_Temperature', None)

    img_pill = Image.fromarray(small_img)
    return img, metadata, illumination_angle, surface_temp

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.clip_limit = 2.0  # Default CLAHE clip limit
        self.tile_grid_size = (8, 8)  # Default CLAHE tile grid size
        self.dark_mode = False  # Dark mode flag
        self.annotating = False  # Flag for annotation mode
        self.initUI()

    def initUI(self):
        # Layouts
        main_layout = QHBoxLayout(self)
        sidebar_layout = QVBoxLayout()
        main_content_layout = QVBoxLayout()

        # Tab Widget for multiple pages
        self.tabs = QTabWidget(self)
        self.image_processing_page = QWidget()
        self.image_viewing_page = QWidget()
        self.comparison_page = QWidget()  # New comparison page tab
        self.annotation_page = QWidget()  # New annotation page
        self.tabs.addTab(self.image_processing_page, "Image Processing")
        self.tabs.addTab(self.image_viewing_page, "Image Viewing")
        self.tabs.addTab(self.comparison_page, "Before & After")
        self.tabs.addTab(self.annotation_page, "Annotate Image")  # Add annotation tab

        # Initialize Image Processing Page
        self.init_image_processing_page(sidebar_layout, main_content_layout)

        # Initialize Image Viewing Page
        self.init_image_viewing_page()

        # Initialize Comparison Page
        self.init_comparison_page()

        # Initialize Annotation Page
        self.init_annotation_page()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        # Set window title and size
        self.setWindowTitle('Image Processing with Space-Tech')
        self.setGeometry(100, 100, 800, 600)

        # Theme Toggle Button
        self.theme_button = QPushButton("Toggle Dark/Light Mode", self)
        self.theme_button.setStyleSheet("color: black;")  # Change text color here
        self.theme_button.clicked.connect(self.toggle_theme)

        # Create a layout for the theme button to place it in the top right corner
        theme_button_layout = QHBoxLayout()
        theme_button_layout.addStretch(1)  # Add stretchable space before the button
        theme_button_layout.addWidget(self.theme_button)

        # Add theme button layout to the main content layout
        main_content_layout.addLayout(theme_button_layout)

        # Add Tooltips after all components are initialized
        self.add_tooltips()

    def init_image_processing_page(self, sidebar_layout, main_content_layout):
        # Sidebar
        sidebar = QFrame(self)
        sidebar.setStyleSheet("background-color: #FCFCFC;")
        sidebar.setFixedWidth(150)

        # Sidebar Label
        sidebar_label = QLabel("Upload your image", self)
        sidebar_label.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        sidebar_layout.addWidget(sidebar_label)
        sidebar_layout.addStretch(1)

        # Developed by Label
        dev_label = QLabel("Developed by\nSpace-Tech", self)
        dev_label.setStyleSheet("font-size: 12px; font-weight: bold; color: black;")
        sidebar_layout.addWidget(dev_label)
        sidebar_layout.addStretch(1)

        sidebar.setLayout(sidebar_layout)
        main_content_layout.addWidget(sidebar)

        # Image display label
        self.image_label = QLabel("Upload and process an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_content_layout.addWidget(self.image_label)

        # Sliders
        self.blend_ratio_slider = QSlider(Qt.Horizontal)
        self.blend_ratio_slider.setRange(0, 100)
        self.blend_ratio_slider.setValue(50)
        main_content_layout.addWidget(QLabel("Blend Ratio (Low-Light Enhancement):"))
        main_content_layout.addWidget(self.blend_ratio_slider)

        # CLAHE Parameters Section
        layout = QVBoxLayout()
        
        # Clip Limit Slider
        self.clip_limit_slider = QSlider(Qt.Horizontal)
        self.clip_limit_slider.setRange(1, 100)  # Adjusting range for clip limit
        self.clip_limit_slider.setValue(int(self.clip_limit * 10))  # Scale to 0.1 increments
        layout.addWidget(QLabel("CLAHE Clip Limit:"))
        layout.addWidget(self.clip_limit_slider)

        # Tile Grid Size Slider
        self.tile_grid_size_slider = QSlider(Qt.Horizontal)
        self.tile_grid_size_slider.setRange(1, 16)  # Maximum size for tile grid
        self.tile_grid_size_slider.setValue(self.tile_grid_size[0])  # Default tile size
        layout.addWidget(QLabel("CLAHE Tile Grid Size:"))
        layout.addWidget(self.tile_grid_size_slider)

        main_content_layout.addLayout(layout)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)  # 100% is the max value
        main_content_layout.addWidget(self.progress_bar)

        # Reset Button
        self.reset_button = QPushButton("Reset Image", self)
        self.reset_button.clicked.connect(self.reset_image)
        main_content_layout.addWidget(self.reset_button)

        self.upload_pds4_button = QPushButton("Upload PDS4 Image", self)  # Ensure this is aligned with the above line
        self.upload_pds4_button.clicked.connect(self.upload_and_process_pds4_image)
        main_content_layout.addWidget(self.upload_pds4_button)


        # Buttons
        self.upload_button = QPushButton("Upload and Enhance Image", self)
        self.upload_button.clicked.connect(self.upload_and_process_image)
        main_content_layout.addWidget(self.upload_button)

        self.image_processing_page.setLayout(main_content_layout)

    def init_image_viewing_page(self):
        # Image Viewing Page
        layout = QVBoxLayout()
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view)
        self.image_viewing_page.setLayout(layout)

    def init_comparison_page(self):
        # Comparison Page
        layout = QHBoxLayout()
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.enhanced_image_label = QLabel("Enhanced Image")
        self.enhanced_image_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.original_image_label)
        layout.addWidget(self.enhanced_image_label)
        
        self.comparison_page.setLayout(layout)

    def init_annotation_page(self):
        # Annotation Page
        layout = QVBoxLayout()
        self.annotation_view = QGraphicsView()
        self.annotation_scene = QGraphicsScene()
        self.annotation_view.setScene(self.annotation_scene)
        layout.addWidget(self.annotation_view)

        # Load Image Button
        self.load_image_button = QPushButton("Load Image for Annotation")
        self.load_image_button.clicked.connect(self.load_image_for_annotation)
        layout.addWidget(self.load_image_button)

        # Toggle Annotation Mode Button
        self.toggle_annotation_button = QPushButton("Start Annotation")
        self.toggle_annotation_button.clicked.connect(self.toggle_annotation_mode)
        layout.addWidget(self.toggle_annotation_button)

        # Save Annotated Image Button
        self.save_annotation_button = QPushButton("Save Annotated Image")
        self.save_annotation_button.clicked.connect(self.save_annotated_image)
        layout.addWidget(self.save_annotation_button)

        self.annotation_page.setLayout(layout)

        # Set mouse event for annotation
        self.annotation_view.setMouseTracking(True)

    def load_image_for_annotation(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_path:
            self.annotation_scene.clear()  # Clear previous scene
            img = Image.open(file_path).convert("RGBA")
            width, height = img.size
            self.image_for_annotation = np.array(img)

            # Display the loaded image
            self.image_item = self.annotation_scene.addPixmap(QPixmap(file_path))
            self.annotation_view.setSceneRect(0, 0, width, height)
            self.annotation_view.setFixedSize(width, height)

    def toggle_annotation_mode(self):
        self.annotating = not self.annotating
        self.toggle_annotation_button.setText("Stop Annotation" if self.annotating else "Start Annotation")
        if self.annotating:
            self.annotation_view.mousePressEvent = self.start_annotation
            self.annotation_view.mouseMoveEvent = self.draw_annotation
            self.annotation_view.mouseReleaseEvent = self.stop_annotation
        else:
            self.annotation_view.mousePressEvent = None
            self.annotation_view.mouseMoveEvent = None
            self.annotation_view.mouseReleaseEvent = None

    def start_annotation(self, event):
        if self.annotating:
            self.last_point = QPoint(event.pos())
            self.pen = QPen(QColor(255, 0, 0), 2, Qt.SolidLine)
            self.annotation_scene.addItem(QGraphicsEllipseItem(self.last_point.x(), self.last_point.y(), 5, 5))
            self.annotation_scene.update()

    def draw_annotation(self, event):
        if self.annotating:
            current_point = QPoint(event.pos())
            if (current_point - self.last_point).manhattanLength() > 0:
                self.annotation_scene.addLine(self.last_point.x(), self.last_point.y(), current_point.x(), current_point.y(), self.pen)
                self.last_point = current_point

    def stop_annotation(self, event):
        if self.annotating:
            current_point = QPoint(event.pos())
            self.annotation_scene.addLine(self.last_point.x(), self.last_point.y(), current_point.x(), current_point.y(), self.pen)
            self.last_point = None

    def save_annotated_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotated Image", "", "PNG Files (*.png)")
        if file_path:
            image_for_saving = self.image_for_annotation.copy()
            image_for_saving = Image.fromarray(image_for_saving)

            # Draw annotations
            painter = QPainter(image_for_saving)
            for item in self.annotation_scene.items():
                if isinstance(item, QGraphicsEllipseItem):
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.drawEllipse(item.rect())
                elif isinstance(item, QGraphicsLineItem):
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    painter.drawLine(item.line())

            painter.end()
            image_for_saving.save(file_path)
            QMessageBox.information(self, "Success", f"Annotated image saved as '{file_path}'")

    def reset_image(self):
        if hasattr(self, 'original_image'):
            self.image_view.setImage(self.original_image.transpose(2, 0, 1))
            self.image_label.setText("Upload and process an image")
            self.original_image_label.clear()
            self.enhanced_image_label.clear()

    def upload_and_process_pds4_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload PDS4 Image", "", "PDS4 Files (*.xml)")
        if file_path:
            try:
                img, metadata, illumination_angle, surface_temp = load_pds4_image(file_path)
                print(metadata)
                print(f"Image data shape: {img.shape}, dtype: {img.dtype}")
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                    img_pil = Image.fromarray(img)
                    qimage = QImage(img_pil.tobytes(), img_pil.width, img_pil.height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimage)
                    self.image_label.setPixmap(pixmap)
                    
                    # Enhance image using metadata for illumination
                    enhanced_img = low_light_enhancement(img, illumination_angle=illumination_angle, surface_temp=surface_temp)

                    img_pil.save("pds4_image_enhanced.png")
                    QMessageBox.information(self, "Success", "PDS4 image saved as 'pds4_image_enhanced.png'")
            except Exception as e:
                print(f"An error occurred: {e}")  # Print the exception for debugging
                QMessageBox.critical(self, "Error", f"Failed to process PDS4 image: {e}")

    def upload_and_process_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg)")
        if file_path:
            try:
                self.upload_button.setEnabled(False)
                self.reset_button.setEnabled(False)
                self.blend_ratio_slider.setEnabled(False)
                self.clip_limit_slider.setEnabled(False)
                self.tile_grid_size_slider.setEnabled(False)
                self.progress_bar.setValue(0)

                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("Unable to load image. Please try a different file.")

                self.progress_bar.setValue(20)

                blend_ratio = self.blend_ratio_slider.value() / 100.0
                enhanced_img = low_light_enhancement(img, blend_ratio)
                self.progress_bar.setValue(60)

                denoised_img = denoise_image(enhanced_img)
                self.progress_bar.setValue(80)

                self.clip_limit = self.clip_limit_slider.value() / 10.0
                self.tile_grid_size = (self.tile_grid_size_slider.value(), self.tile_grid_size_slider.value())

                final_image = apply_clahe(denoised_img, clip_limit=self.clip_limit, tile_grid_size=self.tile_grid_size)
                self.progress_bar.setValue(100)

                # Store and display images
                self.original_image = img  # Store original image for reset
                self.image_view.setImage(final_image.transpose(2, 0, 1))  # Display enhanced image
                self.original_image_label.setPixmap(QPixmap.fromImage(QImage(self.original_image.data,
                                                                             self.original_image.shape[1],
                                                                             self.original_image.shape[0],
                                                                             QImage.Format_RGB888)))
                self.enhanced_image_label.setPixmap(QPixmap.fromImage(QImage(final_image.data,
                                                                             final_image.shape[1],
                                                                             final_image.shape[0],
                                                                             QImage.Format_RGB888)))

                # Allow user to choose where to save the processed image
                save_path, _ = QFileDialog.getSaveFileName(self, "Save Enhanced Image", "enhanced_image_clahe.png", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
                if save_path:
                    Image.fromarray(final_image).save(save_path)
                    QMessageBox.information(self, "Success", f"Image saved as '{save_path}'")

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
            finally:
                # Re-enable controls
                self.upload_button.setEnabled(True)
                self.reset_button.setEnabled(True)
                self.blend_ratio_slider.setEnabled(True)
                self.clip_limit_slider.setEnabled(True)
                self.tile_grid_size_slider.setEnabled(True)

    def toggle_theme(self):
        if self.dark_mode:
            self.setStyleSheet("background-color: white; color: black;")
            self.dark_mode = False
        else:
            self.setStyleSheet("background-color: #2D2D2D; color: white;")
            self.dark_mode = True

    def add_tooltips(self):
        self.blend_ratio_slider.setToolTip("Adjust the blend ratio for low-light enhancement.")
        self.clip_limit_slider.setToolTip("Adjust the clip limit for CLAHE.")
        self.tile_grid_size_slider.setToolTip("Adjust the tile grid size for CLAHE.")
        self.upload_button.setToolTip("Click to upload an image for processing.")
        self.reset_button.setToolTip("Click to reset the processed image to its original state.")
        self.load_image_button.setToolTip("Load an image to annotate.")
        self.toggle_annotation_button.setToolTip("Click to start or stop annotation mode.")
        self.save_annotation_button.setToolTip("Save the annotated image.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())
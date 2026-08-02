# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
 
try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None

class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.
    It publishes the detected QR code payload on the `/qr_detection` topic.
    """
    def __init__(self):
        super().__init__('qr_detector')

        self.loss_grace_period = 0.3
        self.pyzbar_cooldown = 0.3
        self.now = time.monotonic()

        self.last_published = None      # last payload string we actually published (None or "RESET")
        self.last_seen_data = None      # last successfully decoded payload (regardless of publish)
        self.last_seen_time = None      # time.monotonic() of the last successful detection
        self.last_pyzbar_attempt = 0.0  # time.monotonic() of the last pyzbar fallback attempt
 
        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for QR code detection results.
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.get_logger().info("QR Detector Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        # Convert compressed image message to OpenCV format
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return

        self.now= time.monotonic()
        qr_data = self.detect_qr_code(image)


        if qr_data is not None:
            # Publish the decoded QR payload

            self.last_seen_data = qr_data
            self.last_seen_time = self.now
            self._maybe_publish(qr_data)
 

        #    msg = String()
        #    msg.data = qr_data
        #    self.publisher_qr.publish(msg)
        #    self.get_logger().info(f"Published QR Data: {qr_data}")
        else:
            self._check_for_loss()

    def _maybe_publish(self, qr_data):
        """Publish only if this payload differs from the last one we published."""
        if qr_data == self.last_published:
            return
        self._publish(qr_data)

    def _check_for_loss(self):
        """If a tracked code hasn't been seen for longer than the grace period,
        publish a single RESET and clear tracking state."""
        if self.last_seen_time is None:
            return  # nothing was ever tracked, nothing to reset
 
        if self.last_published == "RESET":
            return  # already reset, don't spam
 
        if (self.now - self.last_seen_time) >= self.loss_grace_period:
            self._publish("RESET")
            self.last_seen_data = None
            self.last_seen_time = None

    def _publish(self, payload):
        msg = String()
        msg.data = payload
        self.publisher_qr.publish(msg)
        self.last_published = payload
        self.get_logger().info(f"Published: {payload}")

    def detect_qr_code(self, image):
        """
        Detect and decode QR code in the image.
        
        OPTIMIZATION HINTS:
        - OpenCV has a built-in QR Code detector: cv2.QRCodeDetector().
        - Alternatively, you can use Pyzbar (a popular and robust library for barcode/QR code reading).
        - Ensure to pre-process the image (e.g., convert to grayscale, thresholding, cropping to region 
          of interest where the building/QR board is expected to appear) to improve speed and reliability.
        """
        # --- Method 1: Using OpenCV Built-in QR Detector ---
        try:
            detector = cv2.QRCodeDetector()
            data, bbox, straight_qrcode = detector.detectAndDecode(image)
            if bbox is not None and data != "":
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detection failed: {e}")

        # --- Method 2: Placeholder for Pyzbar ---
        if pyzbar is not None and (self.now - self.last_pyzbar_attempt) >= self.pyzbar_cooldown:
            self.last_pyzbar_attempt = self.now
            try:
                decoded_objects = pyzbar.decode(image)
                for obj in decoded_objects:
                    return obj.data.decode('utf-8')
            except Exception as e:
                self.get_logger().debug(f"pyzbar QR Detection failed: {e}")

        return None

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
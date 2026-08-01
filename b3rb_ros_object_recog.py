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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

from ultralytics import YOLO

# HINT: TensorFlow/Keras can be heavy and might not be installed by default.
# We wrap the import in a try-except block so the node runs even if TensorFlow is missing.
# Install it using: pip install tensorflow
try:
    import tensorflow as tf
except ImportError:
    tf = None

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign boards.
    It publishes the detected sign type/labels on the `/sign_board_detection` topic.
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        self.raw_model_path = "~/cognipilot/cranium/src/b3rb_ros_line_follower/b3rb_ros_line_follower/b3rb_ros_line_follower/runs/detect/NPX_SIGN_CLASSIFIER/yolo11n_sign_classifier-3/weights/best.pt"

        self.model = YOLO(os.path.expanduser(self.raw_model_path))
        if(self.model is None): self.get_logger().info("Model couldn't be loaded!!!")

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""

        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if not self.is_sign_visible(image):
            sign_detected = self.classify_sign(image)

            if sign_detected is not None:
                msg = String()
                msg.data = sign_detected
                self.publisher_sign.publish(msg)
                self.get_logger().info(f"Detected Sign Board: {sign_detected}")

        cv2.imshow("Camera feed", image)
        cv2.waitKey(1)

    def classify_sign(self, image):

        if self.model is not None:
            try:
                predictions = self.model.predict(image, conf=0.3, show=False)[0]
                letters = []
                directions = []
                mappings = {}

                for box in predictions.boxes:
                    clsid = int(box.cls[0])
                    label = self.model.names[clsid]

                    x_centre = float(box.xywhn[0][0])

                    if(label in ["Left", "Right", "Straight"]): directions.append((label, x_centre))
                    else: letters.append((label, x_centre))

                if (not letters or not directions): return None

                for location in letters:
                    min_dis = abs(location[1] - directions[0][1])

                    for arrow in directions:
                        if(abs(arrow[1]-location[1]) <= min_dis):
                            min_dis = abs(arrow[1]-location[1])
                            mappings[location[0]] = arrow[0]

                return str(mappings)

            except Exception as e:
                self.get_logger().debug(f"Inference failed: {e}")

        return None

    def is_sign_visible(self, image)->bool:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        masked_image = cv2.inRange(hsv_image, lower_green, upper_green)
        contours,_ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if(contours):
            largest_contour = max(contours, key = cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            x, y, w, h = cv2.boundingRect(largest_contour)

            if(y < 5 or x+w+5 > image.shape[1] or x<5): return False
            elif(area > 100 and area < 5000): return True

        return False

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

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

# Known decoy/fake QR codes on the track - any of these should be treated as
# if nothing was decoded at all, and never published.
IGNORED_PAYLOADS = {
    '{LOC: FAKE_HOSPITAL_1}',
    '{LOC: FAKE_HOSPITAL_2}',
    '{LOC: FAKE_HOSPITAL_3}',
}

try:
    from pyzbar import pyzbar
except ImportError:
    pyzbar = None


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.

    Publishing behaviour (to avoid spamming /qr_detection and flapping):
      - A payload is published the moment it's decoded, as soon as it
        differs from the last published payload - even a single successful
        scan (e.g. one good frame while approaching a code from an angle)
        is enough to trigger a publish.
      - If the code then goes missing for `loss_grace_period` seconds
        *continuously* (any hit in between resets that timer), a single
        "RESET" message is published and tracking state is cleared. This is
        what prevents flapping: a code that flickers in and out faster than
        the grace period never triggers a RESET in between.
      - Repeated frames of the same code, or repeated no-detect frames
        after RESET has already been sent, produce no publishes at all.
      - Payloads in IGNORED_PAYLOADS (e.g. known decoy/fake codes) are
        never published - they're treated exactly as if no code was seen
        at all in that frame, and can still count toward a loss/RESET if
        one was already being tracked.
    """

    def __init__(self):
        super().__init__('qr_detector')

        # --- Tunables ---
        self.declare_parameter('loss_grace_period', 1.0)    # seconds of *continuous* misses before RESET
        self.declare_parameter('dedup_window', 2.0)         # seconds after a RESET during which the same code reappearing is treated as a continuation, not a new event
        self.loss_grace_period = self.get_parameter('loss_grace_period').value
        self.dedup_window = self.get_parameter('dedup_window').value

        # --- State ---
        self.last_published = None      # last payload string we actually published (None or "RESET")
        self.last_seen_time = None      # time.monotonic() of the most recent successful detection
        self.first_miss_time = None     # time.monotonic() of the first miss since the last successful detection
        self.reset_payload = None       # payload that was active when the last RESET was published
        self.reset_time = None          # time.monotonic() of the last RESET publish

        self.detector = cv2.QRCodeDetector()

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10)

        self.get_logger().info("QR Detector Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to detect QR codes."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        now = time.monotonic()
        qr_data = self.detect_qr_code(image)

        if qr_data:
            self.last_seen_time = now
            self.first_miss_time = None  # any hit clears an in-progress miss streak

            if qr_data == self.reset_payload and self.reset_time is not None \
                    and (now - self.reset_time) < self.dedup_window:
                # Same code that was just reset reappeared quickly - treat this as
                # a continuation of the same encounter, not a new event. Resume
                # tracking silently so future genuine changes still publish normally.
                self.last_published = qr_data
                self.reset_payload = None
                self.reset_time = None
            elif qr_data != self.last_published:
                self._publish(qr_data)
        else:
            self._check_for_loss(now)

    def _check_for_loss(self, now):
        """If a code hasn't been seen for longer than the grace period *continuously*
        since the first miss, publish a single RESET and clear tracking state.
        A single missed frame does not immediately reset anything - it only starts
        (or continues) a timer that any subsequent hit will cancel."""
        if self.last_seen_time is None:
            return  # nothing was ever tracked, nothing to reset

        if self.last_published in (None, "RESET"):
            return  # already reset (or never published), nothing to lose

        if self.first_miss_time is None:
            self.first_miss_time = now
            return

        if (now - self.first_miss_time) >= self.loss_grace_period:
            self.reset_payload = self.last_published
            self.reset_time = now
            self._publish("RESET")
            self.last_seen_time = None
            self.first_miss_time = None

    def _publish(self, payload):
        msg = String()
        msg.data = payload
        self.publisher_qr.publish(msg)
        self.last_published = payload
        self.get_logger().info(f"Published: {payload}")

    def detect_qr_code(self, image):
        """
        Detect and decode a QR code in the image, escalating through
        preprocessing stages only as needed (cheapest first):

          Stage 1 - raw frame:        try as-is, works for the easy/common case.
          Stage 2 - CLAHE + upscale:  evens out lighting/glare and gives more
                                       pixels-per-module for small/far codes.
          Stage 3 - adaptive thresh:  clean binarization for borderline
                                       contrast/mild blur cases.

        Each stage tries pyzbar first (primary, more reliable), then OpenCV
        as a fallback within that stage.
        """
        # --- Stage 1: raw frame ---
        data = self._try_decode(image)
        if data:
            return data

        # --- Stage 2: CLAHE (adaptive histogram equalization) + 2x upscale ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        upscaled = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        data = self._try_decode(upscaled)
        if data:
            return data

        # --- Stage 3: adaptive threshold on top of the enhanced/upscaled image ---
        binarized = cv2.adaptiveThreshold(
            upscaled, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=5)

        data = self._try_decode(binarized)
        if data:
            return data

        return None

    def _try_decode(self, image):
        """Attempt pyzbar first (primary), then OpenCV (fallback), on a single
        image variant. Works with both BGR and single-channel grayscale input -
        both pyzbar and cv2.QRCodeDetector accept either. Payloads in
        IGNORED_PAYLOADS (e.g. known decoy/fake codes) are treated as if
        nothing was decoded at all."""
        if pyzbar is not None:
            try:
                decoded_objects = pyzbar.decode(image)
                if decoded_objects:
                    data = decoded_objects[0].data.decode('utf-8')
                    if data not in IGNORED_PAYLOADS:
                        return data
            except Exception as e:
                self.get_logger().debug(f"pyzbar QR Detection failed: {e}")

        try:
            data, bbox, _ = self.detector.detectAndDecode(image)
            if bbox is not None and data != "" and data not in IGNORED_PAYLOADS:
                return data
        except Exception as e:
            self.get_logger().debug(f"OpenCV QR Detection failed: {e}")

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
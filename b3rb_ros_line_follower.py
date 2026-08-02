# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import time
import math
import json
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 0.2  # Speed capped at 0.2 for precise control dynamically
TURN_MIN = -1.0
TURN_MAX = 1.0

# --- TWEAKABLE TIMERS & DISTANCES ---
SIGN_TIMEOUT = 0.5  # Seconds to wait after the last sign message before executing the turn
# ------------------------------------

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy integrated with NXP AIM Challenge logic.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        
        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        
        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------
        
        # Default controls
        self.target_speed = 0.2
        self.target_turn = 0.0

        # State variables
        self.current_location_qr = False
        self.lidar_ph_override = False
        self.obstacle_in_front = False
        self.avoidance_direction = None  # "LEFT" or "RIGHT" depending on obstacle
        self.near_building = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = "A"  # Default destination is patient A
        self.mission_completed = False

        self.latest_sign_board_info = {"A": 7, "B": 7, "C": 7, "X": 7, "Y": 7, "Z": 7}
        
        # --- Memory to delay turns until the sign is passed ---
        self.last_sign_msg = ""
        self.last_sign_time = 0.0
        self.pending_turn_direction = 0.0
        
        # Signboard state: 0.0 (Center), 1.0 (Blind to Left / Go Right), -1.0 (Blind to Right / Go Left)
        self.active_turn_direction = 0.0

        self.latest_uid = -1
        self.latest_ack = -1
        self.current_uid = 10
        self.on_destination = False

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info("Line Follower controller initialized.")

    def publish_drive_commands(self):
        """Timer callback that periodically publishes current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        """Receives lane boundaries from camera vector extractor and computes steering."""
        speed = SPEED_MAX
        turn = 0.0

        if self.on_destination == True:
            self.rover_move_manual_mode(0.0, 0.0)
            return

        if self.current_destination == "0":
            self.rover_move_manual_mode(0.0, 0.0)
            return

        vectors = message
        half_width = vectors.image_width / 2.0

        # --- NEW TIMEOUT LOGIC ---
        # Check if we have a pending turn AND the sign has left the camera's view
        if self.pending_turn_direction != 0.0 and (time.time() - self.last_sign_time > SIGN_TIMEOUT):
            self.active_turn_direction = self.pending_turn_direction
            self.pending_turn_direction = 0.0  # Clear pending so it only activates once
            state_str = "CENTER" if self.active_turn_direction == 0.0 else ("RIGHT" if self.active_turn_direction == 1.0 else "LEFT")
            self.get_logger().info(f"[STATE] Sign passed! Executing locked turn to: {state_str}")

        if vectors.vector_count == 0:  # None seen
            speed = 0.2
            if self.active_turn_direction != 0.0:
                turn = self.active_turn_direction * -1.0
            else:
                turn = 0.0

        elif vectors.vector_count == 1:  # Curve / Single lane boundary
            if self.obstacle_in_front:
                vector_center_x = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                safe_margin = half_width * 0.45  # Distance to maintain from the edge vector
                
                if self.avoidance_direction == "RIGHT":
                    if vector_center_x < half_width:
                        #We see the left line .steer away from it
                        turn = -0.6
                    else:
                        #We see the right line maintain safe offset distance.
                        target_x = vector_center_x - safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width
                else:
                    #Dodging left
                    if vector_center_x > half_width:
                        # We see the right line .steer away from it
                        turn = 0.6
                    else:
                        # We see the left line maintain safe distance.
                        target_x = vector_center_x + safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width
                speed = 0.15
            else:
                # Same slope-following logic logic for 1 vector
                deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                turn = deviation / half_width
                speed = 0.2

        elif vectors.vector_count == 2:  # Straight track / Both boundaries visible
            middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
            middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2.0
            
            if self.obstacle_in_front:
                # [OVERRIDE]: Turn off center-following to dodge obstacle
                safe_margin = half_width * 0.25  #Distance to maintain from the correct edge line                
                if self.avoidance_direction == "RIGHT":
                    target_x = middle_x_right - safe_margin
                    line_slope = vectors.vector_2[1].x - vectors.vector_2[0].x
                else:
                    target_x = middle_x_left + safe_margin
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    
                turn = (line_slope + (half_width - target_x)) / half_width
                speed = 0.15
                
            elif self.active_turn_direction == 1.0:
                # [BLIND TO LEFT]: Turning right based on sign board.
                # Uses only vector_2 slope to stay parallel (centered) but follow the right side.
                deviation = vectors.vector_2[1].x - vectors.vector_2[0].x
                turn = deviation / half_width
                
                speed = 0.2
                
            elif self.active_turn_direction == -1.0:
                # [BLIND TO RIGHT]: Turning left based on sign board.
                # Uses only vector_1 slope to stay parallel (centered) but follow the left side.
                deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                turn = deviation / half_width
                self.get_logger().info("HELLOBRO")
                speed = 0.2

            else:
                # [NORMAL]: Center following using both lines
                middle_x = (middle_x_left + middle_x_right) / 2.0
                deviation = half_width - middle_x
                turn = deviation / half_width
                speed = 0.2

        if (turn > 0.4 or turn < -0.4) and not self.obstacle_in_front:
            speed = 0.2
        self.rover_move_manual_mode(speed, turn)

    def lidar_callback(self, message):
        """Receives LIDAR range measurements to check building proximity or obstacles."""
        num_readings = len(message.ranges)

        if self.lidar_ph_override == True:
            # ----------------------------------------------------
            # STEP 1: Building Proximity Detection (Patient/Hospital)
            # ----------------------------------------------------
            print("XXXXXXXXXXXXXXXSTEP-2")
            right_side = list(message.ranges[82:98])
            num_sides_detected_right = 0
            left_side = list(message.ranges[262:278])
            num_sides_detected_left = 0

            for x in right_side:
                if x < 1.0:
                    num_sides_detected_right += 1

            for x in left_side:
                if x < 1.0:
                    num_sides_detected_left += 1

            if num_sides_detected_right >= 7 or num_sides_detected_left >= 7:
                print("XXXXXXXXXXXXXXXSTEP-3")
                self.on_destination = True
                self.send_server_update(self.current_destination)
                return

        else:
            # front obstacle detection
            cr=0
            cl=0
            mid = num_readings // 2
            front_right_sector = list(message.ranges[mid - 80 : mid])
            front_left_sector = list(message.ranges[mid : mid + 80])
            for r in front_right_sector:
                if r<1 and r>0.1:
                    cr+=1
            
            for r in front_left_sector:
                if r<1 and r>0.1:
                    cl+=1
        
            # valid_right = [r for r in front_right_sector if r > 0.1 and not math.isinf(r)]
            # valid_left = [r for r in front_left_sector if r > 0.1 and not math.isinf(r)]
            
            # min_right = min(valid_right) if valid_right else float('inf')
            # min_left = min(valid_left) if valid_left else float('inf')
            
            #If an object is detected within 1.2 meters
            if cr>=10 or cl>=10:
                self.obstacle_in_front = True
                self.active_turn_direction = 0.0 # Clear the signboard lock if an obstacle appears
                
                if cl > cr:
                    #Obstacle is closer on the left side of the track.Dodge right
                    self.avoidance_direction = "RIGHT"
                else:
                    #Obstacle is closer on the right side of the track.Dodge left
                    self.avoidance_direction = "LEFT"
            else:
                self.obstacle_in_front = False
                self.avoidance_direction = None


    def server_communication_callback(self, message):
        """Receives coordination commands from the Municipality Server."""
        if message.dest == 1:  # Destined for Buggy
            self.get_logger().info(f"Received Server Message: {message.msg}")
            
            # Handle Server Acknowledgements
            if message.ack == 1:
                self.latest_ack = message.uid
            else:
                self.send_server_ack(message.uid)
                # Parse server assignment payload if present
                if message.msg:
                    self.current_destination = message.msg

    def send_server_update(self, text_msg):
        """Sends status messages to the server with retry/acknowledgement mechanism."""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = self.current_uid  # Rolling message ID
        server_msg.ack = 0
        server_msg.msg = text_msg

        self.latest_uid = server_msg.uid
        issent = False
        tries = 0

        self.rover_move_manual_mode(0.0, 0.0)

        while True:
            self.rover_move_manual_mode(0.0, 0.0)
            if tries >= 10:
                break
            if self.latest_ack == self.latest_uid:
                issent = True
                break

            self.publisher_server.publish(server_msg)
            tries += 1
            rclpy.spin_once(self, timeout_sec=0.1)

        self.on_destination = False
        self.lidar_ph_override = False
        self.current_uid += 1
        return issent

    def send_server_ack(self, uid):
        """Sends acknowledgement back to the server."""
        server_msg = ServerCommunication()
        server_msg.src = 1
        server_msg.dest = 2
        server_msg.uid = uid
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)

    def qr_detection_callback(self, message):
        """Receives QR codes scanned from buildings and manages frame transitions."""
        if message.data:
            self.get_logger().info(f"Heard QR code: {message.data}")
            self.current_location_qr = True
        else:
            # QR went out of frame -> Trigger reset message and enable LIDAR building scan
            if self.current_location_qr:
                self.current_location_qr = False
                self.send_server_update("RESET")
                self.lidar_ph_override = True

    def sign_board_callback(self, message):
        """Receives traffic sign board direction hints."""
        try:
            # Safely parse the string to JSON instead of using dict()
            clean_string = message.data.replace("'", '"')
            sign_data_raw = json.loads(clean_string)
            
            direction_map = {"Left": -1.0, "Right": 1.0, "Straight": 0.0}
            sign_data = {}
            for label, direction in sign_data_raw.items():
                if direction in direction_map:
                    sign_data[label] = direction_map[direction]

            update = True
            for label in ["A","B","C","X","Y","Z"]:
                if not label in sign_data:
                    update = False
                    break

            if update: 
                # The sign is currently in frame. Keep resetting the timer!
                self.last_sign_time = time.time()
                
                # Ignore duplicate consecutive messages to save processing
                if message.data == self.last_sign_msg:
                    return
                self.last_sign_msg = message.data
                
                self.latest_sign_board_info.update(sign_data)
                
                # Assign to PENDING state (waits for the sign to leave the camera frame)
                if self.current_destination in sign_data:
                    self.pending_turn_direction = sign_data[self.current_destination]
            
        except Exception:
            pass # Removed logging to prevent terminal spam for invalid messages


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

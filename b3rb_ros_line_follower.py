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
from enum import IntEnum, auto

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 2.0  # Speed capped at 0.2 for precise control dynamically
TURN_MIN = -1.0
TURN_MAX = 1.0

HIGH_SCALE = 6.0
LOW_SCALE = 5.0

SIGN_TIMEOUT = 12.0

class SignState(IntEnum):
    FINDING = auto()
    FOUND = auto()
    CROSSED = auto()

class State(IntEnum):
    TURNING_LEFT = auto()
    TURNING_RIGHT = auto()
    TURNING_STRAIGHT = auto()
    LINE_FOLLOWING = auto()
    STOPPED = auto()
    OBSTACLE_AVOIDING_RIGHT = auto()
    OBSTACLE_AVOIDING_LEFT = auto()

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
        self.turn = 0.0

        # State variables
        self.buggy_state = State.LINE_FOLLOWING
        self.sign_state = SignState.FINDING
        self.turn_start_time = self.get_clock().now().nanoseconds / 1e9
        self.SCALE = HIGH_SCALE

        self.current_location_qr = False
        self.lidar_ph_override = False
        self.obstacle_in_front = False
        self.avoidance_direction = None  # "LEFT" or "RIGHT" depending on obstacle
        self.near_building = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = "A"  # Default destination is patient A
        self.msg_sent = ""
        self.last_msg_sent = ""
        self.stop = False
        self.last=0.0
        self.mission_completed = False
        self.turn_direction = "Straight"

        self.last_sign_time = 0.0

        self.latest_sign_board_info = {"A": "", "B": "", "C": "", "X": "", "Y": "", "Z": ""}
        self.mappings = {"PATIENT_1" : "A", "PATIENT_2" : "B", "PATIENT_3" : "C", "HOSPITAL_1" : "X", "HOSPITAL_2" : "Y", "HOSPITAL_3" : "Z"}

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
        speed = 0.0
        turn = 0.0
        current_time = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info(f"{self.SCALE} , {self.buggy_state}, {self.sign_state}")

        vectors = message
        image_width = vectors.image_width
        half_width = image_width/2

        if current_time - self.last_sign_time > SIGN_TIMEOUT/self.SCALE and self.sign_state == SignState.FOUND:
            self.sign_state = SignState.CROSSED


        match self.buggy_state:
            case State.LINE_FOLLOWING:
                if(self.stop):
                    self.buggy_state = State.STOPPED

                elif(self.obstacle_in_front):
                    if(self.avoidance_direction == "RIGHT"):
                        self.buggy_state = State.OBSTACLE_AVOIDING_RIGHT
                    else:
                        self.buggy_state = State.OBSTACLE_AVOIDING_LEFT
                
                elif(self.sign_state == SignState.CROSSED):

                    match self.latest_sign_board_info[self.current_destination]:
                        case "Straight":
                            self.buggy_state = State.TURNING_STRAIGHT
                        case "Left":
                            self.buggy_state = State.TURNING_LEFT
                        case "Right":
                            self.buggy_state = State.TURNING_RIGHT
                        case _:
                            pass
                    self.turn_start_time = current_time

            case State.TURNING_LEFT:
                if(current_time - self.turn_start_time > 9.0/self.SCALE):
                    self.buggy_state = State.LINE_FOLLOWING
                    self.sign_state = SignState.FINDING

            case State.TURNING_RIGHT:
                if(current_time - self.turn_start_time > 9.0/self.SCALE):
                    self.buggy_state = State.LINE_FOLLOWING
                    self.sign_state = SignState.FINDING

            case State.TURNING_STRAIGHT:
                if(current_time - self.turn_start_time > 2.0/self.SCALE):
                    self.buggy_state = State.LINE_FOLLOWING
                    self.sign_state = SignState.FINDING

            case State.STOPPED:
                if(not self.stop): self.buggy_state = State.LINE_FOLLOWING

            case State.OBSTACLE_AVOIDING_RIGHT:
                if(not self.obstacle_in_front):
                    self.buggy_state = State.LINE_FOLLOWING

            case State.OBSTACLE_AVOIDING_LEFT:
                if(not self.obstacle_in_front):
                    self.buggy_state = State.LINE_FOLLOWING



        match self.buggy_state:
            case State.LINE_FOLLOWING:
                if vectors.vector_count == 0:  # None seen
                    turn = 0.0

                elif vectors.vector_count == 1:  # Curve / Single lane boundary
                    deviation = vectors.vector_1[1].x - vectors.vector_1[0].x
                    turn = deviation / half_width

                elif vectors.vector_count == 2:  # Straight track / Both boundaries visible
                    middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2.0
                    
                    middle_x = (middle_x_left + middle_x_right) / 2.0
                    deviation = half_width - middle_x
                    turn = deviation / half_width

                speed = 0.2 * self.SCALE

                if(self.sign_state == SignState.FINDING or self.lidar_ph_override or self.current_location_qr): self.SCALE = HIGH_SCALE
                else: self.SCALE = LOW_SCALE

            case State.TURNING_LEFT:
                if vectors.vector_count == 0:
                    turn = 0.6

                elif vectors.vector_count == 1:
                    vector_center_x = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    safe_margin = half_width*0.45

                    if vector_center_x > half_width:
                        # We see the right line .steer away from it
                        turn = 0.6
                    else:
                        # We see the left line maintain safe distance.
                        target_x = vector_center_x + safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width

                elif vectors.vector_count == 2:
                    middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    safe_margin = half_width*0.45

                    target_x = middle_x_left + safe_margin
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    
                    turn = (line_slope + (half_width - target_x)) / half_width

                speed = 0.15 * self.SCALE

            case State.TURNING_RIGHT:
                if vectors.vector_count == 0:
                    turn = -0.6

                elif vectors.vector_count == 1:
                    vector_center_x = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    safe_margin = half_width*0.45

                    if vector_center_x < half_width:
                        #We see the left line .steer away from it
                        turn = -0.6
                    else:
                        #We see the right line maintain safe offset distance.
                        target_x = vector_center_x - safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width

                elif vectors.vector_count == 2:
                    middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2.0
                    safe_margin = half_width*0.45

                    target_x = middle_x_right - safe_margin
                    line_slope = vectors.vector_2[1].x - vectors.vector_2[0].x
                    
                    turn = (line_slope + (half_width - target_x)) / half_width

                speed = 0.15 * self.SCALE

            case State.TURNING_STRAIGHT:
                turn = 0.0
                speed = 0.2 * self.SCALE

            case State.STOPPED:
                turn = 0.0
                speed = 0.0

            case State.OBSTACLE_AVOIDING_RIGHT:
                if vectors.vector_count == 1:
                    vector_center_x = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    safe_margin = half_width * 0.45  # Distance to maintain from the edge vector
                    
                    if vector_center_x < half_width:
                        #We see the left line .steer away from it
                        turn = -0.6
                    else:
                        #We see the right line maintain safe offset distance.
                        target_x = vector_center_x - safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width
                    speed = 0.15 * self.SCALE

                elif vectors.vector_count == 2:
                    middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2.0
                    
                    # [OVERRIDE]: Turn off center-following to dodge obstacle
                    safe_margin = half_width * 0.25  #Distance to maintain from the correct edge line                
                    target_x = middle_x_right - safe_margin
                    line_slope = vectors.vector_2[1].x - vectors.vector_2[0].x
                    
                    turn = (line_slope + (half_width - target_x)) / half_width
                    speed = 0.15 * self.SCALE

                else:
                    turn = -0.6
                    speed = 0.15 * self.SCALE

            case State.OBSTACLE_AVOIDING_LEFT:
                if vectors.vector_count == 1:
                    vector_center_x = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                    safe_margin = half_width * 0.45  # Distance to maintain from the edge vector
                    
                    if vector_center_x > half_width:
                        # We see the right line .steer away from it
                        turn = 0.6
                    else:
                        # We see the left line maintain safe distance.
                        target_x = vector_center_x + safe_margin
                        turn = (line_slope + (half_width - target_x)) / half_width
                    speed = 0.15 * self.SCALE

                elif vectors.vector_count == 2:
                    middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2.0
                    middle_x_right = (vectors.vector_2[0].x + vectors.vector_2[1].x) / 2.0
                    
                    # [OVERRIDE]: Turn off center-following to dodge obstacle
                    safe_margin = half_width * 0.25  #Distance to maintain from the correct edge line                
                    target_x = middle_x_left + safe_margin
                    line_slope = vectors.vector_1[1].x - vectors.vector_1[0].x
                        
                    turn = (line_slope + (half_width - target_x)) / half_width
                    speed = 0.15 * self.SCALE
                else:
                    turn = 0.6
                    speed = 0.15 * self.SCALE


        self.rover_move_manual_mode(speed, turn)

    def lidar_callback(self, message):
        """Receives LIDAR range measurements to check building proximity or obstacles."""
        num_readings = len(message.ranges)

        if self.lidar_ph_override == True and self.msg_sent == self.current_destination:
            # ----------------------------------------------------
            # STEP 1: Building Proximity Detection (Patient/Hospital)
            # ----------------------------------------------------
            print("XXXXXXXXXXXXXXXSTEP-2")
            self.get_logger().info("HEHHHHHHHHEEEEEEEEEE0000000000000000000")
            right_side = list(message.ranges[80:92])
            num_sides_detected_right = 0
            left_side = list(message.ranges[268:280])
            num_sides_detected_left = 0

            for x in right_side:
                if x < 2.0:
                    num_sides_detected_right += 1

            for x in left_side:
                if x < 2.0:
                    num_sides_detected_left += 1

            if num_sides_detected_right >= 9 or num_sides_detected_left >= 9:
                print("XXXXXXXXXXXXXXXSTEP-3")
                self.on_destination = True

                if(self.last_msg_sent != self.msg_sent):
                    self.send_server_update(self.msg_sent)
                    self.last_msg_sent = self.msg_sent
                    self.stop = True

                return

        else:
            #front obstacle detection
            mid = num_readings // 2
            cr = cl = 0
            front_right_sector = list(message.ranges[mid - 20 : mid])
            front_left_sector = list(message.ranges[mid : mid + 20])

            for r in front_right_sector:
                if(r<1.2 and r>0.1): cr+=1

            for r in front_left_sector:
                if(r<1.2 and r>0.1): cl+=1
            
            
            #If an object is detected within 1.2 meters
            if cr > 7 or cl > 7:
                self.obstacle_in_front = True
                if cl >= cr:
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
                    self.stop = False

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
        
        if message.data == "RESET":
            # QR went out of frame -> Trigger reset message and enable LIDAR building scan
            if self.current_location_qr:
                self.current_location_qr = False
                # self.send_server_update("RESET")
                self.lidar_ph_override = True
        elif message.data:
                    self.get_logger().info(f"Heard QR code: {message.data}")
                    self.current_location_qr = True
                    decoded_data = self.mappings[self.legacy_to_json(message.data)]

                    if(self.msg_sent != decoded_data):
                        self.msg_sent = decoded_data

    def sign_board_callback(self, message):
        """Receives traffic sign board direction hints."""
        try:
            sign_data = json.loads(message.data.replace("'", '"'))
            update = True
            self.last_sign_time = self.get_clock().now().nanoseconds / 1e9

            for label in ["A","B","C","X","Y","Z"]:
                if not label in sign_data:
                    update = False
                    break

            if(update):
                if(self.sign_state == SignState.FINDING): self.sign_state = SignState.FOUND

                self.latest_sign_board_info.update(sign_data)
            # else:
            #     if(self.sign_state == SignState.FOUND): self.sign_state = SignState.CROSSED

            
        except Exception as e:
            self.get_logger().error(f"Error occurred: {str(e)}")

    def legacy_to_json(self, legacy_str: str) -> str:
        cleaned = legacy_str.strip().lstrip("{").rstrip("}")

        if ":" not in cleaned:
            raise ValueError("Invalid legacy format: Missing colon separator.")

        key, value = cleaned.split(":", 1)

        value_str = value.strip().strip("'\"")
        return value_str


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

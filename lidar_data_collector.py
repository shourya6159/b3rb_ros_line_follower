import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import csv
import threading
import sys
import termios
import tty
import math

class LidarDataCollector(Node):
    def __init__(self):
        super().__init__('lidar_data_collector')

        # Subscribe to LIDAR
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)

        # State variables
        self.recording = True  # Start recording immediately
        self.current_label = 0
        
        # Open CSV file for writing dataset in append mode
        self.csv_file = open('parking_lidar_dataset.csv', 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        self.get_logger().info("Lidar Data Collector Initialized.")
        
        print("\r\n--- RECORDING ACTIVE ---")
        print("Press '1' -> Label 0 (Don't turn)")
        print("Press '2' -> Label 1 (Turn Left)")
        print("Press 'q' -> Save and Quit\n")

        # Start keyboard listener in a background thread
        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()

    def lidar_callback(self, msg):
        if not self.recording:
            return

        # The B3RB LIDAR typically outputs a 360-element array (0 to 359).
        # To get 180 to 360 degrees, we take the second half of the ranges array.
        mid_index = len(msg.ranges) // 2
        lidar_slice = msg.ranges[mid_index:]

        # Clean LIDAR data: Neural networks cannot process 'inf' or 'NaN'
        cleaned_slice = []
        for r in lidar_slice:
            if math.isinf(r) or math.isnan(r):
                cleaned_slice.append(10.0)  # Replace inf/nan with a max distance value (e.g., 10 meters)
            else:
                cleaned_slice.append(round(r, 4)) # Round for cleaner CSV

        # Save to CSV: [Label, Range_180, Range_181, ... Range_360]
        row = [self.current_label] + cleaned_slice
        self.csv_writer.writerow(row)

    def keyboard_listener(self):
        # Read terminal input dynamically without requiring the user to press 'Enter'
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                key = sys.stdin.read(1)
                
                if key == '1':
                    self.current_label = 0
                    print("\rCurrent Label: 0 (Straight/Wait)   ", end='')
                elif key == '2':
                    self.current_label = 1
                    print("\rCurrent Label: 1 (Turn Left)       ", end='')
                elif key == 'q' or key == '\x03': # 'q' or Ctrl+C
                    print("\r\nSaving data and exiting...       ")
                    self.recording = False
                    self.csv_file.close()
                    rclpy.shutdown()
                    break
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

def main(args=None):
    rclpy.init(args=args)
    node = LidarDataCollector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.csv_file.closed:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
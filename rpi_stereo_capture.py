import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from picamera2 import Picamera2
import threading
import time

class RPiStereoCaptureNode(Node):
    """
    Captures stereo frames from two Raspberry Pi camera modules.
    Publishes left and right images for downstream stereo processing.
    """
    
    def __init__(self):
        super().__init__('rpi_stereo_capture')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_0_index', 0),
                ('camera_1_index', 1),
                ('frame_width', 640),
                ('frame_height', 480),
                ('frame_rate', 30),
                ('exposure_time_us', 10000),
                ('analog_gain', 1.0),
                ('use_libcamera', True),
            ]
        )
        
        self.camera_0_index = self.get_parameter('camera_0_index').value
        self.camera_1_index = self.get_parameter('camera_1_index').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.exposure_time_us = self.get_parameter('exposure_time_us').value
        self.analog_gain = self.get_parameter('analog_gain').value
        self.use_libcamera = self.get_parameter('use_libcamera').value
        
        self.left_pub = self.create_publisher(Image, '/stereo/left/image_raw', 10)
        self.right_pub = self.create_publisher(Image, '/stereo/right/image_raw', 10)
        
        self.bridge = CvBridge()
        self.cam_left = None
        self.cam_right = None
        self.running = True
        
        self.get_logger().info(f'RPi Stereo Capture Node initializing...')
        self.get_logger().info(f'Resolution: {self.frame_width}x{self.frame_height} @ {self.frame_rate}Hz')
        self.get_logger().info(f'Use libcamera: {self.use_libcamera}')
        
        # Initialize cameras
        self._init_cameras()
        
        # Capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def _init_cameras(self):
        """Initialize both camera modules using libcamera or legacy API."""
        try:
            if self.use_libcamera:
                self.get_logger().info('Using libcamera (Picamera2)...')
                self._init_libcamera()
            else:
                self.get_logger().info('Using legacy PiCamera API...')
                self._init_picamera()
        except Exception as e:
            self.get_logger().error(f'Camera init failed: {e}')
            raise
    
    def _init_libcamera(self):
        """Initialize cameras with Picamera2 (libcamera backend)."""
        try:
            self.cam_left = Picamera2(self.camera_0_index)
            self.cam_right = Picamera2(self.camera_1_index)
            
            # Configure both cameras identically for stereo
            config_left = self.cam_left.create_video_configuration(
                main={"format": "RGB888", "size": (self.frame_width, self.frame_height)},
                controls={"FrameRate": self.frame_rate}
            )
            config_right = self.cam_right.create_video_configuration(
                main={"format": "RGB888", "size": (self.frame_width, self.frame_height)},
                controls={"FrameRate": self.frame_rate}
            )
            
            self.cam_left.configure(config_left)
            self.cam_right.configure(config_right)
            
            self.cam_left.start()
            self.cam_right.start()
            
            self.get_logger().info('Libcamera stereo cameras initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Libcamera init error: {e}')
            raise
    
    def _init_picamera(self):
        """Fallback: Initialize with legacy PiCamera API."""
        try:
            from picamera import PiCamera
            
            self.cam_left = PiCamera(camera_num=self.camera_0_index)
            self.cam_right = PiCamera(camera_num=self.camera_1_index)
            
            for cam in [self.cam_left, self.cam_right]:
                cam.resolution = (self.frame_width, self.frame_height)
                cam.framerate = self.frame_rate
                cam.exposure_mode = 'off'
                cam.shutter_speed = self.exposure_time_us
                cam.analog_gain = self.analog_gain
                cam.start_preview(fullscreen=False)
            
            self.get_logger().info('Legacy PiCamera stereo cameras initialized')
            
        except Exception as e:
            self.get_logger().error(f'PiCamera init error: {e}')
            raise
    
    def _capture_loop(self):
        """Main capture loop: grab frames and publish."""
        frame_id = 0
        while self.running:
            try:
                if self.use_libcamera:
                    left_frame = self.cam_left.capture_array()
                    right_frame = self.cam_right.capture_array()
                else:
                    # Legacy fallback (less ideal for stereo sync)
                    left_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    right_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    self.cam_left.capture(left_frame, format='rgb')
                    self.cam_right.capture(right_frame, format='rgb')
                
                # Convert RGB to BGR for OpenCV compatibility
                left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
                right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
                
                # Publish as ROS2 Image messages
                stamp = self.get_clock().now().to_msg()
                
                left_msg = self.bridge.cv2_to_imgmsg(left_bgr, encoding='bgr8')
                left_msg.header.stamp = stamp
                left_msg.header.frame_id = 'stereo_left'
                left_msg.header.seq = frame_id
                self.left_pub.publish(left_msg)
                
                right_msg = self.bridge.cv2_to_imgmsg(right_bgr, encoding='bgr8')
                right_msg.header.stamp = stamp
                right_msg.header.frame_id = 'stereo_right'
                right_msg.header.seq = frame_id
                self.right_pub.publish(right_msg)
                
                frame_id += 1
                
                if frame_id % 30 == 0:
                    self.get_logger().debug(f'Published frame pair {frame_id}')
                
            except Exception as e:
                self.get_logger().error(f'Capture error: {e}')
                time.sleep(0.1)
    
    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info('Shutting down RPi stereo capture...')
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.use_libcamera:
            if self.cam_left:
                self.cam_left.stop()
            if self.cam_right:
                self.cam_right.stop()
        else:
            if self.cam_left:
                self.cam_left.close()
            if self.cam_right:
                self.cam_right.close()
        
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RPiStereoCaptureNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
controller.py

Written by Claude, rewriten by Fade

Pure Pursuit steering controller — ROS2 node.
Consumes the lookahead point from maps_navigation and a magnetic compass heading,
publishes cmd_vel Twist messages to drive the robot.

Requires:
    pip install geomag

ROS2 topics:
    Subscribes:
        /gps              (sensor_msgs/NavSatFix)  — raw GPS position
        /compass          (std_msgs/Float32)        — magnetic heading in degrees (0-360, 0=North)
        /cv/lane_offset   (std_msgs/Float32)        — FUTURE: signed lateral offset from lane centre (metres)
        /cv/confidence    (std_msgs/Float32)        — FUTURE: CV confidence 0.0-1.0
        /obstacle/distance (std_msgs/Float32)       — FUTURE: distance to nearest obstacle (metres)     NOTE - CHANGE TO ACTUAL MSG TYPE TO HANDLE MULTIPLE OBSTACLE
        /obstacle/bearing  (std_msgs/Float32)       — FUTURE: bearing to nearest obstacle (degrees, robot frame)    NOTE - CHANGE 

    Publishes:
        /cmd_vel          (geometry_msgs/Twist)     — linear.x + angular.z

Blending architecture (CommandArbiter):
    All three systems run simultaneously. Their influence is blended, not overridden.
    - Pure Pursuit   → macro direction (always active, baseline)
    - CV             → road centering trim (scales with CV confidence)
    - Obstacle       → repulsion force (scales with proximity, CV keeps it road-bounded)
    One hard stop exists: obstacle within OBSTACLE_STOP_M for longer than OBSTACLE_STOP_TIMEOUT_S.
"""


import math
try:
    import geomag
    GEOMAG_AVAILABLE = True
except ImportError:
    GEOMAG_AVAILABLE = False
    print('[Controller] WARNING - Geomag unavailable, using static declination values.')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from maps_nav import RobotNavigator, LatLng


'''
PURE PURSUIT CONFIG
'''



class PurePursuitConfig:
    # --- Robot physical properties ---
    WHEELBASE_M         = 0.5     # Distance between front and rear axles (metres). Measure your robot.

    # --- Lookahead ---
    LOOKAHEAD_TIME_S    = 0.8     # lookahead_distance = speed * this value (seconds)
    MIN_LOOKAHEAD_M     = 2.0     # Floor — don't look closer than this even at low speed
    MAX_LOOKAHEAD_M     = 10.0    # Ceiling — don't look further than this at high speed

    # --- Speed ---
    CRUISE_SPEED_MPS    = 1.0     # Normal forward speed (metres/second). Start conservative.
    MIN_SPEED_MPS       = 0.2     # Speed floor during sharp turns
    TURN_SLOWDOWN_DEG   = 20.0    # Heading error above this starts slowing the robot down

    # --- Route completion ---
    ARRIVAL_THRESHOLD_M = 3.0     # How close to the final waypoint counts as "arrived"

    # --- Magnetic declination ---
    # If geomag is unavailable, set this manually for your area.
    # Find your value at: https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
    # Positive = east declination, negative = west.
    MANUAL_DECLINATION_DEG = 0.0

    # --- CV blending ---
    # CV angular correction is scaled by confidence before being added to PP output.
    # At confidence 1.0 the full correction is applied; at 0.0 CV has no effect.
    CV_MAX_ANGULAR_CORRECTION  = 0.3   # rad/s — cap on how hard CV can trim steering

    # --- Obstacle blending ---
    # Repulsion grows as the obstacle gets closer and is added on top of PP+CV angular,
    # so CV naturally bounds how far it can push the robot sideways off the road.
    OBSTACLE_INFLUENCE_START_M = 5.0   # metres — repulsion begins at this distance
    OBSTACLE_STOP_M            = 0.5   # metres — triggers hard stop safety net
    OBSTACLE_STOP_TIMEOUT_S    = 3.0   # seconds — time to wait at hard stop before giving up
    OBSTACLE_MAX_ANGULAR       = 0.8   # rad/s — max repulsion contribution to angular.z
    OBSTACLE_SPEED_SCALE       = True  # Set False to disable proximity speed reduction (not recommended)


'''
PURE PURSUIT CONTROLLER
'''

class PurePursuitController:
    def __init__(self, config, navigator):
        self.cfg = config
        self.nav = navigator
        self.declination: float|None = None

    def _apply_declination(self, magnetic_deg, position):
        # Convert magnetic heading from compass to true heading
        if self.declination is None:
            if GEOMAG_AVAILABLE:
                self.declination = geomag.declination(position.lat, position.lng)
                print("[Controller] Magnetic Declination at current position - ", self.declination)
            else:
                self.declination = self.cfg.MANUAL_DECLINATION_DEG
                print('[Controller] Using manual declination')
            
        true_deg = (magnetic_deg + self.declination)%360
        return true_deg
    
    def _dynamic_lookahead(self, speed_mps):
        # Scale lookahead distance with speed for smooth curves
        dist = speed_mps*self.cfg.LOOKAHEAD_TIME_S
        return max(self.cfg.MIN_LOOKAHEAD_M, min(self.cfg.MAX_LOOKAHEAD_M, dist))
    
    @staticmethod
    def _bearing(a: LatLng, b:LatLng) -> float:
        '''
        Compute true bearing from point A to point B in rads
        0 = North, pi/2 = East, pi = South, -pi/2 = West
        '''
        lat1 = math.radians(a.lat)
        lat2 = math.radians(b.lat)
        dlng = math.radians(b.lng-a.lng)

        x = math.sin(dlng)*math.cos(lat2)
        y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlng)

        return math.atan2(x, y) # returns radians in -pi to pi
    
    @staticmethod
    def _wrap_angle(angle_rad):
        # Wrap angle from -pi to pi
        while angle_rad >  math.pi: angle_rad -= 2 * math.pi
        while angle_rad < -math.pi: angle_rad += 2 * math.pi
        return angle_rad
    
    def _speed_factor(self, heading_error_deg):
        # Returns speed multiplier, max speed for turns less than TURN_SLOWDOWN_DEG, linearly reducing to MIN_SPEED_MPS at 90 degree turn
        if heading_error_deg <= self.cfg.TURN_SLOWDOWN_DEG:
            return 1.0
        factor = 1.0 - (heading_error_deg-self.cfg.TURN_SLOWDOWN_DEG)/(90.0-self.cfg.TURN_SLOWDOWN_DEG)
        return max(0.0, factor)
    


    def compute(self, position:LatLng, magnetic_heading_deg:float) ->tuple[float, float]:
        '''
        Runs one pure pursuit iteration.

        args:
            position in latlng, can be snapped or raw gps
            magnetic_heading_deg, raw compass heading deg, in 0-360

        returns:
            (linear x, angular z) to be passed into command blender
        '''

        # Correct compass with magnetic declination
        true_heading_deg = self._apply_declination(magnetic_heading_deg, position)
        true_heading_rad = math.radians(true_heading_deg)

        # Generate lookahead point
        current_speed = self.cfg.CRUISE_SPEED_MPS
        lookahead_dist = self._dynamic_lookahead(current_speed)
        lookahead = self.nav.get_lookahead_point(position, lookahead_dist)

        # Compute heading error
        bearing_to_target = self._bearing(position, lookahead)

        heading_error_rad = self._wrap_angle(bearing_to_target-true_heading_rad)
        heading_error_deg = math.degrees(heading_error_rad)

        # Pure Pursuit
        # steering = arctan(2*wheelbase*sin(alpha)/lookahead)
        #CHECKKKKKKKKK ATAN OR ATAN2
        steering_angle = math.atan2(2.0*self.cfg.WHEELBASE_M*math.sin(heading_error_rad), lookahead_dist)

        # Linear speed control
        speed_factor = self._speed_factor(abs(heading_error_deg))
        linear_x = max(self.cfg.MIN_SPEED_MPS, self.cfg.CRUISE_SPEED_MPS*speed_factor)

        # Angular velocity
        angular_z = (linear_x*math.tan(steering_angle))/self.cfg.WHEELBASE_M

        return linear_x, angular_z
    



'''
COMMAND BLENDER
'''


class CommandArbiter:
    """
    Single owner of cmd_vel. Blends Pure Pursuit, CV, and obstacle avoidance
    into one command. No system fully overrides another, influence is proportional.

    Blending order (additive, not priority-based):
        1. Pure Pursuit          → baseline angular + linear
        2. CV correction         → angular trim, scaled by confidence (road-bounded)
        3. Obstacle repulsion    → angular + speed scaling, grows with proximity
           CV is still active during obstacle repulsion, so it naturally prevents
           the robot from being pushed off the road edge.
        4. Hard stop safety net  → linear=0 only if obstacle is critically close
           for longer than OBSTACLE_STOP_TIMEOUT_S. Last resort, not primary strategy.
    """

    def __init__(self, config):
        self.cfg = config

        #Pure Pursuit Baseline
        self.pp_linear = 0.0
        self.pp_angular = 0.0

        # CV Lane Keep
        self.cv_angular_correction = 0.0  #positive = right trim
        self.cv_confidence = 0.0

        # Obtsacle Input
        # Coming Soon
        

        # Hard Stop
        # Currently not configured

    
    def update_pure_pursuit(self, linear:float, angular:float):
        self.pp_linear = linear
        self.pp_angular = angular

    def update_cv(self, angular_correction:float, confidence: float):
        self.cv_angular_correction = angular_correction
        self.cv_confidence = max(0.0, min(1.0, confidence))

    def update_obstacle(self, distance_m:float, bearing_deg:float):
        # Coming Soon
        pass


    def command_blender(self, dt_s: float | None) -> tuple[float, float]:
        '''
        Blends all commands into a single output for cmd_vel

        args:
            dt_s: Time since last call, meant for hard stop
        
        returns:
            (linear_x, angular_z) values to be published on cmd_vel
        '''

        if dt_s:
            # Hard stop logic if applied goes here
            pass


        # Layer 1 - Pure Pursuit (Baseline Nav)
        angular = self.pp_angular
        linear = self.pp_linear

        # Layer 2 - CV Lane Keep
        # (Influence of this layer on steering is based on the confidence)
        cv_trim = self.cv_angular_correction*self.cv_confidence
        cv_trim = max(-self.cfg.CV_MAX_ANGULAR_CORRECTION, min(self.cfg.CV_MAX_ANGULAR_CORRECTION, cv_trim))
        angular += cv_trim

        # Layer 3 - Obstacle Repulsion
        # Coming Soon

        return linear, angular




'''
ROS2 NODE
'''

class PurePursuitNode(Node):

    def __init__(self):
        super().__init__("pure_pursuit")

        # Parameters
        self.declare_parameter("api_key",     "")
        self.declare_parameter("dest_lat",    0.0)
        self.declare_parameter("dest_lng",    0.0)

        api_key  = self.get_parameter("api_key").value
        dest_lat = self.get_parameter("dest_lat").value
        dest_lng = self.get_parameter("dest_lng").value

        
        # Core Objects
        self.cfg = PurePursuitConfig()
        self.navigator = RobotNavigator(api_key=api_key, lookahead_m=self.cfg.MIN_LOOKAHEAD_M)
        self.controller = PurePursuitController(self.cfg, self.navigator)
        self.arbiter = CommandArbiter(self.cfg)


        # State
        self.current_position: LatLng|None = None
        self.magnetic_heading: float|None = None
        self.route_planned: bool = False
        self.dest = (dest_lat, dest_lng)


        # ROS2 pubs and subs
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.create_subscription(NavSatFix, '/gps', self._gps_callback, 10)
        self.create_subscription(Float32, '/compass', self._compass_callback, 10)

        # CV-LaneKeep
        # Coming Soon

        # Obstacle Avoid
        # Coming Soon

        # Time Tracking for Hard Stop
        # Coming Soon

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('[Controller] PurePursuit node started, Waiting for GPS Fix')



    # CALLBACKS

    def _gps_callback(self, msg):
        self.current_position = LatLng(lat=msg.latitude, lng=msg.longitude)

        # Plan route on first valid GPS fix
        if not self.route_planned and self.dest != (0.0, 0.0):
            origin = (msg.latitude, msg.longitude)
            success = self.navigator.plan_route(origin=origin, destination=self.dest)
            if success:
                self.route_planned = True
                self.get_logger().info("[Controller] Route planned. Navigation active.")
            else:
                self.get_logger().error("[Controller] ERROR -Route planning failed. Check API key and destination.")

    def _compass_callback(self, msg: Float32):
        self.magnetic_heading = msg.data


    # CV Topic Callbacks
    # Obstacle Topic Callbacks
    # Coming Soon


    def _maybe_snap(self, position:LatLng)->LatLng:
        # Only call snap to road API if there is meaningful drift
        if not self.navigator.polyline:
            return position
        nearest = self.navigator.polyline[self.navigator.current_index]
        drift = self.navigator._haversine(position, nearest)
        if drift>3.5:
            return self.navigator.snap_to_road(position)
        return position

    def _publish(self, linear: float, angular:float):
        msg= Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_vel_pub.publish(msg)

    # MAIN CONTROL LOOP

    def _control_loop(self):
        # No path yet
        if not self.route_planned:
            return
        
        # No fix or no compass
        if self.current_position is None or self.magnetic_heading is None:
            return

        # Check arrival
        if self.navigator.is_route_complete(self.current_position, self.cfg.ARRIVAL_THRESHOLD_M):
            self.get_logger().info("[Controller] Arrived at destination")
            self._publish(0.0, 0.0)
            return
        
        # Snap GPS if needed
        position = self._maybe_snap(self.current_position)

        # Pure Pursuit
        linear, angular = self.controller.compute(position, self.magnetic_heading)

        self.arbiter.update_pure_pursuit(linear, angular)

        # Hard stop timing logic goes here
        # Future
        
        final_linear, final_angular = self.arbiter.command_blender(dt_s=None)
        
        self._publish(final_linear, final_angular)





def main(args=None):
    rclpy.init(args=args)
    node=PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
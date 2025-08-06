#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import tempfile
import time
import os
import threading

# from autoware_internal_planning_msgs.msg import PathWithLaneId #, VelocityLimit
from autoware_planning_msgs.msg import Path, PathPoint
from tier4_planning_msgs.msg import PathWithLaneId
from autoware_perception_msgs.msg import TrafficLightGroupArray, PredictedObjects
from nav_msgs.msg import Odometry, OccupancyGrid
from autoware_map_msgs.msg import LaneletMapBin
from geometry_msgs.msg import AccelWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import ParameterEvent

import lanelet2
from lanelet2.io import Origin, load
from lanelet2_extension_python.projection import MGRSProjector
from lanelet2.geometry import to2D, distance as lanelet_distance
from lanelet2.core import BasicPoint2d
import lanelet2_extension_python.utility.query as query
import lanelet2_extension_python.utility.utilities as utilities

from shapely.geometry import LineString, Point
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from langchain_openai import ChatOpenAI

from .my_logger import MyLogger

# input
clockTopic = '/clock'
accelerationTopic = '/localization/acceleration'
kinematicTopic = '/localization/kinematic_state'
vectorMapTopic = '/map/vector_map'
parameterTopic = '/parameter_events'
objectsTopic = '/perception/object_recognition/objects'
obstacleSegmentationTopic = '/perception/obstacle_segmentation/pointcloud'
occupancyMapTopic = '/perception/occupancy_grid_map/map'
trafficLightGroupArrayTopic = '/perception/traffic_light_recognition/traffic_signals'
pathWithLaneIdTopic = '/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id'
# velocityLimitTopic = '/planning/scenario_planning/max_velocity_default'

# output
pathTopic = '/planning/scenario_planning/lane_driving/behavior_planning/path'


class LLMControlNode(Node):
    def __init__(self):
        super().__init__('my_custom_planner_node')

        self.lanelet_map = None
        self.projector = None
        self.current_path: PathWithLaneId = None
        self.current_pose: Odometry = None
        self.traffic_signals: TrafficLightGroupArray = None
        self.objects: PredictedObjects = None
        self.current_decision = "GO"

        self.last_time_llm_used = 0
        self.threshold = 3.0
        self.data_lock = threading.Lock()  # Renamed for clarity
        self.decision_thread = threading.Thread(target=self.decision_making_loop, daemon=True)

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.traffic_signal_sub = self.create_subscription(TrafficLightGroupArray, trafficLightGroupArrayTopic,
                                                           self.traffic_light_callback, 10)
        self.path_sub = self.create_subscription(PathWithLaneId, pathWithLaneIdTopic, self.path_with_lane_id_callback,
                                                 10)
        self.map_sub = self.create_subscription(LaneletMapBin, vectorMapTopic, self.map_callback, map_qos)
        self.odom_sub = self.create_subscription(Odometry, kinematicTopic, self.odometry_callback, 10)
        self.objects_sub = self.create_subscription(PredictedObjects, objectsTopic, self.objects_callback, 10)

        self.publisher_path = self.create_publisher(Path, pathTopic, 10)

        self.llm_logger = MyLogger()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        # self.llm = ChatOpenAI(model="gpt-4o")

        self.publishing_timer = self.create_timer(0.1, self.path_publishing_loop)
        # self.processing_timer = self.create_timer(0.1, self.processing_loop)
        self.decision_thread.start()
        self.get_logger().info("LLM Control Node started. Waiting for data...")

    def map_callback(self, msg: LaneletMapBin):
        if self.lanelet_map is not None:
            return

        self.get_logger().info(f"Received map message with data size: {len(msg.data)} bytes")

        temp_file_path = os.path.join(tempfile.gettempdir(), f"temp_autoware_map_{os.getpid()}.bin")
        self.get_logger().info(f'map temp path: {temp_file_path} ')

        try:
            with open(temp_file_path, "wb") as f:
                f.write(msg.data)

            origin = Origin(49.0088285, 8.4231316)
            self.projector = MGRSProjector(origin)
            self.lanelet_map = load(temp_file_path, self.projector)

            self.get_logger().info('Lanelet2 map loaded successfully')
            self.destroy_subscription(self.map_sub)
        except Exception as e:
            self.get_logger().error(f"Failed to load Lanelet2 map: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def traffic_light_callback(self, msg: TrafficLightGroupArray):
        with self.data_lock:
            self.traffic_signals = msg
        # self.get_logger().info("Received TrafficLightGroupArray message.")

    def path_with_lane_id_callback(self, msg: PathWithLaneId):
        with self.data_lock:
            self.current_path = msg
        # self.get_logger().info("Received PathWithLaneId message.")

    def odometry_callback(self, msg: Odometry):
        with self.data_lock:
            self.current_pose = msg
        # self.get_logger().info(f'vehicle_pose = {self.current_pose.pose.pose.position}')
        # self.get_logger().info("Received odometry message.")

    def objects_callback(self, msg: PredictedObjects):
        with self.data_lock:
            self.objects = msg

    def decision_making_loop(self):
        while rclpy.ok():
            time.sleep(self.threshold)

            with self.data_lock:
                if not all(
                        [self.lanelet_map, self.current_path, self.current_pose, self.objects]):
                    self.get_logger().warn("Decision loop: waiting for data...", throttle_duration_sec=5)
                    continue

                # Copy data to process outside the lock
                path_copy = self.current_path
                pose_copy = self.current_pose
                signals_copy = self.traffic_signals
                objects_copy = self.objects

            nearest_light_info = self.find_nearest_relevant_traffic_light(path_copy, pose_copy, signals_copy)

            recognized_objects_str = self.get_recognized_objects(objects_copy)
            self.get_logger().info(recognized_objects_str)

            new_decision = "GO"  # Default to GO if no relevant light is found
            if nearest_light_info:
                prompt = self.build_prompt(
                    color=nearest_light_info['color'],
                    distance=nearest_light_info['distance'],
                    speed=pose_copy.twist.twist.linear.x
                )
                self.get_logger().info(f"Querying LLM with prompt:\n{prompt}")
                response = self.query_llm(prompt)

                if response and response.strip().upper() in ["GO", "STOP"]:
                    new_decision = response.strip().upper()
                else:
                    self.get_logger().warn(f"Received invalid LLM response: '{response}'. Defaulting to GO.")

            # --- Atomically update the shared decision ---
            with self.data_lock:
                if self.current_decision != new_decision:
                    self.get_logger().info(f"LLM decision changed: {self.current_decision} -> {new_decision}")
                    self.current_decision = new_decision

    def path_publishing_loop(self):
        with self.data_lock:
            # Check for necessary data without blocking
            if not self.current_path:
                return
            path_to_process = self.current_path
            decision_to_apply = self.current_decision

        # This part is always fast
        new_path = self.apply_decision(path_to_process, decision_to_apply)
        self.publisher_path.publish(new_path)

    def apply_decision(self, path_msg, decision):
        new_path = Path()
        new_path.header = path_msg.header
        self.get_logger().info(f'decision is: {decision}')
        for point_with_lane in path_msg.points:
            path_point = PathPoint()
            path_point.pose = point_with_lane.point.pose

            if decision == "STOP":
                path_point.longitudinal_velocity_mps = 0.0
                path_point.lateral_velocity_mps = 0.0
                path_point.heading_rate_rps = 0.0
            else:
                path_point.longitudinal_velocity_mps = point_with_lane.point.longitudinal_velocity_mps
                path_point.lateral_velocity_mps = point_with_lane.point.lateral_velocity_mps
                path_point.heading_rate_rps = point_with_lane.point.heading_rate_rps

            path_point.is_final = point_with_lane.point.is_final
            new_path.points.append(path_point)

        new_path.left_bound = path_msg.left_bound
        new_path.right_bound = path_msg.right_bound

        if not new_path.points:
            self.get_logger().warn("Generated path is empty. LLM decision may be invalid.")

        return new_path

    def find_nearest_relevant_traffic_light(self, current_path, current_pose, traffic_signals):

        try:
            path_lane_ids = {p.lane_ids[0] for p in current_path.points if p.lane_ids}
            if not path_lane_ids:
                self.get_logger().info('no path lane ids')
                return None

            # self.get_logger().info(f'lane ids: {path_lane_ids}')
            vehicle_pos_3d = current_pose.pose.pose.position
            self.get_logger().info(f'vehicle 3d pose: {vehicle_pos_3d}')
            vehicle_pos_2d = BasicPoint2d(vehicle_pos_3d.x, vehicle_pos_3d.y)
            self.get_logger().info(f'vehicle 2d pose: {vehicle_pos_2d}')
            min_distance = float('inf')
            nearest_light_info = None

            self.get_logger().info(f'len traffic signals: {len(traffic_signals.traffic_light_groups)}')
            for light_group in traffic_signals.traffic_light_groups:
                self.get_logger().info(f'light group info: {light_group}')
                traffic_light_id = light_group.traffic_light_group_id

                reg_elem = self.lanelet_map.regulatoryElementLayer.get(traffic_light_id)

                if not reg_elem:
                    self.get_logger().warn(f"Traffic light ID {traffic_light_id} from perception not found in map.")
                    continue

                self.get_logger().info(f'reg elem: {reg_elem}')

                # self.get_logger().info(f'reg elem found: {reg_elem}')
                # is_relevant = False
                # for key, value_list in reg_elem.parameters.items():
                #     self.get_logger().info(f'key: {key}, value: {value_list}')
                #     for lanelet_ref in value_list:
                #         if lanelet_ref.id in path_lane_ids:
                #             is_relevant = True
                #             break
                #     if is_relevant:
                #         break
                #
                # if not is_relevant:
                #     self.get_logger().info('not relevant')
                #     continue

                if hasattr(reg_elem, 'stopLine'):
                    stop_line_3d = reg_elem.stopLine
                    self.get_logger().info(f'stop line 3d found: {stop_line_3d}')

                    stop_line_2d = to2D(stop_line_3d)
                    self.get_logger().info(f'stop line 2d: {stop_line_2d}')
                    self.get_logger().info(
                        f'stop line 2d type: {type(stop_line_2d)} / stop line 3d type: {type(stop_line_3d)} / vehicle pose 2d: {type(vehicle_pos_2d)}')

                    dist = lanelet_distance(vehicle_pos_2d, stop_line_2d)
                    self.get_logger().info(f'dist: {dist}')
                    if dist < min_distance:
                        min_distance = dist
                        nearest_light_info = {
                            "distance": dist,
                            "color": self.get_dominant_color(light_group.elements)
                        }
            return nearest_light_info
        except Exception as e:
            self.get_logger().error(f"Error in find_nearest_relevant_traffic_light: {e}")
            return None

    def get_recognized_objects(self, objects_msg):
        if objects_msg.objects is None:
            self.get_logger().info('no objects in this frame')
            return 'no object exists in current frame'

        output = []
        output.append(f'total objects found: {len(objects_msg.objects)}\n')
        for i, obj in enumerate(objects_msg.objects):
            object_id_str = ''.join(f'{x:02x}' for x in obj.object_id.uuid)
            output.append(f"--- Object #{i + 1} ---\n")
            output.append(f"  ID: {object_id_str}\n")
            output.append(f"  Existence Probability: {obj.existence_probability:.2f}\n")

            # --- Classification ---
            if obj.classification:
                # Get the classification with the highest probability
                best_class = max(obj.classification, key=lambda c: c.probability)
                class_label = self.get_object_label(best_class.label)
                output.append(f"  Classification: {class_label} (Prob: {best_class.probability:.2f})\n")
            else:
                output.append("  Classification: Not available\n")

            # --- Kinematics (Position, Velocity, Acceleration) ---
            kinematics = obj.kinematics
            pos = kinematics.initial_pose_with_covariance.pose.position
            vel = kinematics.initial_twist_with_covariance.twist.linear
            accel = kinematics.initial_acceleration_with_covariance.accel.linear

            output.append("  Kinematics:\n")
            output.append(f"    Position (x, y, z):      ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) m\n")
            output.append(f"    Linear Velocity (x, y, z): ({vel.x:.2f}, {vel.y:.2f}, {vel.z:.2f}) m/s\n")
            output.append(f"    Linear Accel (x, y, z):    ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) m/s^2\n")

            # --- Shape ---
            shape = obj.shape
            dims = shape.dimensions
            shape_type_map = {0: 'BOUNDING_BOX', 1: 'CYLINDER', 2: 'POLYGON'}
            shape_type_str = shape_type_map.get(shape.type, 'UNKNOWN')

            output.append("  Shape:")
            output.append(f"    Type: {shape_type_str}")
            output.append(f"    Dimensions (l, w, h): ({dims.x:.2f}, {dims.y:.2f}, {dims.z:.2f}) m")
            output.append("-" * 20 + "\n")

        return "".join(output)

    def build_prompt(self, color: str, distance: float, speed: float) -> str:
        prompt_template = f"""
            You are a highly constrained, deterministic logic engine. Your only function is to evaluate the provided data against the following strict rules and output a single word: "STOP" or "GO". You must not use any external knowledge or creative interpretation. Follow the rules in the exact order they are presented.

            **Rule 1: Green Light**
            - If `Color` is **GREEN**, the output is **GO**.

            **Rule 2: Red or Amber Light**
            - If `Color` is **RED** or **AMBER**, apply the following sub-rules:
                - **If `Speed` is very low (less than 1.5 m/s):**
                    - If `Distance` is less than **5 meters**, output **STOP**.
                    - Otherwise, output **GO**.
                - **If `Speed` is higher (1.5 m/s or more):**
                    - Calculate `Time to Stop Line = Distance / Speed`.
                    - If `Time to Stop Line` is less than **4.0 seconds**, output **STOP**.
                    - Otherwise, output **GO**.

            **Rule 3: Unknown Color**
            - If `Color` is **UNKNOWN** or any other value, the output is **GO**.

            ---
            **Situation Data:**
            - Color: {color}
            - Distance: {distance:.1f}
            - Speed: {speed:.1f}
            ---

            Provide the single-word output based strictly on these rules.
        """
        return prompt_template.strip()

    def query_llm(self, prompt):
        try:
            self.get_logger().info(f"time before llm call: {time.time()}")
            response = self.llm.invoke(prompt)
            self.get_logger().info(f"time after llm call: {time.time()}")
            if hasattr(response, 'content'):
                return response.content
            else:
                self.get_logger().warn(f"LLM response object has no 'content' attribute: {type(response)}")
                return None
        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            self.llm_logger.log_error(f"LLM Invocation Error: {e}")
            return None

    def get_dominant_color(self, elements):
        for e in elements:
            if e.color == e.RED: return "RED"
            if e.color == e.AMBER: return "AMBER"
            if e.color == e.GREEN: return "GREEN"
        return "UNKNOWN"

    def get_object_label(self, o_type):
        classification_map = {
            0: 'UNKNOWN',
            1: 'CAR',
            2: 'TRUCK',
            3: 'BUS',
            4: 'TRAILER',
            5: 'MOTORCYCLE',
            6: 'BICYCLE',
            7: 'PEDESTRIAN'
        }
        return classification_map[o_type]


'''
    def processing_loop(self):
        if not all([self.lanelet_map, self.current_path, self.current_pose]):
            self.get_logger().warn("Waiting for map, path or pose data...", throttle_duration_sec=5)
            return

        with self.llm_lock:
            decision_to_apply = self.current_decision

        distance = None
        now = time.time()
        if now - self.last_time_llm_used >= self.threshold:
            nearest_light_info = self.find_nearest_relevant_traffic_light()
            recognized_objects = self.get_recognized_objects()
            self.get_logger().info(recognized_objects)
            if nearest_light_info:
                distance = nearest_light_info['distance']
                color = nearest_light_info['color']
                speed_mps = self.current_pose.twist.twist.linear.x

                prompt = self.build_prompt(color, distance, speed_mps)
                # self.get_logger().info(f"LLM input prompt: {prompt}")
                self.last_time_llm_used = now
                self.start_llm_thread(prompt)
                # response = self.query_llm(prompt)
                # if response:
                #     self.current_decision = response.strip().upper()
                #     self.get_logger().info(f"LLM response: {self.current_decision}")
                #     self.llm_logger.log_response(self.current_decision)
                #     path = self.apply_decision(self.current_path, self.current_decision, distance)

        #     else:
        #         self.current_decision = "GO"
        #         path = self.apply_decision(self.current_path, self.current_decision, None)
        # else:
        #     path = self.apply_decision(self.current_path, self.current_decision, None)

        path = self.apply_decision(self.current_path, decision_to_apply, distance)
        self.publisher_path.publish(path)
        # self.current_path = None
        # self.traffic_signals = None
        # self.current_pose = None
        '''


# def start_llm_thread(self, prompt):
#     def llm_worker():
#         self.get_logger().info(f"LLM input prompt: {prompt}")
#         self.llm_logger.log_request(prompt)
#         response = self.query_llm(prompt)
#         if response:
#             new_decision = response.strip().upper()
#             if new_decision in ["GO", "STOP"]:
#                 with self.llm_lock:
#                     self.current_decision = new_decision
#                 self.llm_logger.log_response(self.current_decision)
#                 self.get_logger().info(f"LLM response: {self.current_decision}")
#         # self.last_time_llm_used = time.time()
#
#     if self.llm_thread is None or not self.llm_thread.is_alive():
#         self.llm_thread = threading.Thread(target=llm_worker, daemon=True)
#         self.llm_thread.start()


def main(args=None):
    rclpy.init(args=args)
    node = LLMControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
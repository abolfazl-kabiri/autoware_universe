#!/usr/bin/env python3
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import tempfile
import time
import os

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
        self.current_decision = "GO"
        self.last_time_llm_used = 0
        self.threshold = 2

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.traffic_signal_sub = self.create_subscription(TrafficLightGroupArray, trafficLightGroupArrayTopic,
                                                           self.traffic_light_callback, 10)
        self.path_sub = self.create_subscription(PathWithLaneId, pathWithLaneIdTopic, self.path_with_lane_id_callback,
                                                 10)
        self.map_sub = self.create_subscription(LaneletMapBin, vectorMapTopic, self.map_callback, map_qos)
        self.odom_sub = self.create_subscription(Odometry, kinematicTopic, self.odometry_callback, 10)

        self.publisher_path = self.create_publisher(Path, pathTopic, 10)

        self.llm_logger = MyLogger()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        self.processing_timer = self.create_timer(0.1, self.processing_loop)
        self.get_logger().info("LLMTrafficLightControlNode started. Waiting for data...")

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
            # self.lanelet_map = lanelet2.io.load("/tmp/lanelet2_map.osm", origin)
            self.lanelet_map = load(temp_file_path, self.projector)

            self.get_logger().info('Lanelet2 map loaded successfully')
            self.destroy_subscription(self.map_sub)
        except Exception as e:
            self.get_logger().error(f"Failed to load Lanelet2 map: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def traffic_light_callback(self, msg: TrafficLightGroupArray):
        self.traffic_signals = msg
        # self.get_logger().info("Received TrafficLightGroupArray message.")

    def path_with_lane_id_callback(self, msg: PathWithLaneId):
        self.current_path = msg
        # self.get_logger().info("Received PathWithLaneId message.")

    def odometry_callback(self, msg: Odometry):
        self.current_pose = msg
        # self.get_logger().info(f'vehicle_pose = {self.current_pose.pose.pose.position}')
        # self.get_logger().info("Received odometry message.")

    def processing_loop(self):
        if not all([self.lanelet_map, self.current_path, self.current_pose]):
            self.get_logger().warn("Waiting for map, path or pose data...", throttle_duration_sec=5)
            return

        now = time.time()
        if now - self.last_time_llm_used >= self.threshold:
            nearest_light_info = self.find_nearest_relevant_traffic_light()
            if nearest_light_info:
                distance = nearest_light_info['distance']
                color = nearest_light_info['color']
                speed_mps = self.current_pose.twist.twist.linear.x

                prompt = self.build_prompt(color, distance, speed_mps)
                self.get_logger().info(f"LLM input prompt: {prompt}")
                self.llm_logger.log_request(prompt)
                response = self.query_llm(prompt)
                self.last_time_llm_used = now
                if response:
                    self.current_decision = response.strip().upper()
                    self.get_logger().info(f"LLM response: {self.current_decision}")
                    self.llm_logger.log_response(self.current_decision)
                    path = self.apply_decision(self.current_path, self.current_decision, distance)

            else:
                self.current_decision = "GO"
                path = self.apply_decision(self.current_path, self.current_decision, None)
        else:
            path = self.apply_decision(self.current_path, self.current_decision, None)

        self.publisher_path.publish(path)
        self.current_path = None
        self.traffic_signals = None
        self.current_pose = None

    def find_nearest_relevant_traffic_light(self):

        try:
            path_lane_ids = {p.lane_ids[0] for p in self.current_path.points if p.lane_ids}
            if not path_lane_ids:
                self.get_logger().info('no path lane ids')
                return None

            # self.get_logger().info(f'lane ids: {path_lane_ids}')
            vehicle_pos_3d = self.current_pose.pose.pose.position
            self.get_logger().info(f'vehicle 3d pose: {vehicle_pos_3d}')
            vehicle_pos_2d = BasicPoint2d(vehicle_pos_3d.x, vehicle_pos_3d.y)
            # vehicle_pos_2d = Point(vehicle_pos_3d.x, vehicle_pos_3d.y)
            self.get_logger().info(f'vehicle 2d pose: {vehicle_pos_2d}')
            min_distance = float('inf')
            nearest_light_info = None

            self.get_logger().info(f'len traffic signals: {len(self.traffic_signals.traffic_light_groups)}')
            for light_group in self.traffic_signals.traffic_light_groups:
                self.get_logger().info(f'light group info: {light_group}')
                traffic_light_id = light_group.traffic_light_group_id

                reg_elem = self.lanelet_map.regulatoryElementLayer.get(traffic_light_id)
                if not reg_elem:
                    self.get_logger().warn(f"Traffic light ID {traffic_light_id} from perception not found in map.")
                    continue

                print(f'reg elem: {reg_elem}')

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

    def build_prompt(self, color: str, distance: float, speed: float) -> str:

        prompt_template = f"""
            You are an autonomous driving decision-making assistant.
            The vehicle is approaching a traffic light.

            Current Situation:
            - Nearest Relevant Traffic Light Color: {color}
            - Distance to Stop Line: {distance:.1f} meters
            - Current Speed: {speed:.1f} m/s

            Decide whether the car should STOP or GO.
            A 'STOP' decision means planning a smooth stop before the stop line.
            A 'GO' decision means continuing at the planned speed.

            Output only one word: "STOP" or "GO".
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

    def apply_decision(self, path_msg, decision, stop_distance):
        new_path = Path()
        new_path.header = path_msg.header
        self.get_logger().info(f'decision is: {decision}')
        for point_with_lane in path_msg.points:
            path_point = PathPoint()
            path_point.pose = point_with_lane.point.pose

            if decision == "STOP":
                path_point.longitudinal_velocity_mps = 0.0
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

    def get_dominant_color(self, elements):
        for e in elements:
            if e.color == e.RED: return "RED"
            if e.color == e.AMBER: return "AMBER"
            if e.color == e.GREEN: return "GREEN"
        return "UNKNOWN"


def main(args=None):
    rclpy.init(args=args)
    node = LLMControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
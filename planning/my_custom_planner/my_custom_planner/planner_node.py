#!/usr/bin/env python3
import geometry_msgs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import tempfile
import time
import os
import threading
import json
import uuid

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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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


def calc_signed_arc_length(path_points, ego_pose, target_point_geo):
    # Convert path points to a list of (x,y) tuples
    path_2d = [(p.point.pose.position.x, p.point.pose.position.y) for p in path_points]

    # Create shapely LineString for the path
    if len(path_2d) < 2:
        return float('inf')  # Not enough points to form a path
    path_line = LineString(path_2d)

    ego_point = Point(ego_pose.position.x, ego_pose.position.y)
    target_point = Point(target_point_geo.x, target_point_geo.y)

    # Project ego position and target point onto the path
    # This gives distance along the path from its start
    ego_dist_on_path = path_line.project(ego_point)
    target_dist_on_path = path_line.project(target_point)

    # Calculate signed arc length
    signed_length = target_dist_on_path - ego_dist_on_path

    return signed_length

class LLMControlNode(Node):
    def __init__(self):
        super().__init__('my_custom_planner_node')

        self.lanelet_map = None
        self.projector = None
        self.current_path: PathWithLaneId = None
        self.current_pose: Odometry = None
        self.traffic_signals: TrafficLightGroupArray = None
        self.objects: PredictedObjects = None
        self.vehicle_initial_pose = None
        self.current_decision = "GO"
        self.chat_history = []
        self.vehicle_length = 4.0

        self.last_time_llm_used = 0
        self.threshold = 2.0
        self.data_lock = threading.Lock()
        self.decision_thread = threading.Thread(target=self.decision_making_loop, daemon=True)

        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.traffic_signal_sub = self.create_subscription(TrafficLightGroupArray, trafficLightGroupArrayTopic, self.traffic_light_callback, 10)
        self.path_sub = self.create_subscription(PathWithLaneId, pathWithLaneIdTopic, self.path_with_lane_id_callback,10)
        self.map_sub = self.create_subscription(LaneletMapBin, vectorMapTopic, self.map_callback, map_qos)
        self.odom_sub = self.create_subscription(Odometry, kinematicTopic, self.odometry_callback, 10)
        self.objects_sub = self.create_subscription(PredictedObjects, objectsTopic, self.objects_callback, 10)

        self.publisher_path = self.create_publisher(Path, pathTopic, 10)

        self.llm_logger = MyLogger()
        # self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.llm = ChatOpenAI(model="gpt-4o")

        self.publishing_timer = self.create_timer(0.1, self.path_publishing_loop)
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

    def path_with_lane_id_callback(self, msg: PathWithLaneId):
        with self.data_lock:
            self.current_path = msg

    def odometry_callback(self, msg: Odometry):
        with self.data_lock:
            self.current_pose = msg
        if self.vehicle_initial_pose is None:
            self.vehicle_initial_pose = msg.pose.pose.position

    def objects_callback(self, msg: PredictedObjects):
        with self.data_lock:
            self.objects = msg

    def decision_making_loop(self):
        while rclpy.ok():
            time.sleep(self.threshold)

            with self.data_lock:
                if not all(
                        [self.lanelet_map, self.current_path, self.current_pose]):
                    self.get_logger().warn("Decision loop: waiting for data...", throttle_duration_sec=5)
                    continue

                # Copy data to process outside the lock
                path_copy = self.current_path
                pose_copy = self.current_pose
                signals_copy = self.traffic_signals
                objects_copy = self.objects

            nearest_light_info = self.find_nearest_relevant_traffic_light(path_copy, pose_copy, signals_copy)
            recognized_objects_str = self.get_recognized_objects_for_prompt(objects_copy)
            ego_path = self.get_ego_path_for_prompt(path_copy)

            new_decision = "GO"  # Default to GO if no relevant light is found
            if ego_path and (nearest_light_info or recognized_objects_str):
                prompt = self.build_prompt(
                    color=nearest_light_info['color'] if nearest_light_info else "UNKNOWN",
                    distance=nearest_light_info['distance'] if nearest_light_info else -1.0,
                    speed=pose_copy.twist.twist.linear.x,
                    ego_path_info=ego_path,
                    objects_info=recognized_objects_str
                )
                # self.get_logger().info(f"Querying LLM with prompt:\n{prompt}")
                response = self.query_llm(prompt)

                if response and response.strip().startswith("```json"):
                    cleaned_str = response.strip().replace("```json", "").replace("```", "")
                else:
                    cleaned_str = response

                try:
                    # Parse the JSON response
                    response_json = json.loads(cleaned_str)
                    command = response_json.get("command", "GO").upper()
                    reason = response_json.get("reason", "No reason provided.")

                    if command in ["GO", "STOP"]:
                        new_decision = command
                        # self.llm_logger.log_response(f"Decision: {command}, Reason: {reason}")
                        self.get_logger().info(f"LLM Response: {command}, Reason: {reason}")
                    else:
                        # self.llm_logger.log_error(f"Invalid command in response: {response}")
                        self.get_logger().warn(f"LLM returned invalid command: '{command}'. Defaulting to GO.")

                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    self.get_logger().error(f"Failed to parse LLM JSON response: {e}. Full response: '{response}'")
                    self.llm_logger.log_error(f"JSON parsing error: {response}")

                # if response and response.strip().upper() in ["GO", "STOP"]:
                #     new_decision = response.strip().upper()
                #     self.get_logger().info(f"LLM response: {self.current_decision}")
                #     self.llm_logger.log_response(self.current_decision)
                # else:
                #     self.get_logger().warn(f"Received invalid LLM response: '{response}'. Defaulting to GO.")

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

    def find_nearest_relevant_traffic_light(self, current_path: PathWithLaneId, current_pose: Odometry,
                                    traffic_signals: TrafficLightGroupArray):

        # vehicle_pos = current_pose.pose.pose.position
        min_distance = float('inf')
        relevant_light_info = None
        path_lane_ids = {p_with_lane.lane_ids[0] for p_with_lane in current_path.points if p_with_lane.lane_ids}

        if not path_lane_ids:
            self.get_logger().debug("No lane IDs found in the current path.")
            return None

        for lane_id in path_lane_ids:
            try:
                lanelet = self.lanelet_map.laneletLayer.get(lane_id)
            except Exception:
                self.get_logger().debug(f"Lanelet with ID {lane_id} not found in map.")
                continue

            # self.get_logger().info(f'lanelet.trafficLights(): {lanelet.trafficLights()}')
            self.get_logger().info(f'len traffic signals: {len(traffic_signals.traffic_light_groups)}')
            self.get_logger().info(f'len traffic signals in lanelet {lanelet} : {len(lanelet.trafficLights())}')

            for reg_elem in lanelet.trafficLights():
                if not reg_elem.stopLine:
                    self.get_logger().warn(
                        f"Traffic light regulatory element {reg_elem.id} has no stop line defined in map. Skipping.")
                    continue

                stop_line_3d = reg_elem.stopLine
                stop_line_2d = to2D(stop_line_3d)
                self.get_logger().info(f'stop line 3d found: {stop_line_3d}')
                self.get_logger().info(f'stop line 2d: {stop_line_2d}')

                # Convert shapely LineString to Lanelet2 BasicPoint2d for distance calculation
                # Take the first point of the stop line for distance calculation, or its centroid
                if len(stop_line_2d) > 0:

                    shapely_stop_line = LineString([(p.x, p.y) for p in stop_line_2d])
                    centroid_point = shapely_stop_line.centroid
                    # Calculate signed arc length from current vehicle pose to the stop line along the path
                    # This is crucial for determining if the stop line is ahead of the vehicle
                    stop_line_point_geo = geometry_msgs.msg.Point()
                    stop_line_point_geo.x = centroid_point.x  # Use first point for simplicity
                    stop_line_point_geo.y = centroid_point.y
                    # Assuming Z is not critical for traffic light stop line
                    # stop_line_point_geo.z = stop_line_3d[0].z if len(stop_line_3d) > 0 else vehicle_pos.z
                    self.get_logger().info(f'stop line created position: {stop_line_point_geo}')

                    signed_dist = calc_signed_arc_length(current_path.points, current_pose.pose.pose,
                                                         stop_line_point_geo)

                    signed_dist -= self.vehicle_length
                    self.get_logger().info(f'signed dist: {signed_dist}')


                    # Only consider traffic lights whose stop lines are ahead of the vehicle (positive signed distance)
                    # and are within a reasonable look-ahead range (e.g., 200 meters)
                    LOOK_AHEAD_DISTANCE = 100.0  # meters
                    if signed_dist > -0.5 and signed_dist < LOOK_AHEAD_DISTANCE:  # Allow slight overlap for robust detection
                        # Find the corresponding perceived traffic signal state
                        found_signal = None
                        for group in traffic_signals.traffic_light_groups:
                            if group.traffic_light_group_id == reg_elem.id:  # Match by regulatory element ID
                                found_signal = group
                                break

                        if found_signal:
                            color = self.get_dominant_color(
                                found_signal.elements[0].color)  # Assume first element is dominant
                            # Keep track of the *nearest* relevant traffic light ahead
                            if signed_dist < min_distance:
                                min_distance = signed_dist
                                relevant_light_info = {
                                    "distance": signed_dist,
                                    "color": color,
                                    "traffic_light_group_id": reg_elem.id  # Keep ID for debugging
                                }
                                self.get_logger().debug(
                                    f"Found relevant traffic light {reg_elem.id} at {signed_dist:.2f}m with color {color}")

        if relevant_light_info:
            self.get_logger().info(
                f"Nearest relevant traffic light: ID {relevant_light_info.get('traffic_light_group_id', 'N/A')}, Color: {relevant_light_info['color']}, Distance: {relevant_light_info['distance']:.2f}m")
        else:
            self.get_logger().info("No relevant traffic light found on path ahead.")

        return relevant_light_info


    def find_nearest_relevant_traffic_light2(self, current_path, current_pose, traffic_signals):

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
                            "color": self.get_dominant_color(light_group.elements[0].color)
                        }
            return nearest_light_info
        except Exception as e:
            self.get_logger().error(f"Error in find_nearest_relevant_traffic_light: {e}")
            return None

    def get_recognized_objects_for_prompt(self, objects_msg: PredictedObjects) -> str:
        if objects_msg.objects is None:
            self.get_logger().info('no objects in this frame')
            return 'no object exists in current frame'

        output = []
        for i, obj in enumerate(objects_msg.objects):
            class_label = None
            # path_str = None
            if obj.existence_probability < 0.3:
                continue

            if obj.classification:
                # Get the classification with the highest probability
                best_class = max(obj.classification, key=lambda c: c.probability)
                class_label = self.get_object_label(best_class.label)
            #     output.append(f"  Classification: {class_label} (Prob: {best_class.probability:.2f})\n")
            # else:
            #     output.append("  Classification: Not available\n")

            # --- Kinematics (Position, Velocity, Acceleration) ---
            best_path = max(obj.kinematics.predicted_paths, key=lambda p: p.confidence)

            kinematics = obj.kinematics
            pos = kinematics.initial_pose_with_covariance.pose.position
            pose_str = f"Position: ({pos.x - self.vehicle_initial_pose.x:.1f}, {pos.y - self.vehicle_initial_pose.y:.1f})"

            # if best_path.path:
                # Sample points: start, middle, end
                # path_points = best_path.path
                # num_points = len(path_points)
                # self.get_logger().info(f'num points for object {class_label}: {num_points}')
                # num_samples = 5
                # indices_to_pick = range(0, min(num_points, num_samples))
                # sampled_points_str = ", ".join(
                #     f"({path_points[i].position.x - self.vehicle_initial_pose.x:.1f}, {path_points[i].position.y - self.vehicle_initial_pose.y:.1f})"
                #     for i in indices_to_pick
                # )
                # path_str = f"Predicted Path (coords): [{sampled_points_str}]"

            uuid_bytes = bytes(obj.object_id.uuid)
            uuid_obj = uuid.UUID(bytes=uuid_bytes)
            object_id_str = str(uuid_obj)
            output.append(f"- Object {object_id_str}: {class_label}, {pose_str}")
        return "\n".join(output) if output else "No significant objects detected."

            #
            # pos = kinematics.initial_pose_with_covariance.pose.position
            # vel = kinematics.initial_twist_with_covariance.twist.linear
            # accel = kinematics.initial_acceleration_with_covariance.accel.linear
            #
            # output.append("  Kinematics:\n")
            # output.append(f"    Position (x, y, z):      ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) m\n")
            # output.append(f"    Linear Velocity (x, y, z): ({vel.x:.2f}, {vel.y:.2f}, {vel.z:.2f}) m/s\n")
            # output.append(f"    Linear Accel (x, y, z):    ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) m/s^2\n")
            #
            # # --- Shape ---
            # shape = obj.shape
            # dims = shape.dimensions
            # shape_type_map = {0: 'BOUNDING_BOX', 1: 'CYLINDER', 2: 'POLYGON'}
            # shape_type_str = shape_type_map.get(shape.type, 'UNKNOWN')
            #
            # output.append("  Shape:")
            # output.append(f"    Type: {shape_type_str}")
            # output.append(f"    Dimensions (l, w, h): ({dims.x:.2f}, {dims.y:.2f}, {dims.z:.2f}) m")
            # output.append("-" * 20 + "\n")

        # return "".join(output)

    def get_ego_path_for_prompt(self, path_msg: PathWithLaneId) -> str:
        if not path_msg or not path_msg.points:
            return "Ego vehicle path not available."

        path_points = path_msg.points
        num_points = len(path_points)
        self.get_logger().info(f'num points for vehicle: {num_points}')
        num_samples = 10  # Provide more detail for the ego path
        indices_to_pick = range(0, min(num_points, num_samples))

        # if num_points <= num_samples:
        #     indices_to_pick = range(num_points)
        # else:
        #     indices_to_pick = np.linspace(0, num_points - 1, num_samples, dtype=int)

        print(f'indices to pick for vehicle: {indices_to_pick}')
        formatted_path = []
        for i in indices_to_pick: #todo
            point = path_points[i].point
            formatted_path.append(
                f"  - Pos: ({point.pose.position.x - self.vehicle_initial_pose.x:.1f}, {point.pose.position.y - self.vehicle_initial_pose.y:.1f}), Vel: {point.longitudinal_velocity_mps:.1f} m/s"
            )
        return "Upcoming planned path points:\n" + "\n".join(formatted_path)

    def build_prompt(self, color: str, distance: float, speed: float, ego_path_info: str, objects_info: str) -> str:

        traffic_light_str = f'''
                    **2. Traffic Light Status:**
                    - Nearest relevant light color: {color}
                    - Distance to stop line: {distance:.1f} meters
                '''
        if color == "UNKNOWN" or distance == -1.0:  # If no relevant light was found
            traffic_light_str = "**2. Traffic Light Status:**\n- No relevant traffic light in the current path."

        prompt_template = f'''
            You are a highly constrained, deterministic logic engine for an autonomous vehicle. Your function is to apply a strict set of rules to the provided situation data and output a decision in JSON format.
    
            **Output Format:**
            Your output MUST be a JSON object with two keys: "command" and "reason".
            - "command": Must be the single word "GO" or "STOP".
            - "reason": A brief explanation of which rule and data points led to your decision.
    
            ---
            **SITUATION DATA:**
    
            **1. Ego Vehicle Status:**
            - Current Speed: {speed:.1f} m/s
            {ego_path_info}
            {traffic_light_str}
            **3. Nearby Objects:**
            {objects_info}
    
            ---
            **RULES (apply in this exact order of priority):**
    
            **Rule 1: Collision Avoidance (Highest Priority)**
            - Analyze the `Ego Vehicle Path` and each `Nearby Object's` predicted path.
            - If any object's predicted path is projected to intersect or come within 2.0 meters of the ego vehicle's path, your command MUST be **STOP**.
            - If this rule is triggered, state which object is the cause in your reason.
    
            **Rule 2: Traffic Light Adherence**
            - If Rule 1 did not result in a STOP, evaluate the traffic light:
            - **A) Green Light:** If the `Nearest relevant light color` is GREEN, the command is **GO**.
            - **B) Red or Amber Light:** If the `Nearest relevant light color` is RED or AMBER:
                - If the `Distance to stop line` is **less than 8.0 meters**, the command MUST be **STOP**.
                - If the `Distance to stop line` is **greater than or equal to 8.0 meters**, the command is **GO** (this allows the vehicle to safely approach the stop line).
            - **C) No Light:** If there is no relevant traffic light, proceed to Rule 3.
    
            **Rule 3: Default Action**
            - If no other rule has issued a command, the default command is **GO**.
    
            ---
            **TASK:**
            Based strictly on the rules above, analyze the situation data and provide your decision in the specified JSON format.
            '''
        # return prompt.strip()
        return prompt_template.strip()

    def query_llm(self, prompt):
        try:
            self.get_logger().info(f"time before llm call: {time.time()}")
            self.get_logger().info(f"Querying LLM with prompt:\n{prompt}")
            self.llm_logger.log_request(prompt)

            self.chat_history.append(HumanMessage(content=prompt))
            trimmed_history = self.chat_history[-6:]
            response = self.llm.invoke(trimmed_history)

            self.get_logger().info(f"time after llm call: {time.time()}")
            self.get_logger().info(f'LLM raw response: {response}')
            self.llm_logger.log_response(response)

            if hasattr(response, 'content'):
                self.chat_history.append(AIMessage(content=response.content))
                return response.content
            else:
                self.get_logger().warn(f"LLM response object has no 'content' attribute: {type(response)}")
                return None
        except Exception as e:
            self.get_logger().error(f"LLM error: {e}")
            self.llm_logger.log_error(f"LLM Invocation Error: {e}")
            return None

    def get_dominant_color(self, color_id):
        return {
            0: "UNKNOWN",
            1: "RED",
            2: "AMBER",
            3: "GREEN",
            4: "WHITE"
        }.get(color_id, "INVALID")

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


def main(args=None):
    rclpy.init(args=args)
    node = LLMControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import tempfile
import time
import os
import threading
import uuid
import math
import copy

import geometry_msgs
from autoware_planning_msgs.msg import Path, PathPoint
from tier4_planning_msgs.msg import PathWithLaneId
from autoware_perception_msgs.msg import TrafficLightGroupArray, PredictedObjects
from nav_msgs.msg import Odometry, OccupancyGrid
from autoware_map_msgs.msg import LaneletMapBin
import geometry_msgs.msg
from geometry_msgs.msg import AccelWithCovarianceStamped
from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import ParameterEvent

import lanelet2
from lanelet2.io import Origin, load
from lanelet2_extension_python.projection import MGRSProjector
from lanelet2.geometry import to2D, distance as lanelet_distance
from lanelet2.routing import RoutingGraph
from lanelet2.core import BasicPoint2d
import lanelet2_extension_python.utility.query as query
import lanelet2_extension_python.utility.utilities as utilities
from shapely.geometry import LineString, Point

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage

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

# output
pathTopic = '/planning/scenario_planning/lane_driving/behavior_planning/path'


class RosLoggerCallbackHandler(BaseCallbackHandler):
    def __init__(self, ros_logger):
        super().__init__()
        self.ros_logger = ros_logger
        self.file_logger = MyLogger()

    # def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
    #     full_prompt_log = f"\n--- AGENT PROMPT SENT TO LLM (Full Conversation) ---\n"
    #     for msg in messages[0]:
    #         full_prompt_log += f"[{msg.__class__.__name__}]\n{msg.content}\n\n"
    #
    #     full_prompt_log += "--------------------"
    #     self.ros_logger.info(full_prompt_log)
    #     for msg in reversed(messages[0]):
    #         if isinstance(msg, HumanMessage):
    #             self.file_logger.log_request(msg.content)
    #             break

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        human_prompt = "No human prompt found."
        for msg in reversed(messages[0]):
            if isinstance(msg, HumanMessage):
                human_prompt = msg.content
                break

        log_message = f"\n--- AGENT PROMPT SENT TO LLM ---\n{human_prompt}\n--------------------"
        self.ros_logger.info(log_message)
        self.file_logger.log_request(human_prompt)

    def on_llm_end(self, response, **kwargs: Any) -> Any:
        self.ros_logger.info(f"\n--- LLM RAW RESPONSE ---\n{response}\n--------------------")
        self.file_logger.log_response(str(response))

    def on_agent_action(self, action, **kwargs: Any) -> Any:
        self.ros_logger.info(f"\n--- AGENT ACTION ---\nTool: {action.tool}\nArgs: {action.tool_input}\n--------------------")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        error_message = f"LLM Error: {error}"
        self.ros_logger.error(error_message)
        self.file_logger.log_error(str(error))

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        self.ros_logger.info(f"--- TOOL OUTPUT ---\n{output}\n--------------------")

    def on_agent_finish(self, finish, **kwargs: Any) -> Any:
        self.ros_logger.info(f"--- AGENT FINAL RESPONSE ---\n{finish.return_values.get('output')}\n--------------------")


def calc_signed_arc_length(path_points, ego_pose, target_point_geo):
    path_2d = [(p.point.pose.position.x, p.point.pose.position.y) for p in path_points]
    if len(path_2d) < 2:
        return float('inf')

    path_line = LineString(path_2d)
    ego_point = Point(ego_pose.position.x, ego_pose.position.y)
    target_point = Point(target_point_geo.x, target_point_geo.y)
    ego_dist_on_path = path_line.project(ego_point)
    target_dist_on_path = path_line.project(target_point)
    signed_length = target_dist_on_path - ego_dist_on_path
    return signed_length


class SetDecisionArgs(BaseModel):
    decision: str
    reason: str


class CreateCurveArgs(BaseModel):
    direction: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    shift_distance: float
    reason: str



# ------------------ Tools as standalone functions ------------------
def make_set_driving_decision_tool(node: "LLMControlNode"):
    @tool(args_schema=SetDecisionArgs)
    def set_driving_decision(decision: str, reason: str):
        """
        sets the basic driving decision to GO or STOP. use this for simple cases like traffic lights or clear roads.
        'decision' must be either 'GO' or 'STOP'.
        'reason' is a short explanation for the decision.
        """
        with node.data_lock:
            if decision.upper() in ["GO", "STOP"]:
                node.get_logger().info(f"Tool Call: Setting decision to {decision.upper()}. Reason: {reason}")
                node.current_decision = decision.upper()
                node.hazard = False
                return f"Driving decision successfully set to {decision.upper()}."
            else:
                node.get_logger().warn(f"Invalid decision '{decision}' passed to set_driving_decision tool.")
                return f"Failed to set decision: Invalid value '{decision}' received."
    return set_driving_decision


def make_create_curve_tool(node: "LLMControlNode"):
    @tool(args_schema=CreateCurveArgs)
    def create_curve_maneuver(
        direction: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        shift_distance: float,
        reason: str,
    ):
        """
        generates a path with a curve to perform a maneuver like overtaking.
        'direction' must be 'left' or 'right'.
        'start_x' and 'start_y' define the beginning coordinates of the curve.
        'end_x' and 'end_y' define the ending coordinates of the curve.
        'shift_distance' is the lateral distance in meters to shift the path.
        'reason' is a short explanation for why the curve is needed.
        """
        with node.data_lock:
            if not node.current_path:
                node.get_logger().error("Cannot create curve: current_path is not available.")
                return

            node.get_logger().info(f"Tool Call: Creating a {direction} curve. Reason: {reason}")

            start_p = geometry_msgs.msg.Point(x=start_x, y=start_y)
            end_p = geometry_msgs.msg.Point(x=end_x, y=end_y)

            candidate_path = node.curve(
                path_msg=node.current_path,
                start_point=start_p,
                end_point=end_p,
                direction=direction,
                shift_distance=shift_distance,
            )
            node.last_candidate_path = candidate_path
            node.hazard = True
            return f"Successfully generated a {direction} curve maneuver path."
    return create_curve_maneuver


class LLMControlNode(Node):
    def __init__(self):
        super().__init__('my_custom_planner_node')

        ############################# STATUS VARIABLES #######################################
        self.lanelet_map = None
        self.projector = None
        self.current_path: PathWithLaneId = None
        self.current_pose: Odometry = None
        self.traffic_signals: TrafficLightGroupArray = None
        self.objects: PredictedObjects = None
        self.acceleration: AccelWithCovarianceStamped = None
        self.last_candidate_path: Path = None
        self.routing_graph: RoutingGraph = None
        self.vehicle_initial_pose = None
        self.current_decision = "GO"
        self.vehicle_length = 4.0
        self.hazard = False
        self.HAZARD_FINISH_THRESHOLD_POINTS = 5
        self.LLM_REQUEST_THRESHOLD = 4.0


        ############################# ROS2 configuration #######################################
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.traffic_signal_sub = self.create_subscription(TrafficLightGroupArray, trafficLightGroupArrayTopic, self.traffic_light_callback, 10)
        self.path_sub = self.create_subscription(PathWithLaneId, pathWithLaneIdTopic, self.path_with_lane_id_callback,10)
        self.map_sub = self.create_subscription(LaneletMapBin, vectorMapTopic, self.map_callback, map_qos)
        self.odom_sub = self.create_subscription(Odometry, kinematicTopic, self.odometry_callback, 10)
        self.objects_sub = self.create_subscription(PredictedObjects, objectsTopic, self.objects_callback, 10)
        self.acceleration_sub = self.create_subscription(AccelWithCovarianceStamped, accelerationTopic, self.acceleration_callback, 10)
        self.publisher_path = self.create_publisher(Path, pathTopic, 10)
        self.publishing_timer = self.create_timer(0.1, self.path_publishing_loop)


        ############################# LLM configuration #######################################
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", disable_streaming=False)
        # self.llm = ChatOpenAI(model="gpt-4o")

        system_message_content = """
        You are a highly constrained, deterministic logic engine for an autonomous vehicle. Your function is to apply a strict set of rules to the provided situation data and use the available tools to execute the decision.

        **RULES (apply in this exact order of priority):**

        **Rule 1: Collision Avoidance (Highest Priority)**
        - Analyze the `Ego Vehicle Path` and each `Nearby Object's` position.
        - If any object is projected to intersect or come within 1.0 meters of the ego vehicle's path, you MUST call a tool that results in a **STOP** or an avoidance maneuver by **CURVE**.
        - If this rule is triggered, state which object is the cause in your reason.

        **Rule 2: Traffic Light Adherence**
        - If Rule 1 was not triggered, evaluate the traffic light:
        - **A) Green Light:** If the `Nearest relevant light color` is GREEN, the decision should be **GO**.
        - **B) Red or Amber Light:** If the `Nearest relevant light color` is RED or AMBER:
            - If the `Distance to stop line` is **less than 8.0 meters**, the decision MUST be **STOP**.
            - If the `Distance to stop line` is **greater than or equal to 8.0 meters**, the decision is **GO** (to approach the stop line).
        - **C) No Light:** If there is no relevant traffic light, proceed to Rule 3.

        **Rule 3: Default Action**
        - If no other rule has issued a command, the default action is **GO**.

        **TASK:**
        Based strictly on the rules above, analyze the situation data and call the single most appropriate tool to execute your decision.
        You MUST always call a tool using the structured tool calling format.
        Do not output plain text or free-form JSON. Always return tool calls via the function calling interface.
        Never place tool calls inside the 'content' field. Use only the structured 'function_call' schema.

        """
        self.memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="history")
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message_content),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.set_driving_decision_tool = make_set_driving_decision_tool(self)
        self.create_curve_tool = make_create_curve_tool(self)
        self.tools = [self.set_driving_decision_tool, self.create_curve_tool]
        self.ros_callback_handler = RosLoggerCallbackHandler(self.get_logger())
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=1
        )


        ############################# THREADING #######################################
        self.data_lock = threading.Lock()
        self.decision_thread = threading.Thread(target=self.decision_making_loop, daemon=True)
        self.decision_thread.start()
        self.get_logger().info("LLM Control Node started. Waiting for data...")


    ############################# call back functions #######################################
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

            traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
            self.routing_graph = RoutingGraph(self.lanelet_map, traffic_rules)
            self.get_logger().info('Routing graph created successfully')
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

    def acceleration_callback(self, msg: AccelWithCovarianceStamped):
        with self.data_lock:
            self.acceleration = msg


    ############################# path publishing functions #######################################
    def path_publishing_loop(self):
        with self.data_lock:
            if not self.current_path:
                return
            path_to_process = self.current_path
            decision_to_apply = self.current_decision
            hazard_status = self.hazard

        if hazard_status:
            path_to_publish = self.adjust_candidate_path()
        else:
            path_to_publish = self.apply_decision(path_to_process, decision_to_apply)

        self.publisher_path.publish(path_to_publish)
        self.check_hazard_finish()


    def adjust_candidate_path(self):
        with self.data_lock:
            candidate_path = self.last_candidate_path
            current_pose = self.current_pose

        if not candidate_path or not candidate_path.points:
            self.get_logger().warn("adjust_candidate_path: No candidate path available to adjust.")
            return Path()

        if not current_pose:
            self.get_logger().warn("adjust_candidate_path: Current pose not available. Returning original candidate path.")
            return candidate_path

        vehicle_position = current_pose.pose.pose.position
        closest_index = self.find_closest_point_index(candidate_path.points, vehicle_position.x, vehicle_position.y)

        if closest_index == -1:
            self.get_logger().warn("Could not find a closest point on the candidate path. Returning original path.")
            return candidate_path

        new_path = Path()
        new_path.header = candidate_path.header
        remaining_points = candidate_path.points[closest_index:]

        if not remaining_points:
            self.get_logger().warn("Vehicle is past the end of the candidate path. Returning empty path.")
            return Path()

        new_path.points = remaining_points
        new_path_with_bounds = self.generate_bounds(new_path)
        self.get_logger().info(f"Adjusted candidate path: Trimmed {closest_index} points. Remaining: {len(new_path_with_bounds.points)} points.")
        return new_path_with_bounds


    def find_closest_point_index(self, path_points: list, target_x: float, target_y: float) -> int:
        if not path_points:
            return -1

        min_dist_sq = float('inf')
        closest_index = -1

        for i, point_object in enumerate(path_points):
            if hasattr(point_object, 'point'):
                position = point_object.point.pose.position
            else:
                position = point_object.pose.position

            dx = position.x - target_x
            dy = position.y - target_y
            dist_sq = dx ** 2 + dy ** 2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_index = i

        return closest_index


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


    def check_hazard_finish(self):
        with self.data_lock:
            if not self.hazard:
                return

            candidate_path = self.last_candidate_path
            current_pose = self.current_pose
            if not candidate_path or not candidate_path.points or not current_pose:
                return

            vehicle_position = current_pose.pose.pose.position
            closest_index = self.find_closest_point_index( candidate_path.points, vehicle_position.x, vehicle_position.y)
            if closest_index == -1:
                return

            remaining_points_count = len(candidate_path.points) - 1 - closest_index
            if remaining_points_count < self.HAZARD_FINISH_THRESHOLD_POINTS:
                self.get_logger().info(f"Hazard maneuver finished. Remaining points ({remaining_points_count}) are below threshold ({self.HAZARD_FINISH_THRESHOLD_POINTS}).")
                self.hazard = False
                self.last_candidate_path = None


    ############################# decision making functions #######################################
    def decision_making_loop(self):
        while rclpy.ok():
            time.sleep(self.LLM_REQUEST_THRESHOLD)

            with self.data_lock:
                if not all([self.lanelet_map, (self.current_path or self.last_candidate_path), self.current_pose]):
                    self.get_logger().warn("Decision loop: waiting for data...", throttle_duration_sec=5)
                    continue

                path_copy = self.current_path
                pose_copy = self.current_pose
                signals_copy = self.traffic_signals
                objects_copy = self.objects
                accel_copy = self.acceleration

                if self.hazard:
                    self.get_logger().warn("Decision loop: currently in hazard status...", throttle_duration_sec=5)
                    continue

            nearest_light_info = self.find_nearest_relevant_traffic_light(path_copy, pose_copy, signals_copy)
            recognized_objects_str = self.get_recognized_objects_for_prompt(objects_copy)
            ego_path = self.get_ego_path_for_prompt(path_copy)

            if ego_path and (nearest_light_info or recognized_objects_str):
                prompt = self.build_prompt(
                    color=nearest_light_info['color'] if nearest_light_info else "UNKNOWN",
                    distance=nearest_light_info['distance'] if nearest_light_info else -1.0,
                    speed=pose_copy.twist.twist.linear.x,
                    acceleration=accel_copy.accel.accel.linear.x,
                    ego_path_info=ego_path,
                    objects_info=recognized_objects_str
                )

                if prompt:
                    try:
                        self.agent_executor.invoke(
                            {"input": prompt},
                            config={"callbacks": [self.ros_callback_handler]}
                        )
                    except Exception as e:
                        self.get_logger().error(f"AgentExecutor invocation error: {e}")


    def find_nearest_relevant_traffic_light(self, current_path: PathWithLaneId, current_pose: Odometry,
                                    traffic_signals: TrafficLightGroupArray):

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

                if len(stop_line_2d) > 0:

                    shapely_stop_line = LineString([(p.x, p.y) for p in stop_line_2d])
                    centroid_point = shapely_stop_line.centroid
                    stop_line_point_geo = geometry_msgs.msg.Point()
                    stop_line_point_geo.x = centroid_point.x
                    stop_line_point_geo.y = centroid_point.y
                    self.get_logger().info(f'stop line created position: {stop_line_point_geo}')

                    signed_dist = calc_signed_arc_length(current_path.points, current_pose.pose.pose, stop_line_point_geo)
                    signed_dist -= self.vehicle_length
                    self.get_logger().info(f'signed dist: {signed_dist}')


                    LOOK_AHEAD_DISTANCE = 100.0  # meters
                    if signed_dist > -0.5 and signed_dist < LOOK_AHEAD_DISTANCE:  # Allow slight overlap for robust detection
                        found_signal = None
                        for group in traffic_signals.traffic_light_groups:
                            if group.traffic_light_group_id == reg_elem.id:
                                found_signal = group
                                break

                        if found_signal:
                            color = self.get_dominant_color(
                                found_signal.elements[0].color)
                            if signed_dist < min_distance:
                                min_distance = signed_dist
                                relevant_light_info = {
                                    "distance": signed_dist,
                                    "color": color,
                                    "traffic_light_group_id": reg_elem.id
                                }
                                self.get_logger().debug(
                                    f"Found relevant traffic light {reg_elem.id} at {signed_dist:.2f}m with color {color}")

        if relevant_light_info:
            self.get_logger().info(
                f"Nearest relevant traffic light: ID {relevant_light_info.get('traffic_light_group_id', 'N/A')}, Color: {relevant_light_info['color']}, Distance: {relevant_light_info['distance']:.2f}m")
        else:
            self.get_logger().info("No relevant traffic light found on path ahead.")

        return relevant_light_info


    def get_recognized_objects_for_prompt(self, objects_msg: PredictedObjects) -> str:
        if objects_msg.objects is None:
            self.get_logger().info('no objects in this frame')
            return 'no object exists in current frame'

        output = []
        for i, obj in enumerate(objects_msg.objects):
            class_label = None
            if obj.existence_probability < 0.3:
                continue

            if obj.classification:
                best_class = max(obj.classification, key=lambda c: c.probability)
                class_label = self.get_object_label(best_class.label)

            kinematics = obj.kinematics
            pos = kinematics.initial_pose_with_covariance.pose.position
            pose_str = f"Position: ({pos.x - self.vehicle_initial_pose.x:.1f}, {pos.y - self.vehicle_initial_pose.y:.1f})"

            uuid_bytes = bytes(obj.object_id.uuid)
            uuid_obj = uuid.UUID(bytes=uuid_bytes)
            object_id_str = str(uuid_obj)
            output.append(f"- Object {object_id_str}: {class_label}, {pose_str}")
        return "\n".join(output) if output else "No significant objects detected."


    def get_ego_path_for_prompt(self, path_msg: PathWithLaneId) -> str:
        if not path_msg or not path_msg.points:
            return "Ego vehicle path not available."

        path_points = path_msg.points
        num_points = len(path_points)
        self.get_logger().info(f'num points for vehicle: {num_points}')
        num_samples = 10  # Provide more detail for the ego path
        indices_to_pick = range(0, min(num_points, num_samples))
        self.get_logger().info(f'indices to pick for vehicle: {indices_to_pick}')
        formatted_path = []
        for i in indices_to_pick:
            point = path_points[i].point
            formatted_path.append(
                f"  - Pos: ({point.pose.position.x - self.vehicle_initial_pose.x:.1f}, {point.pose.position.y - self.vehicle_initial_pose.y:.1f}), Vel: {point.longitudinal_velocity_mps:.1f} m/s"
            )
        return "Upcoming planned path points:\n" + "\n".join(formatted_path)


    def build_prompt(self, color: str, distance: float, speed: float, acceleration: float, ego_path_info: str, objects_info: str) -> str:
        traffic_light_str = f'''
                    **2. Traffic Light Status:**
                    - Nearest relevant light color: {color}
                    - Distance to stop line: {distance:.1f} meters
                '''
        if color == "UNKNOWN" or distance == -1.0:  # If no relevant light was found
            traffic_light_str = "**2. Traffic Light Status:**\n- No relevant traffic light in the current path."

        situation_data = f"""
            **SITUATION DATA:**

            **1. Ego Vehicle Status:**
            - Current Speed: {speed:.1f} m/s
            - Current acceleration: {acceleration:.1f} m/s^2
            {ego_path_info}
            {traffic_light_str}
            **3. Nearby Objects:**
            {objects_info}
            """
        return situation_data.strip()


    def copy_path_points(self, source_points, dest_points):
        for point_with_lane in source_points:
            path_point = PathPoint()
            path_point.pose = copy.deepcopy(point_with_lane.point.pose)
            path_point.longitudinal_velocity_mps = point_with_lane.point.longitudinal_velocity_mps
            path_point.lateral_velocity_mps = point_with_lane.point.lateral_velocity_mps
            path_point.heading_rate_rps = point_with_lane.point.heading_rate_rps
            path_point.is_final = point_with_lane.point.is_final
            dest_points.append(path_point)
        return dest_points


    def curve(self, path_msg: PathWithLaneId, start_point: geometry_msgs.msg.Point, end_point: geometry_msgs.msg.Point, direction: str, shift_distance: float) -> Path:
        new_path = Path()
        new_path.header = path_msg.header

        if not path_msg.points:
            self.get_logger().warn("Input path is empty. Cannot generate a new path.")
            return new_path

        start_curve_index = self.find_closest_point_index(path_msg.points, start_point.x, start_point.y)
        end_curve_index = self.find_closest_point_index(path_msg.points, end_point.x, end_point.y)

        if start_curve_index == -1 or end_curve_index == -1 or start_curve_index >= end_curve_index:
            self.get_logger().warn("Could not find valid start or end points for the curve. Copying the original path.")
            new_path.points = self.copy_path_points(path_msg.points, [])
            new_path.left_bound = copy.deepcopy(path_msg.left_bound)
            new_path.right_bound = copy.deepcopy(path_msg.right_bound)
            return new_path

        generated_path_points = []
        generated_path_points = self.copy_path_points(path_msg.points[:start_curve_index], generated_path_points)

        turn_angle = math.pi / 2.0 if direction == "left" else -math.pi / 2.0
        turn_duration = end_curve_index - start_curve_index
        if turn_duration > 0:
            for i in range(start_curve_index, end_curve_index):
                point_with_lane = path_msg.points[i]
                path_point = PathPoint()
                path_point.pose = copy.deepcopy(point_with_lane.point.pose)
                turn_progress = (i - start_curve_index) / turn_duration
                shift_amount = shift_distance * math.sin(turn_progress * math.pi)
                orientation = path_point.pose.orientation
                siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
                cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                path_point.pose.position.x += shift_amount * math.cos(yaw + turn_angle)
                path_point.pose.position.y += shift_amount * math.sin(yaw + turn_angle)
                path_point.longitudinal_velocity_mps = point_with_lane.point.longitudinal_velocity_mps
                path_point.lateral_velocity_mps = point_with_lane.point.lateral_velocity_mps
                path_point.heading_rate_rps = point_with_lane.point.heading_rate_rps
                path_point.is_final = point_with_lane.point.is_final
                generated_path_points.append(path_point)

        stabilization_points_count = 10
        end_stabilization_index = min(end_curve_index + stabilization_points_count, len(path_msg.points))
        last_shifted_point = generated_path_points[-1]
        last_original_point = path_msg.points[end_curve_index - 1]
        shift_offset_x = last_shifted_point.pose.position.x - last_original_point.point.pose.position.x
        shift_offset_y = last_shifted_point.pose.position.y - last_original_point.point.pose.position.y

        for i in range(end_curve_index, end_stabilization_index):
            point_with_lane = path_msg.points[i]
            path_point = self.copy_path_points([point_with_lane], [])[0]
            path_point.pose.position.x += shift_offset_x
            path_point.pose.position.y += shift_offset_y
            generated_path_points.append(path_point)

        # generated_path_points = self.copy_path_points(path_msg.points[end_curve_index:], generated_path_points)
        new_path.points = generated_path_points
        new_path = self.generate_bounds(new_path)
        return new_path


    def generate_bounds(self, path: Path) -> Path:
        left_bound = []
        right_bound = []
        bound_distance = 3.0

        for path_point in path.points:
            orientation = path_point.pose.orientation
            siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
            cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            left_point = geometry_msgs.msg.Point()
            left_point.x = path_point.pose.position.x + bound_distance * math.cos(yaw - math.pi / 2.0)
            left_point.y = path_point.pose.position.y + bound_distance * math.sin(yaw - math.pi / 2.0)
            left_point.z = path_point.pose.position.z
            left_bound.append(left_point)

            right_point = geometry_msgs.msg.Point()
            right_point.x = path_point.pose.position.x + bound_distance * math.cos(yaw + math.pi / 2.0)
            right_point.y = path_point.pose.position.y + bound_distance * math.sin(yaw + math.pi / 2.0)
            right_point.z = path_point.pose.position.z
            right_bound.append(right_point)

        path.left_bound = left_bound
        path.right_bound = right_bound
        return path


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
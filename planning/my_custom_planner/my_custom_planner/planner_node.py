#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from autoware_internal_planning_msgs.msg import PathWithLaneId, VelocityLimit
from autoware_planning_msgs.msg import Path, PathPoint
from autoware_perception_msgs.msg import TrafficLightGroupArray, PredictedObjects
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from autoware_map_msgs.msg import LaneletMapBin
from sensor_msgs.msg import PointCloud2
from rosgraph_msgs.msg import Clock
from rcl_interfaces.msg import ParameterEvent


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from .my_logger import MyLogger
import json
import time

#input
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
velocityLimitTopic = '/planning/scenario_planning/max_velocity_default'

#output
pathTopic = '/planning/scenario_planning/lane_driving/behavior_planning/path'


class LLMControlNode(Node):
    def __init__(self):
        super().__init__('my_custom_planner_node')

        self.subscription_traffic_lights = self.create_subscription(
            TrafficLightGroupArray,
            trafficLightGroupArrayTopic,
            self.traffic_light_callback,
            10
        )

        self.subscription_path_with_lane_id = self.create_subscription(
            PathWithLaneId,
            pathWithLaneIdTopic,
            self.path_with_lane_id_callback,
            10
        )

        self.publisher_path = self.create_publisher(
            Path,
            pathTopic,
            10
        )

        self.llm_logger = MyLogger()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        self.decision_timer = self.create_timer(0.1, self.decision_timer_callback)

        self.latest_traffic_light_data = None
        self.latest_path_with_lane_id = None 
        self.last_time_llm_used = 0
        self.threshold = 4
        self.current_decision = "GO"
        self.get_logger().info("LLMTrafficLightControlNode started. Waiting for data...")

    
    def traffic_light_callback(self, msg: TrafficLightGroupArray):
        self.latest_traffic_light_data = msg
        self.get_logger().info("Received TrafficLightGroupArray message.")


    def path_with_lane_id_callback(self, msg: PathWithLaneId):
        self.latest_path_with_lane_id = msg
        self.get_logger().info("Received PathWithLaneId message.")


    def decision_timer_callback(self):
        if self.latest_path_with_lane_id is None:
            self.get_logger().info("Waiting for path data.")
            return
        
        path = None
        now = time.time()
        if now - self.last_time_llm_used >= self.threshold:
            self.last_time_llm_used = now
            prompt = self.build_prompt(self.latest_traffic_light_data, self.latest_path_with_lane_id)
            self.get_logger().info(f"LLM input prompt: {prompt}")
            self.llm_logger.log_request(prompt)
            response = self.query_llm(prompt)
            if response:
                self.current_decision = response.strip().upper()
                
                self.get_logger().info(f"LLM response: {self.current_decision}")
                self.llm_logger.log_response(self.current_decision)
                path = self.apply_decision(self.latest_path_with_lane_id, self.current_decision)

        else:
            path = self.apply_decision(self.latest_path_with_lane_id, self.current_decision)
        
        self.publisher_path.publish(path)
        self.latest_path_with_lane_id = None
        self.latest_traffic_light_data = None        
 


    def build_prompt(self, traffic_light_data: TrafficLightGroupArray, path_data: PathWithLaneId) -> str:
        def get_color_name(color_id):
            return {
                0: "UNKNOWN",
                1: "RED",
                2: "AMBER",
                3: "GREEN",
                4: "WHITE"
            }.get(color_id, "INVALID")

        # def get_shape_name(shape_id):
        #     return {
        #         1: "CIRCLE",
        #         2: "LEFT_ARROW",
        #         3: "RIGHT_ARROW",
        #         4: "UP_ARROW",
        #         5: "UP_LEFT_ARROW",
        #         6: "UP_RIGHT_ARROW",
        #         7: "DOWN_ARROW",
        #         8: "DOWN_LEFT_ARROW",
        #         9: "DOWN_RIGHT_ARROW",
        #         10: "CROSS"
        #     }.get(shape_id, "UNKNOWN")

        # def get_status_name(status_id):
        #     return {
        #         1: "SOLID_OFF",
        #         2: "SOLID_ON",
        #         3: "FLASHING"
        #     }.get(status_id, "UNKNOWN")


        lights = []
        for group in traffic_light_data.traffic_light_groups:
            elements = [
                {
                    "color": get_color_name(e.color),
                    # "shape": get_shape_name(e.shape),
                    # "status": get_status_name(e.status),
                    # "confidence": round(e.confidence, 2)
                } for e in group.elements
            ]
            lights.append({
                "group_id": group.traffic_light_group_id,
                "elements": elements
            })

        prompt_template = f"""
            You are an autonomous driving decision-making assistant. You will receive information about traffic lights

            Decide whether the car should STOP or GO based on the current traffic lights.

            Output only one word: "STOP" or "GO".
            Traffic Lights:
            {json.dumps(lights, indent=2)}
        """

        return prompt_template
    
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
        

    def apply_decision(self, path_msg, decision):
        new_path = Path()
        new_path.header = path_msg.header

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


def main(args=None):
    rclpy.init(args=args)
    node = LLMControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

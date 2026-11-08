"""ros2 launch FireDepth/fire_depth.launch.py camera_x:=0.07 camera_z:=0.12

If the robot's URDF already has the camera frames (TurtleBot3 Waffle does), use
publish_camera_tf:=false optical_frame:=<the URDF's optical frame>.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NODE_SCRIPT = Path(__file__).resolve().parent / "fire_depth_node.py"
# ROS runs on the system Python; the models need the venv that has torch and Depth Anything.
VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv-ros" / "bin" / "python"


def generate_launch_description() -> LaunchDescription:
    settings = {name: LaunchConfiguration(name) for name in (
        "camera_x", "camera_y", "camera_z", "camera_yaw", "camera_pitch",
        "image_topic", "camera_info_topic", "target_frame", "optical_frame", "publish_camera_tf", "python")}
    publish_camera_tf = IfCondition(settings["publish_camera_tf"])
    return LaunchDescription([
        DeclareLaunchArgument("camera_x", default_value="0.0"),
        DeclareLaunchArgument("camera_y", default_value="0.0"),
        DeclareLaunchArgument("camera_z", default_value="0.1"),
        DeclareLaunchArgument("camera_yaw", default_value="0.0"),
        DeclareLaunchArgument("camera_pitch", default_value="0.0", description="Positive tilts the camera down."),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("target_frame", default_value="base_link"),
        DeclareLaunchArgument("optical_frame", default_value="camera_optical_frame"),
        DeclareLaunchArgument("publish_camera_tf", default_value="true"),
        DeclareLaunchArgument("python", default_value=str(VENV_PYTHON)),
        Node(package="tf2_ros", executable="static_transform_publisher", condition=publish_camera_tf, arguments=[
            "--x", settings["camera_x"], "--y", settings["camera_y"], "--z", settings["camera_z"],
            "--yaw", settings["camera_yaw"], "--pitch", settings["camera_pitch"],
            "--frame-id", "base_link", "--child-frame-id", "camera_link"]),
        # camera_link is x forward, z up; the optical frame is z forward, y down.
        Node(package="tf2_ros", executable="static_transform_publisher", condition=publish_camera_tf, arguments=[
            "--yaw", "-1.5707963", "--roll", "-1.5707963",
            "--frame-id", "camera_link", "--child-frame-id", "camera_optical_frame"]),
        ExecuteProcess(output="screen", cmd=[
            settings["python"], str(NODE_SCRIPT), "--ros-args",
            "-p", ["image_topic:=", settings["image_topic"]],
            "-p", ["camera_info_topic:=", settings["camera_info_topic"]],
            "-p", ["target_frame:=", settings["target_frame"]],
            "-p", ["optical_frame:=", settings["optical_frame"]]]),
    ])

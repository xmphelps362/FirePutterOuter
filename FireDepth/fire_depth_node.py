import math

import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from fire_depth import DEFAULT_FIRE_MODEL, FireDepth, focal_length_per_image_width

TRANSFORM_TIMEOUT = Duration(seconds=0.1)


class FireDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("fire_depth")
        image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        camera_info_topic = self.declare_parameter("camera_info_topic", "/camera/camera_info").value
        self._target_frame = self.declare_parameter("target_frame", "base_link").value
        self._optical_frame = self.declare_parameter("optical_frame", "").value
        horizontal_fov_degrees = self.declare_parameter("horizontal_fov_degrees", 65.0).value
        self._fallback_focal_length_per_width = focal_length_per_image_width(horizontal_fov_degrees)
        self._fire_depth = FireDepth(
            self.declare_parameter("fire_model", str(DEFAULT_FIRE_MODEL)).value,
            self.declare_parameter("confidence", 0.3).value,
            horizontal_fov_degrees,
            self.declare_parameter("processing_resolution", 392).value,
        )
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._frame_id = 0
        self._header_by_frame_id: dict[int, Header] = {}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._depth_publisher = self.create_publisher(Image, "/depth/image", 1)
        self._depth_info_publisher = self.create_publisher(CameraInfo, "/depth/camera_info", 1)
        self._fire_publisher = self.create_publisher(Detection3DArray, "/fire/detections", 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)

    def _on_camera_info(self, camera_info: CameraInfo) -> None:
        self._camera_info = camera_info

    def _on_image(self, image_message: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
        height, width = frame_bgr.shape[:2]
        if self._camera_info is not None and self._camera_info.k[0] > 0:
            fx, cx, fy, cy = (self._camera_info.k[i] for i in (0, 2, 4, 5))
        else:
            fx = fy = self._fallback_focal_length_per_width * width
            cx, cy = width / 2, height / 2
        fires, distances = self._fire_depth.measure(frame_bgr, self._frame_id, fx / width)
        camera_header = Header(stamp=image_message.header.stamp,
                               frame_id=self._optical_frame or image_message.header.frame_id)
        self._header_by_frame_id[self._frame_id] = camera_header
        self._frame_id += 1
        self._publish_new_depth()

        try:
            camera_to_target = self._tf_buffer.lookup_transform(
                self._target_frame, camera_header.frame_id, Time.from_msg(camera_header.stamp), TRANSFORM_TIMEOUT)
        except TransformException as error:
            self.get_logger().warn(f"No transform to {self._target_frame}: {error}", throttle_duration_sec=5.0)
            return

        fire_array = Detection3DArray()
        fire_array.header.stamp = image_message.header.stamp
        fire_array.header.frame_id = self._target_frame
        for fire, distance in zip(fires.detections, distances):
            if math.isnan(distance):
                continue
            u, v = fire.center
            # Optical frame convention: z forward, x right, y down.
            camera_point = PointStamped()
            camera_point.point.x = (u - cx) * distance / fx
            camera_point.point.y = (v - cy) * distance / fy
            camera_point.point.z = distance
            target_point = tf2_geometry_msgs.do_transform_point(camera_point, camera_to_target).point

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = fire.cls
            hypothesis.hypothesis.score = fire.confidence
            hypothesis.pose.pose.position = target_point
            detection = Detection3D()
            detection.header = fire_array.header
            detection.results.append(hypothesis)
            detection.bbox.center.position = target_point
            x1, y1, x2, y2 = fire.box
            detection.bbox.size.x = detection.bbox.size.y = (x2 - x1) * distance / fx
            detection.bbox.size.z = (y2 - y1) * distance / fy
            fire_array.detections.append(detection)
        self._fire_publisher.publish(fire_array)

    def _publish_new_depth(self) -> None:
        if self._fire_depth.newest_depth is None:
            return
        depth_frame_id, depth_meters = self._fire_depth.newest_depth
        if depth_frame_id not in self._header_by_frame_id:
            return
        # Depth finishes a few frames late, so it is stamped with the frame it was computed from.
        depth_header = self._header_by_frame_id[depth_frame_id]
        self._header_by_frame_id = {i: h for i, h in self._header_by_frame_id.items() if i > depth_frame_id}
        depth_message = self._bridge.cv2_to_imgmsg(depth_meters, "32FC1")
        depth_message.header = depth_header
        self._depth_publisher.publish(depth_message)
        if self._camera_info is not None:
            self._camera_info.header = depth_header
            self._depth_info_publisher.publish(self._camera_info)


def main() -> None:
    rclpy.init()
    rclpy.spin(FireDepthNode())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import rospy
import tf2_geometry_msgs
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32, Header, String

FIRE_DEPTH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIRE_DEPTH_DIR))
from fire_depth import (DEFAULT_FIRE_MODEL, FireDepth, FireDetector, bearing, camera_point,  # noqa: E402
                        detections_json, nearest_lit_candle)
from moge_depth import MogeDepthEstimator  # noqa: E402

DEPTH_MODEL = FIRE_DEPTH_DIR / "models" / "moge-2-vits.onnx"
TRANSFORM_TIMEOUT = rospy.Duration(0.1)


class FireDepthNode:
    def __init__(self) -> None:
        self._target_frame = rospy.get_param("~target_frame", "base_link")
        self._optical_frame = rospy.get_param("~optical_frame", "")
        self._depth_estimator = MogeDepthEstimator(str(DEPTH_MODEL))
        rospy.loginfo(f"YOLO: {DEFAULT_FIRE_MODEL}. Depth on {self._depth_estimator.providers[0]}.")
        fire_detector = FireDetector(str(DEFAULT_FIRE_MODEL), conf=rospy.get_param("~confidence", 0.3))
        self._fire_depth = FireDepth(fire_detector, self._depth_estimator)
        self._bridge = CvBridge()
        self._camera_info = None
        self._frame_id = 0
        self._header_by_frame_id = {}
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._detections_publisher = rospy.Publisher("fire/detections", String, queue_size=10)
        self._target_publisher = rospy.Publisher("fire/target", PointStamped, queue_size=10)
        self._bearing_publisher = rospy.Publisher("fire/bearing", Float32, queue_size=10)
        self._depth_publisher = rospy.Publisher("depth/image", Image, queue_size=1)
        self._depth_info_publisher = rospy.Publisher("depth/camera_info", CameraInfo, queue_size=1)
        rospy.Subscriber("usb_cam/camera_info", CameraInfo, self._on_camera_info)
        # Without a large buffer, rospy queues old images behind a slow callback.
        rospy.Subscriber("usb_cam/image_raw", Image, self._on_image, queue_size=1, buff_size=2 ** 24)

    def _on_camera_info(self, camera_info: CameraInfo) -> None:
        self._camera_info = camera_info

    def _on_image(self, image_message: Image) -> None:
        frame_bgr = self._bridge.imgmsg_to_cv2(image_message, "bgr8")
        height, width = frame_bgr.shape[:2]
        # usb_cam sends an all-zero K when it has no calibration file.
        calibrated = self._camera_info is not None and self._camera_info.K[0] > 0
        fires, distances = self._fire_depth.measure(
            frame_bgr, self._frame_id, self._camera_info.K[0] / width if calibrated else None)
        camera_header = Header(stamp=image_message.header.stamp,
                               frame_id=self._optical_frame or image_message.header.frame_id)
        self._header_by_frame_id[self._frame_id] = camera_header
        self._frame_id += 1
        self._publish_new_depth()

        positions = [None] * len(fires.detections)
        if any(math.isfinite(distance) for distance in distances):
            if calibrated:
                fx, cx, fy, cy = (self._camera_info.K[i] for i in (0, 2, 4, 5))
            else:
                fx = fy = self._depth_estimator.focal_length_per_width * width
                cx, cy = width / 2, height / 2
            try:
                camera_to_target = self._tf_buffer.lookup_transform(
                    self._target_frame, camera_header.frame_id, camera_header.stamp, TRANSFORM_TIMEOUT)
                for i, (fire, distance) in enumerate(zip(fires.detections, distances)):
                    if math.isfinite(distance):
                        point = PointStamped(header=camera_header)
                        point.point.x, point.point.y, point.point.z = camera_point(fire, distance, fx, fy, cx, cy)
                        target_point = tf2_geometry_msgs.do_transform_point(point, camera_to_target).point
                        positions[i] = (target_point.x, target_point.y, target_point.z)
            except tf2_ros.TransformException as error:
                rospy.logwarn_throttle(5, f"No transform from {camera_header.frame_id} to {self._target_frame}: {error}")

        self._detections_publisher.publish(String(detections_json(fires, distances, positions, self._target_frame)))
        target = nearest_lit_candle(fires.detections, distances)
        self._bearing_publisher.publish(Float32(math.nan if target is None else bearing(fires.detections[target], width)))
        if target is not None and positions[target] is not None:
            target_point = PointStamped(header=Header(stamp=camera_header.stamp, frame_id=self._target_frame))
            target_point.point.x, target_point.point.y, target_point.point.z = positions[target]
            self._target_publisher.publish(target_point)

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


if __name__ == "__main__":
    rospy.init_node("fire_depth")
    FireDepthNode()
    rospy.spin()

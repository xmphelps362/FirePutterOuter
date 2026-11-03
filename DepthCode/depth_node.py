import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_estimator import MetricDepthEstimator


class DepthNode(Node):
    def __init__(self) -> None:
        super().__init__("depth_node")
        fov = self.declare_parameter("horizontal_fov_degrees", 65.0).value
        resolution = self.declare_parameter("processing_resolution", 392).value
        image_topic = self.declare_parameter("image_topic", "/camera/image_raw").value
        self._bridge = CvBridge()
        self._estimator = MetricDepthEstimator(fov, resolution)
        self._publisher = self.create_publisher(Image, "/depth", 1)
        self.create_subscription(Image, image_topic, self._on_image, 1)

    def _on_image(self, image_message: Image) -> None:
        depth_meters = self._estimator.estimate_meters(self._bridge.imgmsg_to_cv2(image_message, "bgr8"))
        depth_message = self._bridge.cv2_to_imgmsg(depth_meters.astype("float32"), "32FC1")
        depth_message.header = image_message.header
        self._publisher.publish(depth_message)


def main() -> None:
    rclpy.init()
    rclpy.spin(DepthNode())


if __name__ == "__main__":
    main()

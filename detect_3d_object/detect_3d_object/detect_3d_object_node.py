import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_ros
import geometry_msgs.msg
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class ObjectDetectionOverlay(Node):
    def __init__(self):
        super().__init__('object_detection_overlay')
        self._force_sim_time()

        self.sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.image_pub_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.local_goal_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        # Subscribe to the camera image, camera info, and detection topics
        self.color_image_sub = self.create_subscription(
            Image, '/color/image', self.color_image_callback, self.sensor_qos)
        self.depth_image_sub = self.create_subscription(
            Image, '/depth/image', self.depth_image_callback, self.sensor_qos)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/depth/camera_info', self.camera_info_callback, self.sensor_qos)
        self.detections_sub = self.create_subscription(
            Detection2DArray, 'detector_node/detections', self.detection_callback, self.sensor_qos)

        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Publish detected Image resized to one-eighth resolution
        self.overlay_topic_base = 'image_containing_object'
        self.image_pub_eighth = self.create_publisher(
            Image, f'{self.overlay_topic_base}/eighth', self.image_pub_qos)
        self.image_pub_eighth_compressed = self.create_publisher(
            CompressedImage, f'{self.overlay_topic_base}/eighth/compressed', self.image_pub_qos)
        self.image_info_pub = self.create_publisher(
            CameraInfo, f'{self.overlay_topic_base}/camera_info', self.sensor_qos)
        self.local_goal_pub = self.create_publisher(
            geometry_msgs.msg.PointStamped, 'local_goal_point', self.local_goal_qos)
        self.local_goal_publishers = {}
        self.downscale_factor = 0.125
        self.label_font_scale = 0.25
        self.label_font_thickness = 1


        # Initialize other variables
        self.bridge = CvBridge()
        self.detections = None
        self.color_image = None
        self.image_header = None
        self.depth_image = None  # Original depth image for 3D calculations
        self.camera_info = None  # Camera intrinsic parameters

        overlay_mode_param = self.declare_parameter('overlay_color_mode', 'rgb')
        self.overlay_color_mode = overlay_mode_param.get_parameter_value().string_value.lower()
        if self.overlay_color_mode not in ('rgb', 'gray'):
            self.get_logger().warn(
                f"Invalid overlay_color_mode '{self.overlay_color_mode}', falling back to 'rgb'.")
            self.overlay_color_mode = 'rgb'

        self.declare_parameter('parent_frame', 'base_link')
        # Object class that should generate a local goal point
        self.declare_parameter('goal_object_class', 'suitcase')
        # Optional identifier (e.g., "person_1") to pick a specific instance
        self.declare_parameter('goal_object_identifier', 'person_1')


    def camera_info_callback(self, msg):
        # Store the camera info for intrinsic parameters
        self.camera_info = msg

    def depth_image_callback(self, msg):
        try:
            # Convert the depth image to a float32 format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {str(e)}")
            return

        self.depth_image = depth_image  # Original depth image for 3D calculations

        if self.detections is not None and self.color_image is not None:
            self.overlay_detections()

    def color_image_callback(self, msg):
        try:
            # Convert incoming image to BGR for OpenCV processing
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            try:
                color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting color image: {str(e)}")
                return

        if color_image.ndim == 2:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
        elif color_image.shape[2] == 4:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        self.color_image = color_image
        self.image_header = msg.header

        self.get_logger().info('Try Image Get...')

        if self.detections is not None and self.depth_image is not None:
            self.overlay_detections()

    def detection_callback(self, msg):
        # Filter detections based on confidence
        filtered_detections = []
        for detection in msg.detections:
            # Assuming the first result contains the hypothesis with the highest confidence
            if detection.results[0].hypothesis.score >= 0.80:  # 80% confidence threshold
                filtered_detections.append(detection)

        # Store the filtered detections
        self.detections = Detection2DArray(detections=filtered_detections)
        
        self.get_logger().info(f'Detection: {self.detections}.')

        if self.color_image is not None and self.depth_image is not None:
            self.overlay_detections()
        
    def overlay_detections(self):
        if self.color_image is None or self.camera_info is None or self.depth_image is None:
            return

        # Get the intrinsic parameters from camera_info
        fx = self.camera_info.k[0]  # Focal length in x
        fy = self.camera_info.k[4]  # Focal length in y
        cx = self.camera_info.k[2]  # Optical center in x
        cy = self.camera_info.k[5]  # Optical center in y

        # Create overlays for full and heavily downscaled outputs
        overlay_image = self.color_image.copy()
        if overlay_image.ndim == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2BGR)
        elif overlay_image.shape[2] == 4:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2BGR)

        if self.overlay_color_mode == 'gray':
            gray_overlay = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
            overlay_image = cv2.cvtColor(gray_overlay, cv2.COLOR_GRAY2BGR)

        overlay_image_downscaled = cv2.resize(
            overlay_image, (0, 0), fx=self.downscale_factor, fy=self.downscale_factor, interpolation=cv2.INTER_AREA)
        downscaled_text_thickness = max(1, int(self.label_font_thickness * self.downscale_factor))

        target_goal_class = self.get_parameter('goal_object_class').get_parameter_value().string_value.strip().lower()
        goal_identifier_param = self.get_parameter('goal_object_identifier').get_parameter_value().string_value.strip()
        selected_instance_index = None
        if goal_identifier_param:
            identifier_lower = goal_identifier_param.lower()
            class_part, sep, maybe_index = identifier_lower.rpartition('_')
            if sep and maybe_index.isdigit():
                if class_part:
                    target_goal_class = class_part
                selected_instance_index = max(1, int(maybe_index))
            else:
                target_goal_class = identifier_lower

        object_instances = {}
        selected_instance_found = False

        for detection_idx, detection in enumerate(self.detections.detections):
            bbox = detection.bbox
            results = detection.results[0]  # Assuming only one result per detection
            detection_class = str(results.hypothesis.class_id)
            self.get_logger().info(f'Detection: {detection_class}.')


            # Calculate the bounding box in pixel coordinates
            x_min = int(bbox.center.position.x - (bbox.size_x / 2))
            y_min = int(bbox.center.position.y - (bbox.size_y / 2))
            x_max = int(bbox.center.position.x + (bbox.size_x / 2))
            y_max = int(bbox.center.position.y + (bbox.size_y / 2))

            # Draw the bounding box
            cv2.rectangle(overlay_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            x_min_downscaled = int(x_min * self.downscale_factor)
            y_min_downscaled = int(y_min * self.downscale_factor)
            x_max_downscaled = int(x_max * self.downscale_factor)
            y_max_downscaled = int(y_max * self.downscale_factor)
            cv2.rectangle(
                overlay_image_downscaled,
                (x_min_downscaled, y_min_downscaled),
                (x_max_downscaled, y_max_downscaled),
                (0, 255, 0),
                max(1, int(2 * self.downscale_factor))
            )

            # Calculate the centroid of the bounding box
            centroid_x = int(bbox.center.position.x)
            centroid_y = int(bbox.center.position.y)

            # Draw the centroid
            cv2.circle(overlay_image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            centroid_x_downscaled = int(centroid_x * self.downscale_factor)
            centroid_y_downscaled = int(centroid_y * self.downscale_factor)
            cv2.circle(
                overlay_image_downscaled,
                (centroid_x_downscaled, centroid_y_downscaled),
                max(1, int(5 * self.downscale_factor)),
                (0, 0, 255),
                -1
            )

            # Get the depth at the centroid from the original depth image
            depth_value = self.depth_image[centroid_y, centroid_x]  # Use depth_image, not display_image

            # If depth is valid (not NaN or infinity)
            if np.isscalar(depth_value) and not np.isnan(depth_value) and not np.isinf(depth_value):
                # Calculate the 3D position in the camera_depth_optical_frame
                depth = (depth_value) / 1000
                x = depth #image depth = x of base_frame
                y = -(centroid_x - cx) * depth / fx #image x = -y of base_frame
                z = (centroid_y - cy) * depth / fy #image y = z of base_frame 

                # Queue TF/local goal publish with instance tracking
                object_class_lower = detection_class.lower()
                distance = float(np.linalg.norm([x, y, z]))
                should_publish_goal = bool(target_goal_class and object_class_lower == target_goal_class)
                instance_info = {
                    'object_class': detection_class,
                    'position': (x, y, z),
                    'distance': distance,
                    'order': detection_idx,
                    'publish_goal': should_publish_goal,
                }
                object_instances.setdefault(object_class_lower, []).append(instance_info)
                

                # Overlay the 3D position on the image
                label_3d = f"X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f} m"
                cv2.putText(overlay_image, label_3d, (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, (255, 255, 0), self.label_font_thickness)
                cv2.putText(
                    overlay_image_downscaled,
                    label_3d,
                    (x_min_downscaled, int((y_max + 20) * self.downscale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.label_font_scale,
                    (255, 255, 0),
                    downscaled_text_thickness
                )

            # Overlay the detected object's name and confidence level
            object_name = detection_class
            confidence = results.hypothesis.score * 100  # Confidence in percentage
            label = f"{object_name}: {confidence:.2f}%"

            # Put the label above the bounding box
            cv2.putText(overlay_image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, self.label_font_scale, (0, 255, 0), self.label_font_thickness)
            cv2.putText(
                overlay_image_downscaled,
                label,
                (x_min_downscaled, int((y_min - 10) * self.downscale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.label_font_scale,
                (0, 255, 0),
                downscaled_text_thickness
            )

        # Convert image to ros msg
        ros_downscaled_image = self.bridge.cv2_to_imgmsg(overlay_image_downscaled, encoding="bgr8")
        if self.image_header is not None:
            ros_downscaled_image.header = self.image_header

        self.image_pub_eighth.publish(ros_downscaled_image)
        self.publish_downscaled_camera_info()

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        success, encoded_image = cv2.imencode('.jpg', overlay_image_downscaled, encode_params)
        if success:
            compressed_msg = CompressedImage()
            if self.image_header is not None:
                compressed_msg.header = self.image_header
            compressed_msg.format = 'jpeg'
            compressed_msg.data = encoded_image.tobytes()
            self.image_pub_eighth_compressed.publish(compressed_msg)

        # Publish TFs/local goals after sorting instances per class by distance
        for class_key, instances in object_instances.items():
            sorted_instances = sorted(instances, key=lambda entry: (entry['distance'], entry['order']))
            for instance_index, entry in enumerate(sorted_instances, start=1):
                x, y, z = entry['position']
                self.publish_tf(x, y, z, entry['object_class'], instance_index=instance_index)
                if entry['publish_goal']:
                    publish_on_base_topic = False
                    if selected_instance_index is None:
                        publish_on_base_topic = True
                    elif instance_index == selected_instance_index:
                        publish_on_base_topic = True
                        selected_instance_found = True
                    self.publish_local_goal(
                        x,
                        y,
                        z,
                        entry['object_class'],
                        instance_index,
                        publish_on_base_topic=publish_on_base_topic
                    )

        if goal_identifier_param and selected_instance_index is not None and not selected_instance_found:
            self.get_logger().warn(
                f"Goal identifier '{goal_identifier_param}' not found; no base local_goal_point published.")

        self.get_logger().info("Published overlay images")

        # Show the image with OpenCV
        # cv2.imshow("Detections Overlay", overlay_image)
        # cv2.waitKey(1)

    def publish_tf(self, x, y, z, object_name, instance_index=None):
        parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        # Create a TransformStamped message
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame  # Parent frame
        suffix = f"_{instance_index}" if instance_index is not None else ""
        t.child_frame_id = f"{object_name}_frame{suffix}"  # Ensure unique TF frame names per object
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0  # Default rotation (no rotation)

        # Publish the transform
        self.tf_broadcaster.sendTransform(t)

    def publish_local_goal(self, x, y, z, object_name, instance_index, publish_on_base_topic=True):
        parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value
        point_msg = geometry_msgs.msg.PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = parent_frame
        point_msg.point.x = float(x)
        point_msg.point.y = float(y)
        point_msg.point.z = float(z)
        # Publish on base topic when the selected instance matches criteria
        if publish_on_base_topic:
            self.local_goal_pub.publish(point_msg)

        # Publish on per-object topic (e.g., local_goal_point/suitcase_frame_1)
        topic_suffix = f"{object_name}_frame_{instance_index}"
        if topic_suffix not in self.local_goal_publishers:
            self.local_goal_publishers[topic_suffix] = self.create_publisher(
                geometry_msgs.msg.PointStamped,
                f"local_goal_point/{topic_suffix}",
                self.local_goal_qos
            )
        self.local_goal_publishers[topic_suffix].publish(point_msg)

    def publish_downscaled_camera_info(self):
        if self.camera_info is None:
            return

        downscaled_info = CameraInfo()
        header = self.image_header if self.image_header is not None else self.camera_info.header
        downscaled_info.header = header
        downscaled_info.height = max(1, int(self.camera_info.height * self.downscale_factor))
        downscaled_info.width = max(1, int(self.camera_info.width * self.downscale_factor))
        downscaled_info.distortion_model = self.camera_info.distortion_model
        downscaled_info.d = list(self.camera_info.d)
        downscaled_info.r = list(self.camera_info.r)
        downscaled_info.binning_x = self.camera_info.binning_x
        downscaled_info.binning_y = self.camera_info.binning_y
        downscaled_info.roi = self.camera_info.roi

        # Scale the intrinsic parameters for the resized image
        downscaled_info.k = list(self.camera_info.k)
        downscaled_info.k[0] *= self.downscale_factor
        downscaled_info.k[2] *= self.downscale_factor
        downscaled_info.k[4] *= self.downscale_factor
        downscaled_info.k[5] *= self.downscale_factor

        downscaled_info.p = list(self.camera_info.p)
        downscaled_info.p[0] *= self.downscale_factor
        downscaled_info.p[2] *= self.downscale_factor
        downscaled_info.p[5] *= self.downscale_factor
        downscaled_info.p[6] *= self.downscale_factor

        self.image_info_pub.publish(downscaled_info)

    def _force_sim_time(self):
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rosgraph_msgs.msg import Clock


class LocalClockPublisher(Node):
    """Publishes a deterministic /clock sourced from the local monotonic clock."""

    def __init__(self):
        super().__init__("local_clock_publisher")

        self.declare_parameter("clock_frequency", 100.0)
        self.declare_parameter("initial_time", 0.0)

        self._declare_use_sim_time()

        frequency = max(self.get_parameter("clock_frequency").value, 1e-3)
        self._period = 1.0 / frequency
        self._initial_time = self.get_parameter("initial_time").value
        self._start_monotonic_ns = time.monotonic_ns()

        self._publisher = self.create_publisher(Clock, "/clock", 10)
        self._timer = self.create_timer(self._period, self._publish_clock)

    def _declare_use_sim_time(self):
        try:
            self.declare_parameter("use_sim_time", False)
        except ParameterAlreadyDeclaredException:
            pass
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])

    def _publish_clock(self):
        elapsed = (time.monotonic_ns() - self._start_monotonic_ns) * 1e-9 + self._initial_time
        secs = math.floor(elapsed)
        nanosecs = int((elapsed - secs) * 1e9)

        msg = Clock()
        msg.clock.sec = int(secs)
        msg.clock.nanosec = nanosecs
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalClockPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

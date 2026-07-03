#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class NavigationController(Node):
    def __init__(self, name):
        super().__init__('navigation_controller')
        """
        Initialize the NavigationController with the robot's name.
        
        :param name: The name of the robot (string)
        """
        self.name = name
        self.cmd_vel_topic = f"/{self.name}/cmd_vel"
        self.publisher = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.get_logger().info(f"Controller initialized for robot '{self.name}' on topic '{self.cmd_vel_topic}'.")

        self.twist = Twist()
        self.twist.linear.x = 0.0
        self.twist.linear.y = 0.0
        self.twist.linear.z = 0.0
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        self.twist.angular.z = 0.0

    def _publish_velocity(self, linear_x=1.0, angular_z=0.0):
        """
        Publish a Twist message with specified linear and angular velocities for a certain duration.
        
        :param linear_x: Linear velocity in the x-direction (float)
        :param angular_z: Angular velocity around the z-axis (float)
        :param duration: Time in seconds to publish the velocity (float)
        """
        
        self.twist.linear.x = linear_x
        self.twist.angular.z = angular_z
        
        self.publisher.publish(self.twist)
        # rospy.loginfo("Movement stopped.")

    def turn_right(self, angular_z=0.5):
        """
        Turn the robot to the right.
        
        :param angular_z: Angular velocity around the z-axis (default: 0.5)
        :param duration: Time in seconds to perform the turn (default: 1.0)
        """

        self.get_logger().info("turning right")
        self._publish_velocity(linear_x= self.twist.linear.x, angular_z=angular_z)

    def turn_left(self, angular_z=-0.5):
        """
        Turn the robot to the left.
        
        :param angular_z: Angular velocity around the z-axis (default: -0.5)
        :param duration: Time in seconds to perform the turn (default: 1.0)
        """
        self.get_logger().info("turning left")
        self._publish_velocity(linear_x=self.twist.linear.x, angular_z=angular_z)

    def go_straight(self, linear_x=0.5, angular_z = 0.0):
        self.get_logger().info("going straight")

        self._publish_velocity(linear_x=linear_x, angular_z= angular_z)

    def reverse(self, linear_x=-0.5, angular_z = 0.0):
        """
        Move the robot in reverse.
        
        :param linear_x: Linear velocity in the x-direction (negative for reverse, default: -0.5)
        :param duration: Time in seconds to move in reverse (default: 2.0)
        """
        self.get_logger().info("reversing")
        self._publish_velocity(linear_x=linear_x , angular_z = angular_z)

    def stop(self):
        self.get_logger().info("stopping")
        self._publish_velocity(linear_x=0.0,angular_z=0.0)
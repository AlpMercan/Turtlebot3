import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import cv2
import tf

# Load map and initialize parameters
map_image = cv2.imread("house.pgm", 0)
HEIGHT, WIDTH = map_image.shape
NUM_PARTICLES = 3000

# Initialize particle filter
particles = np.random.rand(NUM_PARTICLES, 3) * np.array(
    (WIDTH, HEIGHT, np.radians(360))
)

# Robot's current position
rx, ry, rtheta = (WIDTH / 4, HEIGHT / 4, 0)
prev_x, prev_y = (0, 0)


def move_particles(fwd, turn):
    global particles
    particles[:, 0] += fwd * np.cos(particles[:, 2])
    particles[:, 1] += fwd * np.sin(particles[:, 2])
    particles[:, 2] += turn
    particles[:, 0] = np.clip(particles[:, 0], 0, WIDTH - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, HEIGHT - 1)


def sense(x, y, noisy=False):
    SIGMA_SENSOR = 5
    x = int(x)
    y = int(y)
    if noisy:
        return map_image[y, x] + np.random.normal(0.0, SIGMA_SENSOR, 1)
    return map_image[y, x]


def compute_weights(robot_sensor):
    global particles
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        particle_sensor = sense(particles[i, 0], particles[i, 1])
        errors[i] = abs(robot_sensor - particle_sensor)
    weights = np.max(errors) - errors
    weights = np.clip(weights, 0, None) ** 3
    return weights


def resample(weights):
    global particles
    probabilities = weights / np.sum(weights)
    new_index = np.random.choice(NUM_PARTICLES, size=NUM_PARTICLES, p=probabilities)
    particles = particles[new_index, :]


def add_noise():
    global particles
    SIGMA_PARTICLE_STEP = 2
    SIGMA_PARTICLE_TURN = np.pi / 24
    noise = np.random.normal(0, SIGMA_PARTICLE_STEP, (NUM_PARTICLES, 2))
    noise = np.hstack(
        (noise, np.random.normal(0, SIGMA_PARTICLE_TURN, (NUM_PARTICLES, 1)))
    )
    particles += noise


def display():
    global particles, rx, ry, rtheta
    lmap = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    cv2.circle(lmap, (int(rx), int(ry)), 5, (0, 255, 0), 10)
    if len(particles) > 0:
        for p in particles:
            cv2.circle(lmap, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
    cv2.imshow("map", lmap)
    cv2.waitKey(1)


def odometry_callback(msg):
    global prev_x, prev_y, rx, ry, rtheta
    # Get the position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    # Get the orientation as a quaternion
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w,
    )

    # Convert the quaternion to Euler angles
    euler = tf.transformations.euler_from_quaternion(quaternion)

    # Assume the yaw is the third angle
    yaw = euler[2]

    # Calculate forward movement and rotation
    fwd = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
    turn = yaw - rtheta

    # Update global variables
    rx, ry, rtheta = x, y, yaw
    prev_x, prev_y = x, y

    # Move particles based on calculated forward movement and rotation
    move_particles(fwd, turn)


def laser_scan_callback(msg):
    # Process laser scan data
    # In a real-world application, you'd process the scan data more thoroughly.
    # For simplicity, let's just use the front distance
    front_distance = msg.ranges[len(msg.ranges) // 2]

    # Update particle weights based on the front distance
    robot_sensor = sense(rx, ry, noisy=True)
    weights = compute_weights(robot_sensor)
    resample(weights)
    add_noise()


# ROS Node Initialization
rospy.init_node("particle_filter_localization")
laser_sub = rospy.Subscriber("/scan", LaserScan, laser_scan_callback)
odom_sub = rospy.Subscriber("/odom", Odometry, odometry_callback)

# Main loop
rate = rospy.Rate(10)  # 10 Hz
while not rospy.is_shutdown():
    display()
    rate.sleep()

cv2.destroyAllWindows()
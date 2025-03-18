#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [Autonomous Exploration of Unknown Environments],
  abstract: [
    This report describes an integrated system for autonomous exploration of unknown environments using simultaneous localization and mapping (SLAM) with an exploration-oriented planning strategy. Our approach combines a particle-filter-based localization algorithm with occupancy grid mapping and an RRT-based planner driven by an entropy metric. The system is validated within a physics-based simulation environment and demonstrates robust performance despite sensor noise and dynamic challenges. This report explains our methodology, implementation details, design trade-offs, and lessons learned.
  ],
  authors: (
    (
      name: "Gavin Hua",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "ghua@caltech.edu"
    ),
    (
      name: "Gio Huh",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "ghuh@caltech.edu"
    ),
    (
      name: "Taekyung Lee",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "tklee2@caltech.edu"
    ),
  ),
  index-terms: ("SLAM", "Robotics"),
)

= Introduction

The primary goal of this project is to enable a UAV (modeled by its state $x = (p_x, p_y, dot(p)_x, dot(p)_y)^top$) to autonomously explore unknown 2D environments while building an accurate map and localizing itself in real time. The map is composed of a rectangular boundary, polygonal obstacles, and beacons representing room/terrain features used in localization. Key challenges included managing sensor noise, ensuring robust localization in cluttered environments, and generating safe, exploration-driven paths. To meet these challenges, our system integrates several core components:

- *Localization*: A particle-filter approach that leverages room feature ("beacon") measurements.
- *Mapping:* An occupancy grid framework using log-odds updates and ray-tracing to incrementally build the environment's map.
- *Planning:* An RRT-based planner augmented with an entropy map to select informative goal points.
- *Simulation & Control:* A physics simulation node models robot dynamics and sensor noise, while separate control and teleoperation nodes provide both autonomous and manual command inputs.

This modular design facilitates testing of individual components and allows for straightforward integration of additional sensors or planning strategies.

= Notation
For multi-dimensional Gaussian distributions, we overload the notation $x~N(p, sigma^2)$ where $p, sigma$ are scalars to denote $x~cal(N)(vec(p, dots.v, p), "diag"(sigma^2, ..., sigma^2))$. Unless denoted otherwise, all positions are in the world frame.


= Approach

== Simultaneous Localization and Mapping

Our SLAM implementation is a variant of the particle filter. This enables us to incorporate non-Gaussian distributions that commonly arise in distributions of the robot's position. However, the room features' positions ($p^i_x, p^i_y$) are determined with Kalman filter updates (details in the next section), since sensor noise is assumed to be Gaussian.

Compared to applying an Kalman filter to the full state consisting of the robot's position and all known features' positions, our hybrid method does not assume the form of the distribution of $x$. Moreover, it does not require the full covariance matrix (nor its inverse, which is computationally expensive to acquire), nor does it require dynamically resizing the state vector as more beacons are discovered.

Compared to applying a particle filter to track beacon positions, performing Kalman updates allows us to leverage key assumptions (white noise on sensors) to optimize mapping with minimal computation.

== Autonomous Planning and Trajectory Tracking

We deploy an information-theory-informed planner. In order to achieve maximum map information gain, we compute the edges of the known map by taking the gradient of the entropy in the occupancy grid. We then attempt to explore regions with sharp gradients that are close to the robot and are adjacent to unoccupied grid cells, thereby maximizing information gain over time. Once a point in a region is explored, we deploy RRT to quickly compute a valid path, and track the corresponding trajectory (FIXME: how do we get the trajectory?) with a PD controller.

The entropy-gradient method effectively resolves the exploration problem as it rejects regions with uniformly high entropy (unknown and inaccessible to planner) and uniformly low entropy (known regions). A simple PD controller suffices to reject process noise in our simulation. Technical details are discussed in the following section.


= Implementation

== ROS Node Structure
// FIXME some ROS node graph
We utilize RVIZ for visualization.

== Physics Simulation
We assume the controller has first-order control of the plant. Every time step, given the control input $u$, the physics simulation performs an Euler integration of the form
$
  vec(p, dot(p)) = Delta t mat(0, I; 0, 0) vec(p, dot(p)) + vec(0, u + w)
$
where $w ~ cal(N)(0, sigma_c^2)$ represents the controller noise.

If a collision were to occur, the line from $p_t$ to $p_(t+1)$ is traced and the robot is placed on the edge of the obstacle.

Moreover, the physics simulation computes the sensor measurements. The LiDAR sensor is modeled by ray-casting, which returns the distance to the nearest obstacle in the direction of the ray if the obstacle is within $r_max$. The camera sensor is modeled by a Gaussian distribution with mean $p^i$ and covariance $sigma_b^2 I$, where $p^i$ is the position of the $i$th beacon. The camera sensor is able to detec.t beacons if the line of sight is not obstructed.

All geometry functionality is provided by the `Shapely` library

== Mapping
We maintain three data structures for mapping: the occupancy grid, the list of beacon positions, and the list of beacon position covariances.

We perform a log-odds ratio update for the occupancy grid. The occupancy grid is represented by a 2D array of cells, each with a log-odds value theoretically from $(-oo, +oo)$. In practice, however, clipping the values to $[-10, 10]$ improved numerical stability.  The occupancy grid is initialized to $0$ for all cells. The log-odds update is performed as follows:
- for every ray cast by the LiDAR sensor, we perform Bresenham's line algorithm to find the cells that the ray intersects.
- for every cell before the ray's endpoint, we add $L_"FREE"$ to the log-odds value.
- if the ray intersects an obstacle, we add $L_"OCCUPIED"$ to the log-odds value of the cell at the ray's endpoint. Otherwise, we add $L_"FREE"$ to the log-odds value of the cell at the ray's endpoint.

Given a log-odds ratio, the occupancy probability is computed as $p = 1 / (1 + e^(-l))$.

We perform a Kalman update for the beacons. If an observed beacon is not within $r_b$ of a known beacon, then its observed position is added to the list of beacons. Its corresponding covariance is set to the covariance of the point cloud since the camera's covariance is unknown. Otherwise, given known position $p_1$, known covariance $P_1$ and observed position $p_2$, point cloud covariance $P_2$, we perform the Kalman update:
$
  P = (P_1^(-1) + P_2^(-1))^(-1) \
  p = P (P_1^(-1) p_1 + P_2^(-1) p_2)
$
This optimally fuses the two measurements.

== Localization (Particle Filter)
The particle filter maintains $N = 1000$ particles to represent the belief of the robot's position. Each particle is represented by its position $p_p^i$, and they are initialized by sampling $cal(N)(0, sigma_p^2)$. 
At each time step, the particle filter performs the following steps:
- Integrate all particles forward in time using Euler's method.
- Add noise $w_p^i ~cal(N)(0, sigma_p^2)$ to each particle.
- For the set of observed beacons $p_b^o$ for $o in [1, m]$ and the corresponding closest known beacons $p_(b*)^o$ compute the score $s_i$ as
$
  s_i = sum_(o=1)^m 1/(||p_(b*)^o - p_b^o||)
$
If any observation does not have a corresponding beacon within $r_b$, then $s^i = -oo$.
- Resample the particles according to the $P_i = e^(s_i) / (sum_i e^(s_i))$ (the softmax distribution) with replacement.

== Entropy-Based Goal selection

The information entropy map is computed as $H_i = - P_i ln(P_i)$, where $P_i$ is the occupancy probability of cell $i$. To remove noise artifacts, let
$
  H_"av" = 1/9 mat(1, 1, 1; 1, 1, 1; 1, 1, 1) * H_i
$
We then apply a Sobel filter (commonly used for edge extraction in images) to the entropy map $H_"av"$, defined as follows
$
  G_x = mat(-1, 0, 1; -2, 0, 2; -1, 0, 1) * H_"av" \
  G_y = mat(-1, -2, -1; 0, 0, 0; 1, 2, 1) * H_"av"
$
The gradient is then computed as $G = sqrt(G_x^2 + G_y^2)$. An interpretation of this value is that it represents the boundaries between the known (low entropy) and unknown (high entropy) regions. To account for the robot's position and to reward staying close to beacons (to ensure effective localization), we compute the distance to the robot's position $d_c = ||p_c - p||$ and set the cell with the maximum value of $ G / d_c + alpha sum_(i=1)^(n) 1/(||p_b^i - p_c||) $ (where $n$ is the number of known beacons) as the goal. $alpha$ is a tunable parameter that weights the importance of good localization.
== RRT

== PD Controller
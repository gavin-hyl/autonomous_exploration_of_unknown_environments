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
- Controller
- PhysicsSim
- Slam
// - *Algorithm Details:*  
//   The localization module implements a particle filter where particles are initialized around an initial guess with added Gaussian noise. Each particle’s weight is updated based on its consistency with observed beacon positions and LiDAR readings. For numerical stability, the particle weights are normalized using a softmax function:
  
//   $ w_t^{[i]} = frac{exp(q^{[i]} - max_j q^{[j]})}{sum_{j=1}^N exp(q^{[j]} - max_k q^{[k]})}$ 

// - *Vectorized Operations:*  
//   Many operations are vectorized using NumPy to reduce the computational overhead associated with Python loops.

// - *Trade-Offs:*  
//   Increasing the number of particles improves accuracy but increases computation. Parameter tuning is essential to balance these factors.

// === Occupancy Grid Mapping

// - *Mapping Method:*  
//   The mapping module uses a log-odds framework to update an occupancy grid. LiDAR rays are traced using a Bresenham algorithm to mark free space and obstacles. Beacon measurements help refine the positions of landmarks.

// - *Log-Odds and Occupancy Probability:*  
//   The log-odds update for each grid cell is:
  
//   $ L_t = L_{t-1} + logleft(frac{p(z mid m)}{1 - p(z mid m)}right)$ 
  
//   The occupancy probability is then computed as:
  
//   $ p(m mid z) = frac{e^{L_t}}{1 + e^{L_t}}$ 

// - *Entropy Calculation:*  
//   To quantify uncertainty in a cell, entropy is calculated as:
  
//   $ H(p) = -p log_2(p) - (1-p) log_2(1-p)$ 

// === RRT-Based Path Planning

// - *RRT Planner:*  
//   The planner employs a Rapidly-Exploring Random Tree (RRT) algorithm enhanced with an entropy-based metric. An entropy map is generated from the occupancy grid to highlight unexplored or uncertain areas. The planner then selects goal points that maximize information gain while avoiding obstacles.

// - *Steering Function:*  
//   To steer from a start node ((x_s, y_s)) toward a target node ((x_t, y_t)) by a fixed step size (Delta), the new node is computed as:
  
//   $ (x_{text{new}}, y_{text{new}}) = Bigl(x_s + Delta cos(theta),; y_s + Delta sin(theta)Bigr)$ 
  
//   where
  
//   $ theta = arctanleft(frac{y_t - y_s}{x_t - x_s}right)$ 

// === PD Controller Integration

// - *Control Law:*  
//   A proportional–derivative (PD) controller computes the control input for path following:
  
//   $ u(t) = K_p , e(t) + K_d , frac{de(t)}{dt}$ 
  
//   where (e(t)) is the error between the robot’s current position and the target waypoint.

// === Simulation and Control

// - *Physics Simulation:*  
//   A dedicated physics simulation node models robot dynamics, collision detection, and sensor noise using a physics-based model. Geometric processing (e.g., using Shapely) aids in collision checking.

// - *Teleoperation and Autonomous Control:*  
//   Separate nodes allow for both constant motion commands (autonomous control) and manual overrides (teleoperation). This separation facilitates rapid prototyping and testing.

// == Implementation

// The system is implemented as a collection of modular ROS2 nodes:

// - *Localization Module:*  
//   Implements a particle filter (see Localization.py) with motion updates, measurement updates, and resampling. Covariance estimation is performed as:
  
//   $ Sigma = frac{1}{N-1} sum_{i=1}^{N} left(x_t^{[i]} - bar{x}_tright) left(x_t^{[i]} - bar{x}_tright)^T$ 

// - *Mapping Module:*  
//   Maintains an occupancy grid updated using sensor data. The log-odds framework and Bresenham ray tracing are central to its operation.

// - *SLAM Node:*  
//   Integrates localization and mapping, processes sensor topics, and publishes visualization messages (e.g., for particles and estimated poses).

// - *Planner:*  
//   Combines RRT planning with entropy mapping for goal selection. A PD controller is used to follow the planned path, and the planner incorporates mechanisms to avoid revisiting already explored areas.

// - *Simulation and Teleoperation:*  
//   A physics simulation node models real-world dynamics, while a teleoperation node provides keyboard-based control for manual testing.

// - *Data Plotting:*  
//   Post-processing modules (e.g., plotdata.py) generate plots comparing true versus estimated positions, error trends, and particle distributions, facilitating performance evaluation.

// == Contributions and Lessons Learned

// === Contributions

// - *Algorithmic Enhancements:*  
//   - A vectorized particle filter for efficient and robust localization.
//   - Integration of beacon data with LiDAR readings for improved mapping accuracy.
//   - An entropy-driven RRT planner that intelligently selects exploration targets.

// - *System Integration:*  
//   - A modular design that separates localization, mapping, planning, and simulation, simplifying debugging and future enhancements.
//   - Real-time visualization tools that provide immediate feedback on system performance.

// - *Robustness and Adaptability:*  
//   - Dynamic replanning based on updated occupancy grids and exploration boundaries.
//   - Adaptive parameter selection strategies balancing computation and accuracy.

// === Lessons Learned

// - *Parameter Sensitivity:*  
//   The performance of the particle filter and RRT planner is highly sensitive to parameters such as noise levels, particle count, and grid resolution. Iterative tuning via simulation was essential.

// - *Trade-Offs:*  
//   Increasing computational complexity (e.g., more particles or finer grids) improves accuracy but impacts real-time performance. Balancing these trade-offs is crucial.

// - *Modularity Benefits:*  
//   A modular design allowed for isolated testing of individual components, facilitating easier debugging and upgrades.

// - *Integration Challenges:*  
//   Coordinating updates between localization, mapping, and planning required careful management of ROS topics, timing, and data consistency.

// == Discussion and Conclusions

// === Pros and Cons

// - *Pros:*  
//   - *Robust Localization:* The integration of LiDAR and beacon data enhances accuracy.
//   - *Informed Exploration:* Entropy maps combined with RRT planning lead the robot to high-information areas.
//   - *Modularity:* The system’s modular architecture simplifies development and future expansions.
//   - *Real-Time Operation:* Optimized vectorized operations and tuned parameters support real-time performance.

// - *Cons:*  
//   - *Computational Overhead:* High-resolution grids and numerous particles increase processing demands.
//   - *Parameter Dependency:* The system’s performance is sensitive to several tuning parameters, suggesting that automated calibration could be beneficial.
//   - *Integration Complexity:* Coordinating multiple nodes (localization, mapping, planning, simulation) presents integration challenges.

// === Final Thoughts

// Successful autonomous exploration requires more than just making an algorithm “work.” It necessitates a deep understanding of sensor noise, mapping fidelity, planning strategies, and control dynamics. Our system demonstrates a robust integration of these elements, providing a foundation for future research in sensor fusion and intelligent path planning in dynamic, uncertain environments.

// For practitioners, we recommend:
// - *Investing in modular design* to allow for isolated testing and upgrades.
// - *Focusing on parameter calibration* using simulation environments.
// - *Utilizing visualization tools* for real-time feedback and debugging.

// == Additional Equations and Formulas

// Below are the mathematical foundations underlying our system.

// === Particle Filter Localization

// - *Motion Update:*
  
//   $ x_t^{[i]} sim p(x_t mid x_{t-1}^{[i]}, u_t)$ 

// - *Measurement Update:*
  
//   $ w_t^{[i]} = p(z_t mid x_t^{[i]})$   
//   Normalized with softmax:
  
//   $ w_t^{[i]} = frac{exp(q^{[i]} - max_j q^{[j]})}{sum_{j=1}^N exp(q^{[j]} - max_k q^{[k]})} $

// - *Covariance Estimation:*
//   +
//   $ Sigma = frac{1}{N-1} sum_{i=1}^{N} 

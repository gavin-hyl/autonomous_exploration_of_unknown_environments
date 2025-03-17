#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
import numpy as np
import math
import sys
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import random
import threading
import time
from collections import deque
import os
import pickle

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for reinforcement learning-based exploration.
    
    This implementation uses a simple neural network to learn Q-values for
    state-action pairs, enabling the robot to learn optimal exploration strategies.
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        """Initialize the DQN agent."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam
            self.tf = tf
            
            # Save parameters
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.memory = deque(maxlen=10000)
            self.learning_rate = learning_rate
            self.gamma = gamma  # Discount factor
            self.epsilon = epsilon  # Exploration rate
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
            self.tf_imported = True
            
        except ImportError:
            print("TensorFlow not available. Using random action selection instead.")
            self.tf_imported = False
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.epsilon = epsilon
    
    def _build_model(self):
        """Build a neural network model for DQN."""
        if not self.tf_imported:
            return None
            
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target network with the main network's weights."""
        if not self.tf_imported:
            return
            
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory for replay."""
        if not self.tf_imported:
            return
            
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, evaluate=False):
        """Choose an action based on the current state."""
        if not self.tf_imported:
            return random.randrange(self.action_dim)
            
        if (not evaluate) and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
            
        state = np.reshape(state, [1, self.state_dim])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Train the model using experience replay."""
        if not self.tf_imported or len(self.memory) < batch_size:
            return 0
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_dim))
        next_states = np.zeros((batch_size, self.state_dim))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
        
        targets = self.model.predict(states, verbose=0)
        next_state_values = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.amax(next_state_values[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]
    
    def load(self, name):
        """Load model weights from file."""
        if not self.tf_imported:
            return
            
        try:
            self.model.load_weights(name)
            self.target_model.load_weights(name)
        except:
            print(f"Failed to load model weights from {name}")
    
    def save(self, name):
        """Save model weights to file."""
        if not self.tf_imported:
            return
            
        self.model.save_weights(name)


class ReplayBuffer:
    """Buffer to store and sample experiences for replay."""
    
    def __init__(self, capacity=10000):
        """Initialize replay buffer with given capacity."""
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of transitions from the buffer."""
        if len(self.buffer) < batch_size:
            return []
            
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class RLExplorationController(Node):
    """
    Reinforcement Learning-based exploration controller for autonomous robot navigation.
    
    This controller uses a DQN agent to learn and execute exploration strategies
    that maximize information gain and beacon discovery.
    """
    
    def __init__(self):
        """Initialize the RL exploration controller node."""
        super().__init__('rl_exploration_controller')
        
        # Parameters
        self.declare_parameter('state_size', 32)  # Size of state representation
        self.declare_parameter('action_size', 8)  # Number of discrete actions
        self.declare_parameter('learning_rate', 0.001)
        self.declare_parameter('gamma', 0.99)  # Discount factor
        self.declare_parameter('epsilon_initial', 1.0)  # Initial exploration rate
        self.declare_parameter('epsilon_min', 0.1)  # Minimum exploration rate
        self.declare_parameter('epsilon_decay', 0.995)  # Exploration decay rate
        self.declare_parameter('batch_size', 32)  # Batch size for training
        self.declare_parameter('update_frequency', 10)  # How often to update target network
        self.declare_parameter('train', False)  # Whether to train the agent
        self.declare_parameter('model_path', 'rl_explorer_model')  # Path for saving/loading model
        self.declare_parameter('occupancy_threshold', 50)  # Threshold for considering a cell as occupied
        self.declare_parameter('move_step', 0.5)  # Step size for each move
        self.declare_parameter('reward_complete_map', 100.0)  # Reward for completing the map
        self.declare_parameter('reward_discover_beacon', 50.0)  # Reward for discovering a new beacon
        self.declare_parameter('reward_collision', -20.0)  # Penalty for collision
        self.declare_parameter('reward_explore_unknown', 1.0)  # Reward for exploring unknown cells
        
        # Get parameters
        self.state_size = self.get_parameter('state_size').value
        self.action_size = self.get_parameter('action_size').value
        self.learning_rate = self.get_parameter('learning_rate').value
        self.gamma = self.get_parameter('gamma').value
        self.epsilon_initial = self.get_parameter('epsilon_initial').value
        self.epsilon_min = self.get_parameter('epsilon_min').value
        self.epsilon_decay = self.get_parameter('epsilon_decay').value
        self.batch_size = self.get_parameter('batch_size').value
        self.update_frequency = self.get_parameter('update_frequency').value
        self.train_mode = self.get_parameter('train').value
        self.model_path = self.get_parameter('model_path').value
        self.occupancy_threshold = self.get_parameter('occupancy_threshold').value
        self.move_step = self.get_parameter('move_step').value
        self.reward_complete_map = self.get_parameter('reward_complete_map').value
        self.reward_discover_beacon = self.get_parameter('reward_discover_beacon').value
        self.reward_collision = self.get_parameter('reward_collision').value
        self.reward_explore_unknown = self.get_parameter('reward_explore_unknown').value
        
        # Initialize RL agent and replay buffer
        self.agent = DQNAgent(
            state_dim=self.state_size, 
            action_dim=self.action_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon=self.epsilon_initial,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay
        )
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Try to load existing model
        if not self.train_mode and os.path.exists(f"{self.model_path}.h5"):
            self.agent.load(f"{self.model_path}.h5")
            self.get_logger().info(f"Loaded model from {self.model_path}.h5")
        
        # Robot state
        self.position = np.array([0.0, 0.0, 0.0])  # Current position [x, y, z]
        self.occupancy_grid = None  # Occupancy grid map
        self.grid_info = None  # Map metadata
        self.known_beacons = []  # List of known beacon positions
        self.lidar_data = []  # Current LiDAR readings
        self.moving = False  # Flag to indicate if robot is currently moving
        
        # Exploration state
        self.exploration_complete = False
        self.coverage_threshold = 0.95  # Consider exploration complete when 95% of the map is explored
        self.exploration_steps = 0
        self.max_exploration_steps = 1000  # Safeguard to prevent infinite exploration
        self.previous_known_beacons_count = 0
        self.previous_map_coverage = 0.0
        
        # RL training state
        self.current_state = None
        self.current_action = None
        self.episode_rewards = []
        self.episode_reward = 0
        self.total_train_steps = 0
        
        # Subscribers
        self.create_subscription(
            Vector3, '/pos_hat', self.position_callback, 10
        )
        self.create_subscription(
            OccupancyGrid, '/occupancy_grid', self.map_callback, 10
        )
        self.create_subscription(
            MarkerArray, '/estimated_beacons', self.beacons_callback, 10
        )
        self.create_subscription(
            PointCloud2, '/lidar', self.lidar_callback, 10
        )
        self.create_subscription(
            Bool, '/slam_done', self.slam_done_callback, 10
        )
        
        # Publishers
        self.control_pub = self.create_publisher(Vector3, '/control_signal', 10)
        
        # Timer
        self.create_timer(1.0, self.rl_loop)  # Run RL loop every second
        
        # Save training results periodically
        if self.train_mode:
            self.create_timer(60.0, self.save_model)  # Save model every minute
        
        self.get_logger().info('RL Exploration Controller initialized')
    
    def position_callback(self, msg):
        """Update the robot's current position."""
        self.position = np.array([msg.x, msg.y, msg.z])
    
    def map_callback(self, msg):
        """Update the occupancy grid map."""
        self.occupancy_grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.grid_info = msg.info
    
    def beacons_callback(self, msg):
        """Update the list of known beacons."""
        self.known_beacons = []
        for marker in msg.markers:
            beacon_pos = np.array([
                marker.pose.position.x,
                marker.pose.position.y,
                marker.pose.position.z
            ])
            self.known_beacons.append(beacon_pos)
    
    def lidar_callback(self, msg):
        """Update the current LiDAR readings."""
        points = list(read_points(msg, field_names=("x", "y", "z")))
        self.lidar_data = [np.array([p[0], p[1], p[2]]) for p in points]
    
    def slam_done_callback(self, msg):
        """Receive signal that SLAM processing is done."""
        if msg.data and self.moving:
            self.moving = False
            self.get_logger().info(f'Reached position: [{self.position[0]:.2f}, {self.position[1]:.2f}]')
            
            # If we're training, process the step outcome
            if self.train_mode and self.current_state is not None and self.current_action is not None:
                next_state = self.get_state()
                reward = self.calculate_reward()
                done = self.is_exploration_complete()
                
                self.episode_reward += reward
                
                # Store experience in replay buffer
                self.replay_buffer.store(self.current_state, self.current_action, reward, next_state, done)
                
                # Train the agent
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.agent.replay(self.batch_size)
                    self.get_logger().debug(f'Training loss: {loss}')
                
                # Update target network periodically
                self.total_train_steps += 1
                if self.total_train_steps % self.update_frequency == 0:
                    self.agent.update_target_model()
                    self.get_logger().debug('Updated target model')
                
                # Reset for next step
                self.current_state = next_state
                
                if done:
                    self.episode_rewards.append(self.episode_reward)
                    self.get_logger().info(f'Episode completed with reward: {self.episode_reward}')
                    self.episode_reward = 0
                    self.exploration_complete = True
    
    def rl_loop(self):
        """Main loop for RL-based exploration."""
        if self.exploration_complete:
            self.get_logger().info('Exploration complete.')
            return
        
        if self.moving:
            self.get_logger().debug('Still moving, waiting for completion...')
            return
        
        if self.occupancy_grid is None:
            self.get_logger().info('Waiting for map data...')
            return
        
        self.exploration_steps += 1
        if self.exploration_steps >= self.max_exploration_steps:
            self.exploration_complete = True
            self.get_logger().info(f'Reached maximum exploration steps ({self.max_exploration_steps}).')
            return
        
        # Check if exploration is complete based on map coverage
        if self.is_exploration_complete():
            self.exploration_complete = True
            self.get_logger().info('Exploration complete: Map sufficiently explored.')
            return
        
        # Get current state
        current_state = self.get_state()
        
        # Choose action based on current state
        if self.train_mode:
            action = self.agent.act(current_state)
            self.current_state = current_state
            self.current_action = action
        else:
            action = self.agent.act(current_state, evaluate=True)
        
        # Execute the selected action
        self.execute_action(action)
    
    def get_state(self):
        """Create a state representation for the RL agent."""
        if self.occupancy_grid is None or self.grid_info is None:
            # Return zeros if we don't have map data yet
            return np.zeros(self.state_size)
        
        # Convert position to grid coordinates
        grid_x, grid_y = self.world_to_grid(self.position[0], self.position[1])
        
        # Summarize occupancy grid around the robot
        local_grid = self.get_local_grid(grid_x, grid_y, radius=5)
        
        # Calculate distance and direction to nearest unexplored area
        distance_to_unexplored, direction_to_unexplored = self.get_unexplored_info(grid_x, grid_y)
        
        # Calculate distances and directions to known beacons
        beacon_features = self.get_beacon_features()
        
        # Combine features into state vector
        state = np.concatenate([
            local_grid.flatten() / 100.0,  # Normalize to [0, 1]
            [self.position[0] / 10.0, self.position[1] / 10.0],  # Normalize position
            [distance_to_unexplored / 10.0],  # Normalize distance
            direction_to_unexplored,  # Direction vector
            beacon_features
        ])
        
        # Ensure state has the correct dimension
        if len(state) > self.state_size:
            state = state[:self.state_size]
        elif len(state) < self.state_size:
            state = np.pad(state, (0, self.state_size - len(state)))
        
        return state
    
    def get_local_grid(self, grid_x, grid_y, radius=5):
        """Get a local view of the occupancy grid around the robot."""
        if self.occupancy_grid is None:
            return np.zeros((2*radius+1, 2*radius+1))
        
        # Create local grid
        local_grid = np.ones((2*radius+1, 2*radius+1)) * -1  # Unknown cells
        
        # Fill local grid with actual occupancy values
        for y in range(max(0, grid_y - radius), min(self.grid_info.height, grid_y + radius + 1)):
            for x in range(max(0, grid_x - radius), min(self.grid_info.width, grid_x + radius + 1)):
                local_y = y - (grid_y - radius)
                local_x = x - (grid_x - radius)
                if 0 <= local_y < 2*radius+1 and 0 <= local_x < 2*radius+1:
                    local_grid[local_y, local_x] = self.occupancy_grid[y, x]
        
        return local_grid
    
    def get_unexplored_info(self, grid_x, grid_y):
        """Get distance and direction to nearest unexplored area."""
        if self.occupancy_grid is None:
            return 0.0, np.zeros(2)
        
        # Search for unexplored cells
        min_distance = float('inf')
        direction = np.zeros(2)
        
        # Search radius (limit for efficiency)
        search_radius = 20
        
        for y in range(max(0, grid_y - search_radius), min(self.grid_info.height, grid_y + search_radius + 1)):
            for x in range(max(0, grid_x - search_radius), min(self.grid_info.width, grid_x + search_radius + 1)):
                if self.occupancy_grid[y, x] == -1:  # Unexplored cell
                    # Calculate distance
                    dx = x - grid_x
                    dy = y - grid_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance < min_distance:
                        min_distance = distance
                        # Calculate direction (normalize)
                        if distance > 0:
                            direction = np.array([dx / distance, dy / distance])
        
        if min_distance == float('inf'):
            return 0.0, np.zeros(2)
        
        return min_distance, direction
    
    def get_beacon_features(self):
        """Get features related to known beacons."""
        if not self.known_beacons:
            return np.zeros(4)  # No beacons known yet
        
        # Find closest beacon
        closest_distance = float('inf')
        closest_direction = np.zeros(2)
        
        for beacon in self.known_beacons:
            distance = np.linalg.norm(beacon[:2] - self.position[:2])
            if distance < closest_distance:
                closest_distance = distance
                # Calculate direction (normalize)
                if distance > 0:
                    direction = (beacon[:2] - self.position[:2]) / distance
                    closest_direction = direction
        
        return np.array([
            closest_distance / 10.0,  # Normalize distance
            closest_direction[0],
            closest_direction[1],
            len(self.known_beacons) / 10.0  # Normalize number of beacons
        ])
    
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates."""
        if self.grid_info is None:
            return 0, 0
            
        grid_x = int((world_x - self.grid_info.origin.position.x) / self.grid_info.resolution)
        grid_y = int((world_y - self.grid_info.origin.position.y) / self.grid_info.resolution)
        return grid_x, grid_y
    
    def is_exploration_complete(self):
        """Check if exploration is complete based on map coverage."""
        if self.occupancy_grid is None:
            return False
        
        # Count cells that are not unknown (-1 in ROS occupancy grid)
        known_cells = np.sum(self.occupancy_grid >= 0)
        total_cells = self.occupancy_grid.size
        coverage = known_cells / total_cells
        
        self.get_logger().info(f'Map coverage: {coverage:.2f}')
        return coverage >= self.coverage_threshold
    
    def calculate_reward(self):
        """Calculate reward for the current step."""
        if self.occupancy_grid is None:
            return 0.0
        
        reward = 0.0
        
        # Reward for map completion
        if self.is_exploration_complete():
            reward += self.reward_complete_map
        
        # Reward for discovering new beacons
        current_beacon_count = len(self.known_beacons)
        if current_beacon_count > self.previous_known_beacons_count:
            reward += self.reward_discover_beacon * (current_beacon_count - self.previous_known_beacons_count)
            self.previous_known_beacons_count = current_beacon_count
        
        # Reward for exploring new areas
        known_cells = np.sum(self.occupancy_grid >= 0)
        total_cells = self.occupancy_grid.size
        current_coverage = known_cells / total_cells
        if current_coverage > self.previous_map_coverage:
            # Reward proportional to the increase in coverage
            reward += self.reward_explore_unknown * (current_coverage - self.previous_map_coverage) * 100
            self.previous_map_coverage = current_coverage
        
        # Penalty for collisions (if robot is near an obstacle)
        grid_x, grid_y = self.world_to_grid(self.position[0], self.position[1])
        if self.is_near_obstacle(grid_x, grid_y):
            reward += self.reward_collision
        
        return reward
    
    def is_near_obstacle(self, grid_x, grid_y, distance=2):
        """Check if the robot is near an obstacle."""
        if self.occupancy_grid is None:
            return False
            
        for y in range(max(0, grid_y - distance), min(self.grid_info.height, grid_y + distance + 1)):
            for x in range(max(0, grid_x - distance), min(self.grid_info.width, grid_x + distance + 1)):
                if self.occupancy_grid[y, x] >= self.occupancy_threshold:
                    return True
        
        return False
    
    def execute_action(self, action):
        """Execute the selected action."""
        # Convert discrete action to continuous control
        # Actions are evenly spaced directions (8 directions + stay in place)
        if action < self.action_size - 1:  # Movement action
            angle = 2 * math.pi * action / (self.action_size - 1)
            dx = self.move_step * math.cos(angle)
            dy = self.move_step * math.sin(angle)
            move = np.array([dx, dy, 0.0])
        else:  # Stay in place
            move = np.array([0.0, 0.0, 0.0])
        
        # Execute the move
        self.moving = True
        
        # Create and publish control message
        msg = Vector3()
        msg.x = float(move[0])
        msg.y = float(move[1])
        msg.z = 0.0
        self.control_pub.publish(msg)
        
        self.get_logger().info(f'Published control: [{msg.x:.2f}, {msg.y:.2f}], Action: {action}')
    
    def save_model(self):
        """Save the DQN model and training statistics."""
        if not self.train_mode:
            return
            
        # Save model weights
        model_path = f"{self.model_path}.h5"
        self.agent.save(model_path)
        
        # Save training statistics
        stats = {
            'episode_rewards': self.episode_rewards,
            'total_steps': self.total_train_steps,
            'epsilon': self.agent.epsilon
        }
        
        try:
            with open(f"{self.model_path}_stats.pkl", 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            self.get_logger().error(f"Failed to save training statistics: {e}")
        
        self.get_logger().info(f"Saved model and statistics to {self.model_path}")


def main(args=None):
    """Main function for the RL exploration controller."""
    rclpy.init(args=args)
    
    # Create controller
    controller = RLExplorationController()
    
    try:
        # Spin the node
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # Save model on shutdown
        if controller.train_mode:
            controller.save_model()
        
        # Clean up
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

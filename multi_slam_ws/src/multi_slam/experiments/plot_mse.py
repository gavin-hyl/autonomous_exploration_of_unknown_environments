#!/usr/bin/env python3
# plot_mse.py

import matplotlib.pyplot as plt
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from visualization_msgs.msg import Marker
import numpy as np

def read_mse_from_bag(bag_path, use_x=True):
    """Read MSE data from bag file, returns x or y scale values."""
    
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    mse_values = []
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    while reader.has_next():
        topic_name, data, t = reader.read_next()
        if topic_name == '/particle_vs_kalman':
            msg = deserialize_message(data, Marker)
            mse_values.append(msg.scale.x if use_x else msg.scale.y)
    
    return mse_values

def plot_merged_mse(particle_bag_path, kalman_bag_path):
    """Plot MSE data from two bags."""
    
    particle_mse = np.array(read_mse_from_bag(particle_bag_path, use_x=True))
    kalman_mse = np.array(read_mse_from_bag(kalman_bag_path, use_x=False))

    print(f"Average MSE for Particle Filter: {np.mean(particle_mse)}")
    print(f"Average MSE for Kalman Filter: {np.mean(kalman_mse)}")
    print(f"Average Std for Particle Filter: {np.std(particle_mse)}")
    print(f"Average Std for Kalman Filter: {np.std(kalman_mse)}")

    print(f'After Stabilization 20 time steps')    
    print(f"Average MSE for Particle Filter: {np.mean(particle_mse[20:])}")
    print(f"Average MSE for Kalman Filter: {np.mean(kalman_mse[20:])}")
    print(f"Average Std for Particle Filter: {np.std(particle_mse[20:])}")
    print(f"Average Std for Kalman Filter: {np.std(kalman_mse[20:])}")

    print(f'Before Stabilization 20 time steps')    
    print(f"Average MSE for Particle Filter: {np.mean(particle_mse[:20])}")
    print(f"Average MSE for Kalman Filter: {np.mean(kalman_mse[:20])}")
    print(f"Average Std for Particle Filter: {np.std(particle_mse[:20])}")
    print(f"Average Std for Kalman Filter: {np.std(kalman_mse[:20])}")
    
    plt.figure(figsize=(10, 6))
    minimum_length = min(len(particle_mse), len(kalman_mse)) - 1
    time_steps = np.arange(minimum_length)
    plt.plot(time_steps, particle_mse[time_steps], label='Particle Filter')
    plt.plot(time_steps, kalman_mse[time_steps], label='Kalman Filter')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.title('Particle vs Kalman Beacon MSE Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('mse_comparison_particle_vs_kalman.png')
    plt.show()

if __name__ == '__main__':
    particle_bag = 'data/particle'  # Update this
    kalman_bag = 'data/kalman'      # Update this
    plot_merged_mse(particle_bag, kalman_bag)

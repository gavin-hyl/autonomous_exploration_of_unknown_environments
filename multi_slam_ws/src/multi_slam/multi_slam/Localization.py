class Localization:
    def __init__(self, initial_location, std_dev, num_particles, dt):
        self.current_location = initial_location
        self.std_dev = std_dev
        self.num_particles = num_particles
        self.dt = dt
                
        self.occupancy_threshold = 0.5


    def update_position(self, control_signal, beacon_data, estimated_map):
        # control signal: single array (delta_x, delta_y, delta_z)
        # beacon data: list of arrays (delta_x, delta_y, delta_z)
        # estimated map: object to get occupancy grid and beacon data

        predicted_location = self.current_location + control_signal * self.dt

        # generate particles with gaussian noise
        particles = []
        for i in range(self.num_particles):
            # z is fixed to 0
            particle_location = predicted_location + [np.random.normal(0, self.std_dev), np.random.normal(0, self.std_dev), 0.0]
            particles.append(particle_location)
        
        # calculate score for each particle
        scores = []
        for particle in particles:
            scores.append(self.calculate_score(particle, beacon_data, estimated_map))

        # normalize scores (sum to 1)
        scores = np.array(scores)
        scores = scores / np.sum(scores)

        # resample particles
        particles = np.random.choice(particles, size=self.num_particles, p=scores)

        # update current location
        self.current_location = np.mean(particles, axis=0)
        self.std_dev = np.std(particles, axis=0) 

        return self.current_location, self.std_dev

    def calculate_score(self, particle, beacon_data, estimated_map):
        score = 0 

        invalid_particle = False
        # for each beacon measurement
        for beacon_measurement in beacon_data:
            
            global_beacon = self.current_location + beacon_measurement
            # check if the line between particle and beacon is occupied
            for (x, y) in self.create_2d_line(particle[0:2], global_beacon[0:2]):
                if estimated_map.get_occupancy(x, y) > self.occupancy_threshold:
                    invalid_particle = True
            
            (closest_beacon, _) = estimated_map.get_closest_beacon(estimated_map, global_beacon)
            if closest_beacon is not None:
                invalid_particle = True
            
            if invalid_particle:
                score = -float('inf')
                break
            else:
                score += 1 / np.sqrt(sum((closest_beacon - global_beacon) ** 2))
        return score
    
    def create_2d_line(start, end):
        x1, y1 = start
        x2, y2 = end
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        step_x = 1 if x1 < x2 else -1
        step_y = 1 if y1 < y2 else -1
        
        error = dx - dy
        x, y = x1, y1

        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
            
            error2 = error * 2
            
            if error2 > -dy:
                error -= dy
                x += step_x
                
            if error2 < dx:
                error += dx
                y += step_y
        
        return points

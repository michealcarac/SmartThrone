# This file contains all of the environments for training the Agent
# The base encironment sets up all the functions and parameters and are overwritten in the subclasses
import numpy as np
import matplotlib.pyplot as plt

#Testing commits
class Base():

    def __init__(self, height=1024, width=1024, num_obstables=0, box_w=100, box_h=100):
        # Constructor for the base environment
        # Requires:
        #   Height: One dimension of the 2D environment, default value is 1024
        #   Width: The other dimension of the 2D environment, default value is 1024
        #   Num_obstacles: The number of random obstacles to generate in the environment, default value is 0
        #   Box_w: Max width of the random boxes, default value is 100
        #   Box_h: Max height of the random boxes, default value is 100
        # Returns:
        #   Created object

        self.height = height
        self.width = width
        self.obstacles = self.generate_obstacles(num_obstables, box_w, box_h)

    def generate_obstacles(self, num, box_w, box_h):
        # Generates a list containing the walls and random obstacle boxes to avoid
        # Requires:
        #   Num: The number of obstacles to make in the environment
        #   Box_w: Max width of the random boxes
        #   Box_h: Max height of the random boxes
        # Returns:
        #   Obstacles: A list of lines and rectangles in the environment

        # Generate the walls first
        obstacles = []
        left_wall = ((0, 0), (0, self.height))
        bottom_wall = ((0, self.height), (self.width, self.height))
        right_wall = ((self.width, 0), (self.width, self.height))
        top_wall = ((self.width, 0), (0, 0))


        # Generate num random boxes in the environment
        # Overlap is not checked for to allow for more complicated structures
        for i in range(num):
            rand_x = np.random.randint(0, self.width)
            rand_y = np.random.randint(0, self.height)
            rand_w = np.random.randint(0, box_w)
            rand_h = np.random.randint(0, box_h)
            top_left = (rand_x - rand_w // 2, rand_y - rand_h // 2)
            bottom_left = (rand_x - rand_w // 2, rand_y + rand_h // 2)
            top_right = (rand_x + rand_w // 2, rand_y - rand_h // 2)
            bottom_right = (rand_x + rand_w // 2, rand_y + rand_h // 2)
            box = (
                (top_left, top_right), (bottom_right, top_right), (bottom_left, bottom_right), (bottom_left, top_left))
            for wall in box:
                obstacles.append(wall)
        obstacles.append(left_wall)
        obstacles.append(bottom_wall)
        obstacles.append(right_wall)
        obstacles.append(top_wall)
        return obstacles

    def get_observation(self, x, y, theta, los):
        # A function that simulates looking around the environment from the point (x, y, theta) in the directions of los
        # Requires:
        #   x: The x position of the observation
        #   y: The y position of the observation
        #   theta: The starting direction of the observation
        #   los: A list containing the angles +- of theta to look in
        # Returns:
        #   observation: a list of distances to objects in the directions of los, same dimension as los
        observations = []
        for radian in los:
            best_dist = np.math.inf
            # self.obstacles = [self.obstacles[1]]
            for obs in self.obstacles:
                if radian == 0:
                    m1 = 0
                else:
                    m1 = np.math.tan(radian)
                (x3, y3), (x4, y4) = obs
                if x4 == x3:
                    m2 = 1000000
                else:
                    m2 = 0
                a = y - m1 * x
                b = y3 - m2 * x3
                A = [[1, -m2],[1, -m1]]
                B = [b,a]
                X = np.linalg.inv(A).dot(B)
                y1, x1 = X
                dist = np.sqrt((X[1] - x) ** 2 + (X[0] - y) ** 2)
                # x1, y1 = dist * np.math.cos(theta) + x, dist * np.sin(theta) + y
                theta1 = np.math.atan2(y1-y,x1-x)
                # print(X)
                # print(radian, theta1)
                if dist < best_dist and np.abs(radian - theta1) <= 0.001 and self.check_on_line((x1,y1), obs):
                    best_dist = dist
                # observations.append(dist)
            observations.append(best_dist)
        return observations

    def check_on_line(self, point, line):
        # Funtion to check if a given point is on a line
        # Requires:
        #   Point: The point to check
        #   Line: The line to check if the line on
        # Returns:
        #   on_line: Boolean for whether the point is on the line
        x, y = point
        (x3, y3), (x4, y4) = line
        print(point, line)
        if x4 == x3:
            # Vertical line
            on_line = y3 < y < y4 and np.abs(x4 - x) <= 0.001
        else:
            on_line = x3 < x < y4 and np.abs(y4 - y) <= 0.001

        return on_line


    def show_env(self, lines=[], point=None):
        # A function to show the environment and all within it
        # Requires:
        #   Nothing
        # Returns:
        #   Nothing

        plt.figure(0)
        for obs in self.obstacles:
            data = list(zip(*obs))
            plt.plot(data[0], data[1], 'k')

        for line in lines:
            data = list(zip(*line))
            plt.plot(data[0], data[1], 'r')

        if point:
            print('Plotting point')
            plt.plot(point[0], point[1], '*g')

        plt.axis('on')
        plt.xlim(-10, self.width+10)
        plt.ylim(-10, self.height+10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


if __name__ == "__main__":
    env = Base(width=100, height=100, num_obstables=1, box_w=100, box_h=100)
    thetas = []
    for i in range(1, 360, 100):
        thetas.append(i*np.pi/180)
    x, y = 20, 20
    dists = env.get_observation(x, y, 0, thetas)
    lines = []
    for theta in thetas:
        for dist in dists:
            x1, y1 = dist*np.math.cos(theta)+x, dist*np.sin(theta)+y
            line = [[x,y],[x1,y1]]
            lines.append(line)
    env.show_env(point=(x, y), lines=lines)
    print(len(env.obstacles))

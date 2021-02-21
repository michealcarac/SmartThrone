# This file contains the agents that will be made and used for training
import numpy as np
from EnvironmentClasses import Base


class InvalidMove(ArithmeticError):
    # TODO Write custom error class for handling collision errors
    pass


class Manual():

    def __init__(self, x, y, theta, FOV, LINEAR, ROTATIONAL, env):
        # Constructor for the Manual Agent class
        # Requires:
        #   x: The x position of the agent
        #   y: The y position of the agent
        #   theta: The rotation of the agent, 0 is along the x axis
        #   FOV: The list of angle difference to look in from theta
        #   LINEAR: The speed of the agent in the forward direction
        #   ROTATIONAL: The speed of the agent in rotation
        #   env: An environment object used for making observations in
        # Returns:
        #   The constructed agent object
        self.x, self.y, self.theta = x, y, theta
        self.FOV = FOV
        self.LINEAR, self.ROTATIONAL = LINEAR, ROTATIONAL
        self.env = env

    def move(self, dir):
        # A function to control the movement of an agent based on the specified direction
        # Requires:
        #   dir: The direction to move in, either 0, 1, 2 for straight, left, right respectively
        # Modifies:
        #   x, y,theta: The postions and rotation of the agent is modified according to the direction of movement
        # Returns:
        #   Nothing
        # Raises:
        #   InvalidMove: An InvalidMove error is raised if the agent is going to collide with an obstacle after the move is performed

        if dir == 0:
            dists = self.env.get_observation(self.x, self.y, self.theta)
            if dists[0] < self.LINEAR:
                raise InvalidMove
            else:
                self.x += self.LINEAR * np.math.cos(self.theta)
                self.y += self.LINEAR * np.math.sin(self.theta)
        elif dir == 1:
            self.theta += self.ROTATIONAL
            if self.theta > 2 * np.math.pi:
                self.theta -= 2 * np.math.pi
        else:
            self.theta -= self.ROTATIONAL
            if self.theta < 0:
                self.theta += 2 * np.math.pi

    def make_heading(self):
        # Helper function to show the heading of the agent
        # Requires:
        #   Nothing
        # Returns:
        #   line: a line from the agent to its next forward position

        x1 = self.x
        y1 = self.y
        x2 = self.x + self.LINEAR * np.math.cos(self.theta)
        y2 = self.y + self.LINEAR * np.math.sin(self.theta)
        return [(x1, y1), (x2, y2)]

    def show_agent(self):
        # A function to show the agent in the environment and the direction its facing
        # Requires:
        #   Nothing
        # Returns:
        #   Nothing
        heading = self.make_heading()
        self.env.show_env(lines=[heading], point=(self.x, self.y))


if __name__ == "__main__":
    env = Base()
    agent = Manual(200, 200, 0, range(-45, 45), 100, np.math.radians(5), env)
    agent.show_agent()

from kesslergame import KesslerController
from typing import Dict, Tuple
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class BasicController(KesslerController):
    def __init__(self):
        """
        Initialize any required variables or state for the controller.
        """
        self.distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
        self.relative_velocity = ctrl.Antecedent(np.arange(-50, 51, 1), 'relative_velocity')
        self.angular_direction = ctrl.Antecedent(np.arange(-180, 181, 1), 'angular_direction')
        
        self.thrust = ctrl.Consequent(np.arange(0, 101, 1), 'thrust')
        self.turn_rate = ctrl.Consequent(np.arange(-90, 91, 1), 'turn_rate')

        # Membership functions for distance
        self.distance['close'] = fuzz.trapmf(self.distance.universe, [0, 0, 10, 30])
        self.distance['medium'] = fuzz.trimf(self.distance.universe, [10, 30, 60])
        self.distance['far'] = fuzz.trapmf(self.distance.universe, [30, 60, 100, 100])

        # Membership functions for relative velocity
        self.relative_velocity['approaching'] = fuzz.trapmf(self.relative_velocity.universe, [-50, -50, -10, 0])
        self.relative_velocity['static'] = fuzz.trimf(self.relative_velocity.universe, [-10, 0, 10])
        self.relative_velocity['receding'] = fuzz.trapmf(self.relative_velocity.universe, [0, 10, 50, 50])

        # Membership functions for angular direction
        self.angular_direction['left'] = fuzz.trapmf(self.angular_direction.universe, [-180, -180, -90, 0])
        self.angular_direction['center'] = fuzz.trimf(self.angular_direction.universe, [-45, 0, 45])
        self.angular_direction['right'] = fuzz.trapmf(self.angular_direction.universe, [0, 90, 180, 180])

        # Membership functions for thrust
        self.thrust['low'] = fuzz.trimf(self.thrust.universe, [0, 2000, 4000])
        self.thrust['medium'] = fuzz.trimf(self.thrust.universe, [3000, 6000, 8000])
        self.thrust['high'] = fuzz.trimf(self.thrust.universe, [700, 9000, 10000])


        # Membership functions for turn rate
        self.turn_rate['sharp_left'] = fuzz.trimf(self.turn_rate.universe, [-90, -90, -45])
        self.turn_rate['left'] = fuzz.trimf(self.turn_rate.universe, [-45, -22.5, 0])
        self.turn_rate['straight'] = fuzz.trimf(self.turn_rate.universe, [-22.5, 0, 22.5])
        self.turn_rate['right'] = fuzz.trimf(self.turn_rate.universe, [0, 22.5, 45])
        self.turn_rate['sharp_right'] = fuzz.trimf(self.turn_rate.universe, [45, 90, 90])

        # Define rules
        self.rules = [
            ctrl.Rule(self.distance['close'] & self.relative_velocity['approaching'] & self.angular_direction['center'], 
                      (self.thrust['high'], self.turn_rate['sharp_left'])),
            ctrl.Rule(self.distance['far'] & self.relative_velocity['receding'], 
                      (self.thrust['low'], self.turn_rate['straight'])),
            # Add more rules as needed
        ]

        # Build control system
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Called every time step to decide the ship's actions.

        Arguments:
            ship_state (dict): Contains state information for your ship.
            game_state (dict): Contains state information for all objects in the game.

        Returns:
            float: thrust (m/s^2)
            float: turn_rate (degrees/s)
            bool: fire (True to shoot)
            bool: drop_mine (True to drop a mine)
        """
        asteroid = game_state['asteroids'][0]
        dist = np.hypot(asteroid["position"][0] - ship_state["position"][0], asteroid["position"][1] - ship_state["position"][1])
        rel_vel = np.hypot(asteroid['velocity'][0] - ship_state['velocity'][0], 
                   asteroid['velocity'][1] - ship_state['velocity'][1])
        angle = np.arctan2(asteroid["position"][1] - ship_state["position"][1], asteroid["position"][0] - ship_state["position"][0])
        
        self.simulator.input['distance'] = dist
        self.simulator.input['relative_velocity'] = rel_vel
        self.simulator.input['angular_direction'] = angle
        
        self.simulator.compute()
        thrust = self.simulator.output['thrust']
        turn_rate = self.simulator.output['turn_rate']
        
        fire = True
        drop_mine = False
        
        # Return actions
        return thrust, turn_rate, fire, drop_mine
    

    @property
    def name(self) -> str:
        """
        Return the name of this controller.
        """
        return "Basic Controller"
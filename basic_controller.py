from kesslergame import KesslerController
from typing import Dict, Tuple
import math

class BasicController(KesslerController):
    def __init__(self):
        """
        Initialize any required variables or state for the controller.
        """
        self.eval_frames = 0

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
        # Extract ship's position and heading
        ship_x, ship_y = ship_state["position"]
        ship_heading = ship_state["heading"]  # In degrees

        # Find the closest asteroid
        closest_asteroid = None
        min_distance = float("inf")
        for asteroid in game_state["asteroids"]:
            asteroid_x, asteroid_y = asteroid["position"]
            distance = math.sqrt((asteroid_x - ship_x)**2 + (asteroid_y - ship_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_asteroid = asteroid

        # If no asteroids exist, do nothing
        if not closest_asteroid:
            return 0.0, 0.0, False, False

        # Calculate angle to the closest asteroid
        asteroid_x, asteroid_y = closest_asteroid["position"]
        delta_x = asteroid_x - ship_x
        delta_y = asteroid_y - ship_y
        target_angle = math.degrees(math.atan2(delta_y, delta_x))  # Angle in degrees

        # Calculate the turn rate needed to face the asteroid
        angle_diff = (target_angle - ship_heading + 360) % 360  # Normalize to [0, 360)
        if angle_diff > 180:
            angle_diff -= 360  # Normalize to [-180, 180]

        turn_rate = max(-90, min(90, angle_diff))  # Limit turn rate to ±90°/s

        # Determine if aligned to fire (within 5° of target)
        fire = abs(angle_diff) < 5

        # Apply some thrust if the asteroid is far away
        thrust = 1.0 if min_distance > 200 else 0.0

        # Decide not to drop mines for now
        drop_mine = False

        # Return actions
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Return the name of this controller.
        """
        return "Basic Controller"
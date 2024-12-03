# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple


class BabyController(KesslerController):
    def __init__(self):
        self.eval_frames = 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """

        thrust = 0.0
        turn_rate = 0.0
        fire = True
        drop_mine = False
        
        ship_x, ship_y = ship_state["position"]
        ship_heading = ship_state["heading"]
        
        # Analyze asteroid positions and find the closest one
        closest_asteroid = None
        closest_distance = float('inf')
        asteroid_directions = []

        for asteroid in game_state["asteroids"]:
            asteroid_x, asteroid_y = asteroid["position"]
            distance = math.sqrt((asteroid_x - ship_x)**2 + (asteroid_y - ship_y)**2)

            # Track closest asteroid
            if distance < closest_distance:
                closest_distance = distance
                closest_asteroid = asteroid

            # Calculate angle to each asteroid
            angle_to_asteroid = math.atan2(asteroid_y - ship_y, asteroid_x - ship_x)
            asteroid_directions.append(angle_to_asteroid)

        # Move away if too close to any asteroid
        if closest_distance < 200:  # Threshold for "too close"
            thrust = 50
            drop_mine = True
        else:
            thrust = 20

        # Calculate turn direction towards region with fewer asteroids
        sparse_direction = self.find_sparse_direction(asteroid_directions, ship_heading)
        turn_rate = math.degrees(sparse_direction)

        # Always fire
        fire = True
        
        self.eval_frames +=1

        return thrust, turn_rate, fire, drop_mine

    def find_sparse_direction(self, asteroid_directions: list, ship_heading: float) -> float:
        """
        Find the direction with the fewest asteroids.
        """
        # Divide the full circle into segments (e.g., 8 directions)
        num_segments = 8
        segment_density = [0] * num_segments
        segment_width = 2 * math.pi / num_segments

        for angle in asteroid_directions:
            segment = int((angle + math.pi) / segment_width) % num_segments
            segment_density[segment] += 1

        # Find the sparsest segment
        sparsest_segment = segment_density.index(min(segment_density))
        sparse_angle = (sparsest_segment * segment_width) - math.pi

        # Adjust to ship's current heading
        return sparse_angle - math.radians(ship_heading)

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Baby Controller"


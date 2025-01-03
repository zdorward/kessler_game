# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

class FuzzyController2(KesslerController):
    
    def __init__(self):
        self.eval_frames = 0 #What is this?
        self.time_since_last_mine = 0 
        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)

        ### MOVEMENT FUZZY LOGIC
        movement_distance = ctrl.Antecedent(np.arange(0, 200, 1), 'distance')  # Adjust the range as needed
        movement_ship_speed = ctrl.Antecedent(np.arange(0, 120, 1), 'ship_speed')  # Maximum ship speed
        movement_thrust = ctrl.Consequent(np.arange(-300, 301, 1), 'thrust')  # Thrust range (-200 for backward, 200 for forward)

        # Define fuzzy sets for distance
        movement_distance['too_close'] = fuzz.zmf(movement_distance.universe, 0, 80)
        movement_distance['safe'] = fuzz.trimf(movement_distance.universe, [80, 100, 130])
        movement_distance['far'] = fuzz.smf(movement_distance.universe, 130, 200)

        # Define fuzzy sets for ship speed
        movement_ship_speed['slow'] = fuzz.zmf(movement_ship_speed.universe, 0, 30)
        movement_ship_speed['moderate'] = fuzz.trimf(movement_ship_speed.universe, [30, 50, 90])
        movement_ship_speed['fast'] = fuzz.smf(movement_ship_speed.universe, 90, 120)

        # Define fuzzy sets for thrust
        movement_thrust['backward'] = fuzz.trimf(movement_thrust.universe, [-300, -300, -150])
        movement_thrust['none'] = fuzz.trimf(movement_thrust.universe, [-150, 0, 150])
        movement_thrust['forward'] = fuzz.trimf(movement_thrust.universe, [150, 300, 300])

        # Define fuzzy movement rules
        movement_rule1 = ctrl.Rule(movement_distance['too_close'], movement_thrust['backward'])
        movement_rule2 = ctrl.Rule(movement_distance['safe'], movement_thrust['none'])
        movement_rule3 = ctrl.Rule(movement_distance['far'] & movement_ship_speed['slow'], movement_thrust['forward'])
        movement_rule4 = ctrl.Rule(movement_distance['far'] & movement_ship_speed['moderate'], movement_thrust['forward'])
        movement_rule5 = ctrl.Rule(movement_ship_speed['fast'], movement_thrust['none'])

        # Create control system and simulation
        movement_control = ctrl.ControlSystem([movement_rule1, movement_rule2, movement_rule3, movement_rule4, movement_rule5])
        self.movement_sim = ctrl.ControlSystemSimulation(movement_control)


        ### INITIALIZE FUZZY SYSTEM FOR DROPPING MINES 
        # Distance to nearest asteroid
        distance = ctrl.Antecedent(np.arange(0, 300, 1), 'distance')
        distance['near'] = fuzz.zmf(distance.universe, 0, 70)
        distance['medium'] = fuzz.trimf(distance.universe, [50, 120, 200])
        distance['far'] = fuzz.smf(distance.universe, 150, 300)

        # Relative velocity
        relative_velocity = ctrl.Antecedent(np.arange(-200, 200, 1), 'relative_velocity')
        relative_velocity['approaching'] = fuzz.zmf(relative_velocity.universe, -200, -50)
        relative_velocity['static'] = fuzz.trimf(relative_velocity.universe, [-100, 0, 100])
        relative_velocity['departing'] = fuzz.smf(relative_velocity.universe, 50, 200)

        # Ship speed
        ship_speed = ctrl.Antecedent(np.arange(0, 100, 1), 'ship_speed')
        ship_speed['slow'] = fuzz.zmf(ship_speed.universe, 0, 30)
        ship_speed['medium'] = fuzz.trimf(ship_speed.universe, [20, 50, 80])
        ship_speed['fast'] = fuzz.smf(ship_speed.universe, 60, 100)

        # Time since last mine drop (in game frames which is ~1800 for a 60 second game)
        time_since_mine = ctrl.Antecedent(np.arange(0, 1800, 1), 'time_since_mine')
        time_since_mine['short'] = fuzz.trimf(time_since_mine.universe, [0, 0, 600])
        time_since_mine['medium'] = fuzz.trimf(time_since_mine.universe, [500, 600, 900])
        time_since_mine['long'] = fuzz.trimf(time_since_mine.universe, [800, 1000, 1800])

        # Drop mine decision
        drop_mine = ctrl.Consequent(np.arange(-1, 1, 0.1), 'drop_mine')
        drop_mine['no'] = fuzz.zmf(drop_mine.universe, -1, 0)
        drop_mine['yes'] = fuzz.smf(drop_mine.universe, 0, 1)

        # Number of nearby asteroids
        asteroid_density = ctrl.Antecedent(np.arange(0, 15, 1), 'asteroid_density')
        asteroid_density['low'] = fuzz.zmf(asteroid_density.universe, 0, 4)
        asteroid_density['medium'] = fuzz.trimf(asteroid_density.universe, [1, 8, 11])
        asteroid_density['high'] = fuzz.smf(asteroid_density.universe, 7, 15)


        # Mine rules
        mrule1 = ctrl.Rule(distance['near'] & relative_velocity['approaching'], drop_mine['yes'])
        mrule2 = ctrl.Rule(distance['near'] & relative_velocity['static'], drop_mine['yes'])
        mrule3 = ctrl.Rule(distance['medium'] & ship_speed['slow'] | time_since_mine['short'], drop_mine['no'])
        mrule4 = ctrl.Rule(distance['far'], drop_mine['yes'])
        mrule5 = ctrl.Rule(ship_speed['fast'] & relative_velocity['departing'], drop_mine['no'])
        mrule6 = ctrl.Rule(distance['far'] & ship_speed['slow'] | time_since_mine['short'], drop_mine['no'])
        mrule7 = ctrl.Rule(ship_speed['slow'], drop_mine['no'])
        mrule8 = ctrl.Rule(ship_speed['fast'] & relative_velocity['approaching'], drop_mine['yes'])
        mrule9 = ctrl.Rule(distance['medium'] & ship_speed['fast'], drop_mine['yes'])
        mrule10 = ctrl.Rule(distance['medium'], drop_mine['yes'])
        mrule11 = ctrl.Rule(time_since_mine['long'], drop_mine['yes'])
        mrule12 = ctrl.Rule(time_since_mine['short'], drop_mine['no'])
        mrule13 = ctrl.Rule(time_since_mine['medium'] & ship_speed['fast'], drop_mine['no'])
        mrule_surrounded1 = ctrl.Rule(asteroid_density['low'], drop_mine['no'])
        mrule_surrounded2 = ctrl.Rule(asteroid_density['medium'], drop_mine['no'])
        mrule_surrounded3 = ctrl.Rule(asteroid_density['high'], drop_mine['yes'])

        self.mine_control = ctrl.ControlSystem()
        self.mine_control.addrule(mrule1)
        self.mine_control.addrule(mrule2)
        self.mine_control.addrule(mrule3)
        self.mine_control.addrule(mrule4)
        self.mine_control.addrule(mrule5)
        self.mine_control.addrule(mrule6)
        self.mine_control.addrule(mrule7)
        self.mine_control.addrule(mrule8)
        self.mine_control.addrule(mrule9)
        self.mine_control.addrule(mrule10)
        self.mine_control.addrule(mrule11)
        self.mine_control.addrule(mrule12)
        self.mine_control.addrule(mrule13)
        self.mine_control.addrule(mrule_surrounded1)
        self.mine_control.addrule(mrule_surrounded2)
        self.mine_control.addrule(mrule_surrounded3)



    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        self.time_since_last_mine += 1
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = 0.0
        curr_speed = (ship_state["velocity"][0]**2 + ship_state["velocity"][1]**2)**0.5

        ### movement
        self.movement_sim.input['distance'] = closest_asteroid['dist']
        self.movement_sim.input['ship_speed'] = curr_speed

        try:
            self.movement_sim.compute()
            thrust = self.movement_sim.output['thrust']
        except Exception as e:
            print("Error during movement fuzzy logic computation:", e)
            thrust = 0.0  # Fallback thrust


        ### Mine drop control system simulation, input, and computation
        dropping_mines = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        
        # Calculate relative velocity
        rel_vel = (
            ship_state["velocity"][0] * closest_asteroid["aster"]["velocity"][0] +
            ship_state["velocity"][1] * closest_asteroid["aster"]["velocity"][1]
        )
        # Define the radius for "nearby" asteroids
        surround_radius = 75.0  # Adjust based on game scale

        # Count asteroids within the radius
        nearby_asteroids = sum(
            1 for a in game_state["asteroids"]
            if math.sqrt((ship_state["position"][0] - a["position"][0])**2 +
                        (ship_state["position"][1] - a["position"][1])**2) <= surround_radius
        )

        dropping_mines.input['distance'] = closest_asteroid['dist']
        dropping_mines.input['relative_velocity'] = rel_vel
        dropping_mines.input['ship_speed'] = curr_speed
        dropping_mines.input['time_since_mine'] = self.time_since_last_mine
        dropping_mines.input['asteroid_density'] = nearby_asteroids


        # Compute fuzzy logic
        try:
            dropping_mines.compute()
            # Safely check for the output key
            if 'drop_mine' in dropping_mines.output:
                drop_mine_decision = dropping_mines.output['drop_mine']
                drop_mine = drop_mine_decision > 0
                if drop_mine:
                    #print(self.time_since_last_mine)
                    self.time_since_last_mine = 0 # Resets the timer when a mine is dropped
            else:
                #print("Warning: 'drop_mine' not found in outputs.")
                drop_mine = False
        except Exception as e:
            print("Error during fuzzy logic computation:", e)
            drop_mine = False  # Fallback if computation fails
        
        # print(game_state)
        #DEBUG
        # print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Fuzzy Controller"

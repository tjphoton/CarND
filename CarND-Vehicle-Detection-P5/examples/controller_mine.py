"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0  # colective thrust in [N]
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self):
        self,
        """Initialize the controller object and control gains"""
        # self.k_p_z = 2.0
        # self.k_d_z = 1.0
        # self.k_p_xy = 6.0
        # self.k_d_xy = 4.0
        # self.k_p_yaw = 8.0
        # self.k_p_roll_pitch = 8.0
        # self.k_p_p = 20.0
        # self.k_p_q = 20.0
        # self.k_p_r = 20.0

        self.k_p_z = 6
        self.k_d_z = 1.5
        self.k_p_xy = 24
        self.k_d_xy = 4
        self.k_p_roll_pitch = 8
        self.k_p_yaw = 4.5
        self.k_p_p = 20
        self.k_p_q = 20
        self.k_p_r = 5

        self.max_tilt = 1.0
        self.max_ascent_rate = 5
        self.max_descent_rate = 2
        self.max_speed = 5.0

        # self.k_p_z = 10
        # self.k_d_z = 0.2
        # self.k_p_xy = 2
        # self.k_d_xy = 0.2
        # self.k_p_yaw = 2.5
        # self.k_p_roll_pitch = 8
        # self.k_p_p = 0.1
        # self.k_p_q = 0.1
        # self.k_p_r = 0.04

        # self.k_p_z = 1
        # self.k_d_z = 20
        # self.k_p_xy = 1
        # self.k_d_xy = 4
        # self.k_p_yaw = 1
        # self.k_p_roll_pitch = 5
        # self.k_p_p = 23
        # self.k_p_q = 23
        # self.k_p_r = 5

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]

        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                time0 = 0.0
                time1 = 1.0
            else:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        velocity_cmd = (position1 - position0) / (time1 - time0)            
        position_cmd = velocity_cmd * (current_time - time0) + position0

        return (position_cmd, velocity_cmd, yaw_cmd)
    
    def lateral_position_control(self, 
                                 local_position_cmd, local_velocity_cmd, 
                                 local_position, local_velocity,
                                 acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command
            
        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        location_err = local_position_cmd - local_position
        velocity_err = local_velocity_cmd - local_velocity

        p_term = self.k_p_xy * location_err
        d_term = self.k_d_xy * velocity_err

        acceleration_cmd = p_term + d_term + acceleration_ff
        return acceleration_cmd
        # return np.array([0.0, 0.0])

    def altitude_control(self, 
                         altitude_cmd, vertical_velocity_cmd, 
                         altitude, vertical_velocity, 
                         attitude, 
                         acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)
            
        Returns: thrust command for the vehicle (+up)
        """
        R = euler2RM(*attitude)

        z_err = altitude_cmd - altitude
        z_err_dot = vertical_velocity_cmd - vertical_velocity
        acc_cmd = self.k_p_z * z_err + self.k_d_z * z_err_dot + acceleration_ff
        # R33 = np.cos(attitude[0])*np.cos(attitude[1])
        thrust = acc_cmd * DRONE_MASS_KG  / R[2,2]
        print("R33:", R[2,2], "thrust:", thrust)
        # return thrust
        return np.clip(thrust, 0, MAX_THRUST)
    
    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame
        
        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            thrust_cmd: vehicle thruts command in Newton
            
        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """

        R = euler2RM(*attitude)

        # target_R13 = min(max(acceleration_cmd[0].item()/thrust_cmd.item(),-1.0),1.0)
        # target_R23 = min(max(acceleration_cmd[1].item()/thrust_cmd.item(),-1.0),1.0)

        # if thrust_cmd > 0:
        #     b = np.clip(acceleration_cmd * DRONE_MASS_KG / thrust_cmd , -1.0, 1.0)
        #     target_R13 = b[0]
        #     target_R23 = b[1]

        #     #Limit maximum tilt
        #     target_pitch = np.arcsin(-target_R13)
        #     target_roll = np.arctan2(target_R23,R[2,2])
        #     tilt_norm = target_roll*target_roll + target_pitch*target_pitch 
        #     if abs(tilt_norm) > 0.5:
        #         target_pitch = target_pitch * 0.5/tilt_norm
        #         target_roll = target_roll * 0.5/tilt_norm
        #         target_R13 = -np.sin(target_pitch)
        #         target_R23 = np.sin(target_roll)*np.cos(target_pitch)

        #     b_x_c_dot = self.k_p_roll_pitch * (target_R13 - R[0,2])
        #     b_y_c_dot = self.k_p_roll_pitch * (target_R23 - R[1,2])
        #     pq_cmd =  np.array([[R[1,0], -R[0,0]],
        #                        [R[1,1], -R[0,1]]]) @ np.array([b_x_c_dot, b_y_c_dot]).T / R[2,2]
        # else:
        #     pq_cmd = np.array([0.0, 0.0])

        # return np.array([0.0, 0.0])
        c_d = thrust_cmd/DRONE_MASS_KG
        
        if thrust_cmd > 0.0:
            print("acc:", acceleration_cmd)
            print("acc0:", acceleration_cmd[0])
            print("acc1:", acceleration_cmd[1])
            print("acceleration_cmd[0]/c_d:", acceleration_cmd[0]/c_d)
            print(self.max_tilt)
            print(self.k_p_roll_pitch)

            target_R13 = -np.clip(acceleration_cmd[0]/c_d, -self.max_tilt, self.max_tilt) 
            target_R23 = -np.clip(acceleration_cmd[1]/c_d, -self.max_tilt, self.max_tilt) 
            
            p_cmd = (1/R[2, 2]) * \
                    (-R[1, 0] * self.k_p_roll_pitch * (R[0, 2]-target_R13) + \
                     R[0, 0] * self.k_p_roll_pitch * (R[1, 2]-target_R23))
            q_cmd = (1/R[2, 2]) * \
                    (-R[1, 1] * self.k_p_roll_pitch * (R[0, 2]-target_R13) + \
                     R[0, 1] * self.k_p_roll_pitch * (R[1, 2]-target_R23))
        else:  # Otherwise command no rate
            print("negative thrust command")
            p_cmd = 0.0
            q_cmd = 0.0
            thrust_cmd = 0.0
        return np.array([p_cmd, q_cmd])

    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        rate_err = body_rate_cmd - body_rate 
        moment = np.array([self.k_p_p, self.k_p_q, self.k_p_r]) * rate_err * MOI

        print("body rate:", moment)
        # return moment
        return np.clip(moment, -MAX_TORQUE, MAX_TORQUE)

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate
        
        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians
        
        Returns: target yawrate in radians/sec
        """
        # yaw_err = yaw_cmd - yaw
        # yawrate = self.k_p_yaw * yaw_err
        
        # return yawrate
        yaw_cmd = np.mod(yaw_cmd,2.0*np.pi)
        
        yaw_error = yaw_cmd-yaw
        if(yaw_error > np.pi):
            yaw_error = yaw_error-2.0*np.pi
        elif(yaw_error<-np.pi):
            yaw_error = yaw_error+2.0*np.pi
        
        yawrate_cmd = self.k_p_yaw*yaw_error
        return yawrate_cmd
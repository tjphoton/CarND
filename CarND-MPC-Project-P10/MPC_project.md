## MPC Control Project (Udacity Self-Driving Car Engineer Nanodegree)

Xinjie Qiu, September 16, 2018

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project Model Predictive Control is implemented to drive the car around the track. This time the cross track error is not given, rather we have to calculate that ourselves! Additionally, there's a 100 millisecond latency between actuations commands on top of the connection latency.

### 1. Model Implementation

The model used is a Kinematic model neglecting the complex interactions between the tires and the road. The model equations are as follow:

```
x[t] = x[t-1] + v[t-1] * cos(psi[t-1]) * dt
y[t] = y[t-1] + v[t-1] * sin(psi[t-1]) * dt
psi[t] = psi[t-1] + v[t-1] / Lf * delta[t-1] * dt
v[t] = v[t-1] + a[t-1] * dt
cte[t] = f(x[t-1]) - y[t-1] + v[t-1] * sin(epsi[t-1]) * dt
epsi[t] = psi[t] - psides[t-1] + v[t-1] * delta[t-1] / Lf * dt
```

Where:
* x, y : Car's position.
* psi : Car's heading direction.
* v : Car's velocity.
* cte : Cross-track error.
* epsi : Orientation error.
* a : Car's acceleration (throttle).
* delta : Steering angle.
* Lf : the distance between the car of mass and the front wheels

The objective is to find the acceleration (a) and the steering angle (delta) that minimize the objective function that combines different factors:

* Square sum of cte and epsi. 
* Square sum of the difference actuators to penalize a lot of actuator's actions. 
* Square sum of the difference between two consecutive actuator values to penalize sharp changes.
The weight of each factor were tuned manually to obtain a successful track ride.

### 2. Timestep Length and Elapsed Duration (N & dt)

The timestep length (N) and the lapsed duration between timesteps (dt) define the prediction horizon (the product of two variables, N and dt), which is the duration over which future predictions are made.

N, dt, and T are hyperparameters you will need to tune for each model predictive controller. T should be as large as possible (In the case of driving a car, T should be a few seconds), while dt should be as small as possible to accurately approximate a continuous reference trajectory.

To keep T = 1 second, after trying with N from 10 to 20 and dt 100 to 500 milliseconds, it's decided to choose the following values:
```
size_t N = 10;
double dt = 0.1; 
```

### 3. Polynomial Fitting and MPC Preprocessing

The waypoints provided by the simulator are transformed to the car coordinate system  A 3rd order of polynomial is fitted to the transformed waypoints. These polynomial coefficients are used to calculate the cte and epsi. They are used by the solver to create a reference trajectory.

### 4. Model Predictive Control with Latency

To handle a 100 millisecond actuator latency, the state values are calculated to predict the control actions after the delay interval.

### 5. The vehicle must successfully drive a lap around the track.

The vehicle successfully drives a lap around the track. 















In this project, we revisited the lake race track from the Behavioral Cloning Project (Project 3 in term 1). This time, however, we implemented a PID controller in C++ to maneuver the vehicle around the track!

The simulator provides the cross track error (CTE) to feed into PID controller to compute the appropriate steering angle.

![Car Animation][image1] 


[//]: # (Image References)
[image1]: ./PID_driving.gif "Car PID Driving Animation"
[image2]: ./PID_tuning_Kp.png "Tuning Kp"
[image3]: ./PID_tuning_Kd.png "Tuning Kd"
[image4]: ./PID_tuning_Ki.png "Tuning Ki"
[image5]: ./PID_tuning_Kd_2.png "Tuning Kd again"
[image6]: ./PID_tuning_Kp_2.png "Tuning Kp again"
[mp4]: ./PID_driving.mp4 "Video"


PID controller Implementation
---
PID stands for **P**roportional **I**ntegral **D**erivative, each term calculate its feedback and apply a corrected based on the difference between a desired target and a measured variable.

In this project, the cross track error (CTE) is provided by the simulator. It is the off lane center distance. Its values is possitive if the vehicle is to the right of the center line, and negative if to the left.

This CTE error itself is P error. The accumulated CTE error is assigned to I error. The differential error (current CTE minus the previous CTE) is the D error. 

These errors are calculated in PID.cpp:
```
void PID::UpdateError(double cte) {
  d_error  = cte - p_error;
  p_error  = cte;
  i_error += cte;
}
```

The sum of each error multiplying by corresponding control gains (P gain, I gain, and D gain) is the total error:
```
double PID::TotalError() {
  return Kp * p_error + Kd * d_error + Ki * i_error;
}
```

The vehicle is controlled by the steering value, which is calculated from the total error with a opposite sign. The absolute value of this steering value is capped at 1.0 to prevent over controlling. 
The steering value is calculated in main.cpp: 
```
pid.UpdateError(cte);
steer_value = - pid.TotalError();

if (steer_value > 1.0)
  steer_value = 1.0;
else if (steer_value < - 1.0)
  steer_value = - 1.0;
```

PID gains tuning
---
The PID gains tuning is the most challenge part of this project. There are a few different methods for tuning the Kp, Ki and Kd.  Ways to tune the PID constants can be done by a computer program (e.g., twiddle, SGD), by maths calculations (e.g., Ziegler-Nichols method), or by manual tuning.

In this project, I decided to practice with manual tuning so we can have a visual clue on the effect of each control gains.

In each iteration, the simulator is controlled by a set of controller gains, Kp, Ki, Kd. Each message from the controller is recorded, including the timestamps and errors. 

#### 1. Kp Tuning
The Kd, Ki terms are set to zero for now, we compare the effects of varing Kp term at 0.00, 0.05 and 0.10.

![Kp Tuning][image2]

Current best gain choice: Kp = 0.05.

Increaseing the Kp gain values causes increasesing the ossiclation frequency. But too high Kp value causes overshoting. 

The witdh of the road is about 10 meters, 5 meters to each side from the center line. Even with the best Kp chosen to 0.050, the vechile still drives off the road. Let's continue tuning with other gains to see whether that will help.

#### 2. Kd Tuning
Keep Kp to 0.05, still set Ki at zero, vary Kd at 0.0, 0.1, 1.0, 5.0, 10.0,

![Kd Tuning][image3] 

Increasing Kd help to reduce the overshot effect. In these Kd gains, Kd = 10.0 has the best result. 

Current best gain choice: Kp = 0.05, Kd = 10.0.

The maximum CTE is now reduced to about 3 meters.


#### 3. Ki Tuning
There's a problems with the above choice of Kp and Kd only control gains. Most of the time, the CTE error shifts to the right side of the center line. This systemmatic error can be reduced by addion the Integral term Ki. 

Keep Kp at 0.05, Kd at 10.0, vary Ki at 0, 0.0001, 0.001, 0.002, 0.005,

![Ki Tuning][image4] 

The introduction of Integral term does help shift the error to the center, also help to reduce the rise time, but increase the overshot, increase the setting time.

Current best gain choice: Kp = 0.05, Kd = 10.0, Ki = 0.002.

The current CTE are bounded within 2 meters, an improvement over PD only controller.

#### 4. 2nd time Kd Tuning 
Re-iteration on Kd gains. keep Kp at 0.05, Ki at 0.002, vary Kd at 10, 20, 50.

![Kd Tuning again][image5] 

Increasing Kd helps reduce overshot, but the ride is not smooth. At some points they also cause high CTE.

Current best gain choice still at: Kp = 0.05, Kd = 10.0, Ki = 0.002.

#### 5. 2nd Time Kp Tuning
Re-iterated on Ki again. Keep Kd at 10.0, Ki at 0.002, vary Kp at 0, 0.050, 0.100, 0.200, 0.500,
![Kp Tuning again][image6] 

Increase Kp help reduce rise time, reduce steady state error, decrease overshot. But it cause problem of more frequent correction. This will cause driving and passenger motion sickness. A compromise choise is Kp = 0.10. 

Current best gain choice: **Kp = 0.10, Kd = 10.0, Ki = 0.002**.
The current CTE are bounded within 1.3 meters, an improvement over previous gain setting.

#### 6. Tuning as a Science as well as an Art

PID tuning is a difficult problem, a optimal tuning is typically a compromise of multiple (and often conflicting) objectives such as short rise time and high stability. Different systems may have different behavior, different applications may have different requirements, and requirements may conflict with one another.

It is certain that a better performance might be achived with further tuning. For now, the control gains give a prettey good performance on driving the vehicle around the track.

Simulation
---

The video below (downloadable at [mp4] or click on image below for youtube video) shows the PID controller drives the vehicle, with the tuned PID gains, successfully drives a lap around the track without leave the drivable portion of the track surface.

The GIF at the beginning of this document is also extracted from this video.

[![PID Driving](./PID_thumbnail.png)](https://youtu.be/vPnpu2z76kI "PID driving")

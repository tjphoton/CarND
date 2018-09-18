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

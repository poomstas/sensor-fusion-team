# %%
import numpy as np
import matplotlib.pyplot as plt

# from filterpy.kalman import KalmanFilter
from kalman_filter import KalmanFilter # My implementation. Interchangeable with the above.

# %% Kalman Filter Initialization
kf = KalmanFilter(dim_x=6, dim_z=2) # State vector is 6x1, observation vector is 2x1
# State vector: [x, y, dx, dy, ddx, ddy] (position, velocity, acceleration)
# Observation vector: [x, y] (position)

# State transition matrix (aka the A matrix). Assume the state transition doesn't involve control input.
kf.F = np.array([[1,0,0,1,0,0],     # x = x + dx * dt
                 [0,1,0,0,1,0],     # y = y + dy * dt
                 [0,0,1,0,0,1],     # dx = dx + ddx * dt
                 [0,0,0,1,0,0],     # dy = dy + ddy * dt
                 [0,0,0,0,1,0],     # ddx = ddx
                 [0,0,0,0,0,1]])    # ddy = ddy

# Observation matrix - only x and y coordinates are observed
kf.H = np.array([[1,0,0,0,0,0],
                [0,1,0,0,0,0]])

# Covariance matrices
kf.P = np.eye(6) * 1000     # Covariance of the state. High initial uncertainty.
kf.R = np.eye(2) * 0.01     # Observation noise. e.g.) GPS sensor noise
kf.Q = np.eye(6) * 0.001    # Process noise. e.g.) Wheel slip

# Generate noisy input
timesteps = 100
xs, ys = [], []

for t in range(timesteps):
    x = 10 + t/5
    y = 5 + np.sin(t/3)
    xs.append(x + np.random.normal(0, 0.3)) # With noise
    ys.append(y + np.random.normal(0, 0.3)) # With noise
    # xs.append(x) # Without noise
    # ys.append(y) # Without noise

# %% Run KF and Visualize
plt.ion()  
fig, ax = plt.subplots()

for i in range(len(xs)):
    x, y = xs[i], ys[i] # Get noisy coord obs
    ax.scatter(x, y, c='b', marker='o') # plot noisy coord obs
    
    # Predict
    kf.predict()
    x_pred, y_pred = kf.x[0], kf.x[1] 
    
    # Update
    z = np.array([[x, y]])
    kf.update(z)
    
    # Plot prediction point
    ax.scatter(x_pred, y_pred, c='r', marker='x')

    # Plot estimate point
    if i > 0:
        x_est_prev, y_est_prev = x_est, y_est
    x_est, y_est = kf.x[0], kf.x[1]
    if i > 0:
        ax.plot([x_est_prev, x_est], [y_est_prev, y_est], c='g', marker='.', linestyle='-', linewidth=1.5, label='Estimate')

    # Draw
    fig.canvas.draw()
    plt.pause(0.2)

    if i == 0:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Kalman Filter')
        ax.legend()
    
plt.ioff()
plt.show()

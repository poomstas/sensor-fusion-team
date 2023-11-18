# %%
import numpy as np

# %%
class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.F = np.zeros((dim_x, dim_x)) # State transition matrix (aka the A matrix)
        self.H = np.zeros((dim_z, dim_x)) # Observation matrix. Assumes the observation is a linear combination of the state vector
        self.H[:, :dim_z] = np.eye(dim_z) # Only the first dim_z columns of H are identity matrix. The rest are zero matrix

        self.P = np.eye(dim_x) # Covariance of the state. Subject to direct modification
        self.R = np.eye(dim_z) # Observation noise. Subject to direct modification
        self.Q = np.eye(dim_x) # Process noise. Subject to direct modification

        self.x = np.random.randn(dim_x, 1) # Randomly intialize the state vector
    
    def predict(self): # Calculate the predicted state of the next timestep
        self.x = np.dot(self.F, self.x) # omitted: B*u + w_k
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q # P_k = F * P_k-1 * F^T + Q
    
    def update(self, z): # Update the state based on measurement z
        # Compute Kalman gain: K_k = P_k * H^T * (H * P_k * H^T + R)^-1
        inv = np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        Kk = np.dot(np.dot(self.P, self.H.T), inv)

        # Compute the estimate using the Kalman gain and the error between the measurement & prediction
        self.x = self.x + np.dot(Kk, z.T - np.dot(self.H, self.x))

        # Update the covariance matrix
        self.P = np.dot((np.eye(self.dim_x) - np.dot(Kk, self.H)), self.P)

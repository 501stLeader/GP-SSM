import numpy as np
import random # For torque profile generation

# --- Torque Profile Function ---
def generate_sinusoidal_torque(num_sines=5, max_amp=0.5, max_freq=2.0):
    """Generates a function that returns a torque based on a sum of sinusoids."""
    amps = [random.uniform(0, max_amp / num_sines) for _ in range(num_sines)]
    freqs = [random.uniform(0.1, max_freq) for _ in range(num_sines)]
    phases = [random.uniform(0, 2 * np.pi) for _ in range(num_sines)]

    print("Generated Torque Profile Parameters:")
    for i in range(num_sines):
        print(f"  Sine {i+1}: Amplitude={amps[i]:.4f}, Frequency={freqs[i]:.2f} Hz, Phase={phases[i]:.2f} rad")

    def torque_func(t):
        torque = 0.0
        for amp, freq, phase in zip(amps, freqs, phases):
            torque += amp * np.sin(2 * np.pi * freq * t + phase)
        return torque
    return torque_func

def simple_pendulum_dynamics_torqued(t, q, m, l, g, torque_func):
    """
    Calculates the derivatives of the state variables for a simple pendulum
    with an applied torque.

    The state vector q is [theta, omega], where:
    theta (float): Pendulum angle in radians.
                   *** theta=0 is the DOWNWARD vertical position. ***
                   *** theta=pi (180 deg) is the UPWARD vertical position. ***
    omega (float): Pendulum angular velocity in rad/s.

    Args:
        t (float): Current time.
        q (np.array): State vector [theta, omega].
        m (float): Mass of the pendulum bob (kg).
        l (float): Length of the pendulum rod (m).
        g (float): Acceleration due to gravity (m/s^2).
        torque_func (callable): A function torque_func(t) that returns the
                                external torque tau applied at the pivot at time t.

    Returns:
        np.array: Derivatives [theta_dot, omega_dot].
    """
    theta, omega = q
    tau = torque_func(t) # Get the torque at the current time t

    theta_dot = omega
    omega_dot = - (g / l) * np.sin(theta) + tau / (m * l**2)

    return np.array([theta_dot, omega_dot])

# --- Mass Matrix Function ---
def calculate_mass_matrix(theta, M, m, l):
    """
    Calculates the 1x1 mass matrix M(q) for the generalized coordinates [theta].
    M(q) = [ ml^2 ]
    theta=0 is DOWNWARD.
    """
    c = np.cos(theta)
    mass_mat = np.array([
        
        [m * l**2]
    ])
    return mass_mat

# --- Gravity Vector Function ---
def calculate_gravity_vector(theta, m, l, g):
    """
    Calculates the gravity vector G(q) for the generalized coordinates [x, theta].
    G(q) = [0, m*g*l*sin(theta)]^T
    where theta=0 is the DOWNWARD vertical position.

    Args:
        theta (float): Pendulum angle (rad, 0=down).
        m (float): Pendulum mass (kg).
        l (float): Pendulum length (m).
        g (float): Acceleration due to gravity (m/s^2).

    Returns:
        np.array: 1x1 gravity vector [G_theta].
    """
    s = np.sin(theta)
    gravity_vec = np.array([
        m * g * l * s
    ])
    return gravity_vec

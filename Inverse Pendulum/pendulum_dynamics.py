import numpy as np

import random # For force profile generation

# --- Force Profile Function ---
def generate_sinusoidal_force(num_sines=5, max_amp=5.0, max_freq=2.0):
    """Generates a function that returns a force based on a sum of sinusoids."""
    amps = [random.uniform(0, max_amp / num_sines) for _ in range(num_sines)]
    freqs = [random.uniform(0.1, max_freq) for _ in range(num_sines)]
    phases = [random.uniform(0, 2 * np.pi) for _ in range(num_sines)]

    print("Generated Force Profile Parameters:")
    for i in range(num_sines):
        print(f"  Sine {i+1}: Amplitude={amps[i]:.2f}, Frequency={freqs[i]:.2f} Hz, Phase={phases[i]:.2f} rad")

    def force_func(t):
        force = 0.0
        for amp, freq, phase in zip(amps, freqs, phases):
            force += amp * np.sin(2 * np.pi * freq * t + phase)
        return force
    return force_func

# --- Dynamics Function (Modified for External Force F) ---
def inverse_pendulum_dynamics_forced(t, q, M, m, l, g, force_func):
    """
    Calculates the derivatives of the state variables for the inverted pendulum
    on a cart, subject to an external force F applied to the cart.

    Args:
        t (float): Current time.
        q (np.array): State vector [x, x_dot, theta, omega].
                      *** theta=0 is the DOWNWARD vertical position. ***
                      *** theta=pi (180 deg) is the UPWARD vertical position. ***
        M (float): Mass of the cart (kg).
        m (float): Mass of the pendulum bob (kg).
        l (float): Length of the pendulum rod (m).
        g (float): Acceleration due to gravity (m/s^2).
        force_func (callable): A function force_func(t) that returns the
                               external force F applied to the cart at time t.

    Returns:
        np.array: Derivatives [x_dot, x_dot_dot, omega, omega_dot].
    """
    x, x_dot, theta, omega = q
    F = force_func(t) # Get the force at the current time t

    s = np.sin(theta)
    c = np.cos(theta)

    common_denominator = l * (M + m * s**2)

    if np.isclose(common_denominator, 0):
        print(f"Warning: Denominator near zero at t={t}, theta={theta}. Clamping accelerations.")
        x_dot_dot = 0
        omega_dot = 0
    else:
        x_dot_dot = (l * (F + m*l*s*omega**2) + m*g*l*s*c) / common_denominator
        omega_dot = (-c * (F + m*l*s*omega**2) - (M+m)*g*s) / common_denominator

    return np.array([x_dot, x_dot_dot, omega, omega_dot])

# --- Mass Matrix Function ---
def calculate_mass_matrix(theta, M, m, l):
    """
    Calculates the 2x2 mass matrix M(q) for the generalized coordinates [x, theta].
    M(q) = [[ M+m,   ml*cos(theta) ],
            [ ml*cos(theta),   ml^2 ]]
    theta=0 is DOWNWARD.
    """
    c = np.cos(theta)
    mass_mat = np.array([
        [M + m, m * l * c],
        [m * l * c, m * l**2]
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
        np.array: 2x1 gravity vector [G_x, G_theta].
    """
    s = np.sin(theta)
    gravity_vec = np.array([
        0,
        m * g * l * s
    ])
    return gravity_vec

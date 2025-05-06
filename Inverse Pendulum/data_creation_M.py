# %% Modifiziertes create_data.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation # For animation
import matplotlib.patches as patches     # For drawing shapes
import argparse
import os
import random # For force profile generation
try:
    import pandas as pd # For CSV saving
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
# Added for train-test split and PKL (though NPZ/CSV is likely better for this data)
from sklearn.model_selection import train_test_split
import pickle

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

    # Using the equations derived from Lagrangian or Newton-Euler methods:
    common_denominator = l * (M + m * s**2)

    if np.isclose(common_denominator, 0):
        print(f"Warning: Denominator near zero at t={t}, theta={theta}. Clamping accelerations.")
        # This should only happen if l=0 or (M=0 and theta=0/pi)
        x_dot_dot = 0
        omega_dot = 0
    else:
        # Re-simplified from common web sources (often derived slightly differently but equivalent)
        # Let's stick to the original script's calculation structure and add F
        x_dot_dot = (l * (F + m*l*s*omega**2) + m*g*l*s*c) / common_denominator
        omega_dot = (-c * (F + m*l*s*omega**2) - (M+m)*g*s) / common_denominator


    return np.array([x_dot, x_dot_dot, omega, omega_dot])

# --- Mass Matrix Function ---
def calculate_mass_matrix(theta, M, m, l):
    """
    Calculates the 2x2 mass matrix M(q) for the generalized coordinates [x, theta].
    M(q) = [[ M+m,   ml*cos(theta) ],
            [ ml*cos(theta),  ml^2 ]]

    Args:
        theta (float): Pendulum angle (rad, 0=down).
        M (float): Cart mass (kg).
        m (float): Pendulum mass (kg).
        l (float): Pendulum length (m).

    Returns:
        np.array: 2x2 mass matrix.
    """
    c = np.cos(theta)
    mass_mat = np.array([
        [M + m, m * l * c],
        [m * l * c, m * l**2]
    ])
    return mass_mat
def wrap_angle(angle_rad):
    """Wraps an angle in radians to the interval [-pi, pi)."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

# --- Plotting Function (Static Plots) ---
# MODIFIED: Now takes theta_wrapped_rad as input
def plot_results(t, x, x_dot, theta_wrapped_rad, omega, x_dot_dot=None, omega_dot=None, F=None):
    """Generates the plot panel of the simulation results, using pre-wrapped angle."""

    # Convert wrapped angle to degrees for plotting
    theta_wrapped_deg = np.degrees(theta_wrapped_rad)
    # ------------------------------------

    num_plots = 4
    if x_dot_dot is not None: num_plots += 1
    if omega_dot is not None: num_plots += 1
    if F is not None: num_plots += 1

    rows = (num_plots + 1) // 2 # Calculate required rows

    plt.figure("Simulation State Variables", figsize=(12, rows * 4)) # Adjust size

    plot_idx = 1
    plt.subplot(rows, 2, plot_idx); plot_idx+=1
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position vs. Time")
    plt.grid(True)

    plt.subplot(rows, 2, plot_idx); plot_idx+=1
    plt.plot(t, x_dot)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Velocity (m/s)")
    plt.title("Cart Velocity vs. Time")
    plt.grid(True)

    plt.subplot(rows, 2, plot_idx); plot_idx+=1
    # --- Plot the PRE-WRAPPED angle ---
    plt.plot(t, theta_wrapped_deg)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (Wrapped, deg)\n(0=down, +/-180=up)")
    plt.title("Pendulum Angle (Wrapped to [-180, 180]) vs. Time")
    plt.ylim(-190, 190) # Set Y-limits for clarity
    # -----------------------------
    plt.grid(True)

    plt.subplot(rows, 2, plot_idx); plot_idx+=1
    plt.plot(t, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angular Velocity (rad/s)")
    plt.title("Pendulum Angular Velocity vs. Time")
    plt.grid(True)

    if x_dot_dot is not None:
        plt.subplot(rows, 2, plot_idx); plot_idx+=1
        plt.plot(t, x_dot_dot)
        plt.xlabel("Time (s)")
        plt.ylabel("Cart Acceleration (m/s^2)")
        plt.title("Cart Acceleration vs. Time")
        plt.grid(True)

    if omega_dot is not None:
        plt.subplot(rows, 2, plot_idx); plot_idx+=1
        plt.plot(t, omega_dot)
        plt.xlabel("Time (s)")
        plt.ylabel("Pendulum Angular Accel (rad/s^2)")
        plt.title("Pendulum Angular Acceleration vs. Time")
        plt.grid(True)

    if F is not None:
        plt.subplot(rows, 2, plot_idx); plot_idx+=1
        plt.plot(t, F)
        plt.xlabel("Time (s)")
        plt.ylabel("Applied Force (N)")
        plt.title("Applied Force vs. Time")
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Inverted Pendulum Simulation Results (Forced)", y=1.02 if rows > 2 else 1.05)
    plt.show(block=False)

# --- Animation Function ---
def animate_pendulum(t_vec, x_vec, theta_vec_sim, anim_params, dt):
    """Creates and displays an animation of the inverted pendulum."""

    # Extract parameters
    PENDULUM_LENGTH = anim_params['l']
    CART_WIDTH = anim_params['cart_w']
    CART_HEIGHT = anim_params['cart_h']
    WHEEL_RADIUS = anim_params['wheel_r']
    GROUND_Y = 0.0

    # --- Angle Convention Adjustment ---
    # Simulation: theta=0 is DOWN. Animation Code: theta=0 is UP.
    # Adjust simulation theta for animation: theta_anim = theta_sim + pi
    theta_vec_anim = theta_vec_sim + np.pi # Adjust for animation

    num_frames = len(x_vec)
    INTERVAL_MS = max(10, int(dt * 1000)) # Use simulation dt, ensure minimum interval

    # --- Setup the Animation Plot ---
    fig_anim, ax_anim = plt.subplots(num="Pendulum Animation", figsize=(10, 6))

    max_pendulum_reach_x = PENDULUM_LENGTH * 1.1
    max_pendulum_reach_y = PENDULUM_LENGTH * 1.1
    min_x = np.min(x_vec) - CART_WIDTH / 2 - max_pendulum_reach_x
    max_x = np.max(x_vec) + CART_WIDTH / 2 + max_pendulum_reach_x
    min_y = GROUND_Y - WHEEL_RADIUS * 2 - 0.5
    max_y = GROUND_Y + WHEEL_RADIUS * 2 + CART_HEIGHT + max_pendulum_reach_y + 0.5

    ax_anim.set_xlim(min_x, max_x)
    ax_anim.set_ylim(min_y, max_y)
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.grid(True)
    ax_anim.set_xlabel("Cart Position (x)")
    ax_anim.set_ylabel("Y Position")
    ax_anim.set_title("Inverted Pendulum Animation")

    ground_line, = ax_anim.plot([min_x, max_x], [GROUND_Y, GROUND_Y], 'k-', lw=2)

    # --- Define Drawing Elements (Initial State) ---
    x0 = x_vec[0]
    th0_anim = theta_vec_anim[0]

    cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2
    cart_rect = patches.Rectangle((x0 - CART_WIDTH / 2, cart_y_bottom),
                                  CART_WIDTH, CART_HEIGHT, fc='royalblue', ec='black')
    ax_anim.add_patch(cart_rect)

    wheel_y = GROUND_Y + WHEEL_RADIUS
    wheel_offset_x = CART_WIDTH / 4
    wheel1 = patches.Circle((x0 - wheel_offset_x, wheel_y), WHEEL_RADIUS, fc='gray', ec='black')
    wheel2 = patches.Circle((x0 + wheel_offset_x, wheel_y), WHEEL_RADIUS, fc='gray', ec='black')
    ax_anim.add_patch(wheel1)
    ax_anim.add_patch(wheel2)

    pivot_x = x0
    pivot_y = cart_y_bottom + CART_HEIGHT

    pendulum_x_end = pivot_x + PENDULUM_LENGTH * np.sin(th0_anim)
    pendulum_y_end = pivot_y + PENDULUM_LENGTH * np.cos(th0_anim)
    pendulum_line, = ax_anim.plot([pivot_x, pendulum_x_end], [pivot_y, pendulum_y_end],
                                  'r-', lw=3, solid_capstyle='round')

    pendulum_bob = patches.Circle((pendulum_x_end, pendulum_y_end),
                                  PENDULUM_LENGTH * 0.08, fc='darkred', ec='black')
    ax_anim.add_patch(pendulum_bob)

    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, verticalalignment='top', fontsize=10)

    # --- Animation Update Function ---
    def update(frame):
        current_x = x_vec[frame]
        current_theta_anim = theta_vec_anim[frame]

        cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2
        cart_rect.set_x(current_x - CART_WIDTH / 2)
        cart_rect.set_y(cart_y_bottom)
        wheel_y = GROUND_Y + WHEEL_RADIUS
        wheel1.center = (current_x - wheel_offset_x, wheel_y)
        wheel2.center = (current_x + wheel_offset_x, wheel_y)

        pivot_x = current_x
        pivot_y = cart_y_bottom + CART_HEIGHT
        pendulum_x_end = pivot_x + PENDULUM_LENGTH * np.sin(current_theta_anim)
        pendulum_y_end = pivot_y + PENDULUM_LENGTH * np.cos(current_theta_anim)
        pendulum_line.set_data([pivot_x, pendulum_x_end], [pivot_y, pendulum_y_end])
        pendulum_bob.center = (pendulum_x_end, pendulum_y_end)

        current_time = t_vec[frame]
        time_text.set_text(f'Time: {current_time:.2f}s')

        return cart_rect, wheel1, wheel2, pendulum_line, pendulum_bob, time_text

    # --- Create and Run Animation ---
    ani = animation.FuncAnimation(fig_anim, update, frames=num_frames,
                                  interval=INTERVAL_MS, blit=True, repeat=False)
    plt.show()



def prepare_csv_dict(t_subset, x_subset, x_dot_subset, theta_wrapped_rad_subset, omega_subset,
                     x_dot_dot_subset, omega_dot_subset, F_subset,tau_subset, mass_matrix_subset):
     """Prepares a dictionary suitable for saving time-series data to CSV."""
     csv_subset_dict = {
         't': t_subset,
         'q_1': x_subset,
         'dq_1': x_dot_subset,
         'q2': theta_wrapped_rad_subset, # Use wrapped angle
         'dq2': omega_subset,
         'ddq1': x_dot_dot_subset,
         'ddq2': omega_dot_subset,
         'output_1': F_subset,
         'output_2': tau_subset,
         'mass_matrix_11': mass_matrix_subset[:, 0, 0],
         'mass_matrix_12': mass_matrix_subset[:, 0, 1], # M_12 = M_21
         'mass_matrix_22': mass_matrix_subset[:, 1, 1]
     }
     return csv_subset_dict



# --- Main Function (Modified) ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Simulate (with force), load, plot, animate an inverted pendulum, and save data splits.")
    parser.add_argument('--load', action='store_true', help='Load data instead of simulating.')
    parser.add_argument('--no-plot', action='store_true', help='Skip static plotting.')
    parser.add_argument('--animate', action='store_true', help='Show pendulum animation.')
    parser.add_argument('--save-dir', type=str, default='pendulum_data', help='Directory to save output files.')
    parser.add_argument('--base-filename', type=str, default='inv_pendulum_forced', help='Base name for output files (NPZ, CSV, PKL, splits).')
    parser.add_argument('--duration', type=float, default=20.0, help='Simulation duration (seconds).')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation time step (seconds).')
    parser.add_argument('--force-sines', type=int, default=5, help='Number of sinusoids in force profile.')
    parser.add_argument('--force-max-amp', type=float, default=10.0, help='Max total amplitude for force profile.')
    parser.add_argument('--force-max-freq', type=float, default=1.5, help='Max frequency (Hz) for force profile.')
    parser.add_argument('--init-theta-deg', type=float, default=170.0, help='Initial pendulum angle in degrees (0=down, 180=up).')
    parser.add_argument('--split', action='store_true', help='Perform train-test split (80/20) and save.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for the test set.')
    parser.add_argument('--save-mass-matrix-pkl', action='store_true', help='Save the mass matrix sequence to a separate PKL file.')


    args = parser.parse_args()

    # Create Save Directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created save directory: {args.save_dir}")

    # Construct filenames
    full_npz_filename = os.path.join(args.save_dir, f"{args.base_filename}_full.npz")
    full_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_full.csv")
    mass_matrix_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_mass_matrix.pkl")
    # --- Filenames for split data ---
    train_npz_filename = os.path.join(args.save_dir, f"{args.base_filename}_train.npz")
    test_npz_filename = os.path.join(args.save_dir, f"{args.base_filename}_test.npz")
    train_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_train.csv") # NEW
    test_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_test.csv")   # NEW
    train_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_train.pkl") # NEW
    test_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_test.pkl")   # NEW


    # --- System & Simulation Parameters ---
    M = 2.0   # Mass of the cart (kg)
    m = 0.5   # Mass of the pendulum bob (kg)
    l = 0.6   # Length of the pendulum rod (m) - Used in simulation & animation
    g = 9.81  # Gravity (m/s^2)
        # --- Animation Parameters --- (Make consistent with simulation)
    anim_params = {
        'l': l,            # Use same pendulum length
        'cart_w': l * 0.6,  # Example: cart width relative to pendulum length
        'cart_h': l * 0.3,  # Example: cart height relative to pendulum length
        'wheel_r': l * 0.08 # Example: wheel radius relative to pendulum length
    }
 # --- Simulation/Loading Logic ---
    theta_sim_wrapped = None # Initialize wrapped angle variable

    if args.load:
        print(f"Attempting to load data from '{full_npz_filename}'...")
        if os.path.exists(full_npz_filename):
            try:
                with np.load(full_npz_filename) as data:
                    t = data['t']
                    x = data['x']
                    x_dot = data['x_dot']
                    theta_sim = data['theta'] # Raw theta from sim (0=down)
                    omega = data['omega']
                    x_dot_dot = data['x_dot_dot']
                    omega_dot = data['omega_dot']
                    F_applied = data['F']
                    mass_matrix_seq = data['mass_matrix']
                    dt = data['dt'].item()
                    # --- Load WRAPPED theta if available, otherwise calculate ---
                    if 'theta_wrapped' in data:
                         theta_sim_wrapped = data['theta_wrapped']
                         print("Loaded pre-calculated wrapped angle.")
                    else:
                         print("Wrapping loaded angle...")
                         theta_sim_wrapped = wrap_angle(theta_sim)
                    # ---------------------------------------------------------
                    M = data['M'].item()
                    m = data['m'].item()
                    l = data['l'].item()
                    g = data['g'].item()
                    sim_params = {'M': M, 'm': m, 'l': l, 'g': g}
                print(f"Data loaded successfully. Detected dt = {dt:.4f}s")
            except Exception as e:
                print(f"Error loading data from {full_npz_filename}: {e}")
                return
        else:
            print(f"Error: File '{full_npz_filename}' not found. Cannot load.")
            return
    else:
        # --- Run Simulation ---
        print("Generating force profile...")
        force_profile = generate_sinusoidal_force(args.force_sines, args.force_max_amp, args.force_max_freq)
        print("Running simulation with external force...")
        theta_0_rad = np.radians(args.init_theta_deg)
        q0 = np.array([0.0, 0.0, theta_0_rad, 0.0])
        t_span = (0, args.duration)
        dt = args.dt
        steps = int((t_span[1] - t_span[0]) / dt) + 1
        t_eval = np.linspace(t_span[0], t_span[1], steps)

        def dynamics_for_solver(t, q):
            return inverse_pendulum_dynamics_forced(t, q, M, m, l, g, force_profile)

        sol = solve_ivp(dynamics_for_solver, t_span, y0=q0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)

        if not sol.success: print(f"Warning: Solver failed! Message: {sol.message}")

        t = sol.t
        q = sol.y.T
        if q.shape[0] != len(t):
             print(f"Warning: Mismatch shapes. Truncating t.")
             t = t[:q.shape[0]]
             t_eval = t_eval[:q.shape[0]]

        x = q[:, 0]
        x_dot = q[:, 1]
        theta_sim = q[:, 2] # Raw theta from sim (0=down)
        omega = q[:, 3]

        # --- Calculate Accelerations, Forces, Mass Matrix ---
        print("Calculating accelerations, forces, and mass matrix...")
        num_points = len(t)
        x_dot_dot = np.zeros(num_points)
        omega_dot = np.zeros(num_points)
        F_applied = np.zeros(num_points)
        mass_matrix_seq = np.zeros((num_points, 2, 2))
        for i in range(num_points):
            derivs = inverse_pendulum_dynamics_forced(t[i], q[i, :], M, m, l, g, force_profile)
            x_dot_dot[i] = derivs[1]
            omega_dot[i] = derivs[3]
            F_applied[i] = force_profile(t[i])
            mass_matrix_seq[i] = calculate_mass_matrix(q[i, 2], M, m, l)

        # --- Wrap the angle AFTER simulation ---
        print("Wrapping angle for saving and plotting...")
        theta_sim_wrapped = wrap_angle(theta_sim)
        # -------------------------------------

        print("Simulation complete.")

        # --- Save Full Results (NPZ and CSV) ---
        try:
            np.savez(full_npz_filename,
                     t=t, x=x, x_dot=x_dot,
                     theta=theta_sim,           # Save RAW angle
                     theta_wrapped=theta_sim_wrapped, # Save WRAPPED angle
                     omega=omega,
                     x_dot_dot=x_dot_dot, omega_dot=omega_dot,
                     F=F_applied,
                     mass_matrix=mass_matrix_seq,
                     dt=dt, M=M, m=m, l=l, g=g
                     )
            print(f"Full simulation data saved to '{full_npz_filename}'.")
        except Exception as e:
            print(f"Error saving data to NPZ {full_npz_filename}: {e}")

        if PANDAS_AVAILABLE:
            try:
                # Use the helper function for CSV dict, passing the WRAPPED angle
                full_csv_dict = prepare_csv_dict(t, x, x_dot, theta_sim_wrapped, omega,
                                                  x_dot_dot, omega_dot, F_applied, mass_matrix_seq)
                df = pd.DataFrame(full_csv_dict)
                df.to_csv(full_csv_filename, index=False, float_format='%.6f')
                print(f"Full simulation data also saved to '{full_csv_filename}'.")
            except Exception as e:
                print(f"Error saving full data to CSV {full_csv_filename}: {e}")
        else:
             print("Warning: pandas not installed. Full CSV file could not be saved.")

        if args.save_mass_matrix_pkl:
            try:
                with open(mass_matrix_pkl_filename, 'wb') as f: pickle.dump(mass_matrix_seq, f)
                print(f"Mass matrix sequence saved to '{mass_matrix_pkl_filename}'.")
            except Exception as e: print(f"Error saving mass matrix to PKL {mass_matrix_pkl_filename}: {e}")

    # --- Train/Test Split ---
    if args.split:
        print(f"Performing train-test split ({1-args.test_size:.0%}/{args.test_size:.0%})...")
        # Check if data is available and theta_sim_wrapped is calculated
        required_vars = ['t', 'x', 'x_dot', 'theta_sim', 'theta_sim_wrapped', 'omega',
                         'x_dot_dot', 'omega_dot', 'F_applied', 'mass_matrix_seq']
        if not all(v in locals() and locals()[v] is not None for v in required_vars) or \
           not all(locals()[arr_name].shape[0] == t.shape[0] for arr_name in required_vars if isinstance(locals()[arr_name], np.ndarray)):
             print("Error: Data arrays missing, mismatched lengths, or wrapped angle not calculated. Aborting split.")
        else:
            try:
                indices = np.arange(t.shape[0])
                train_indices, test_indices = train_test_split(indices, test_size=args.test_size, shuffle=False)

                # --- Prepare Split Data Dictionaries (including wrapped angle) ---
                def prepare_split_dict(subset_indices):
                    data_dict = {
                        't': t[subset_indices],
                        'x': x[subset_indices],
                        'x_dot': x_dot[subset_indices],
                        'theta': theta_sim[subset_indices], # Raw angle
                        'theta_wrapped': theta_sim_wrapped[subset_indices], # Wrapped angle
                        'omega': omega[subset_indices],
                        'x_dot_dot': x_dot_dot[subset_indices],
                        'omega_dot': omega_dot[subset_indices],
                        'F': F_applied[subset_indices],
                        'mass_matrix': mass_matrix_seq[subset_indices],
                        # Add scalar parameters here for PKL/NPZ saving convenience
                        'dt': dt, 'M': M, 'm': m, 'l': l, 'g': g
                    }
                    return data_dict

                # --- Process Training Data ---
                print("Processing training data...")
                train_data = prepare_split_dict(train_indices)
                np.savez(train_npz_filename, **train_data) # NPZ saves everything in dict
                print(f"Training data saved to '{train_npz_filename}'")
                try: # PKL
                    with open(train_pkl_filename, 'wb') as f: pickle.dump(train_data, f)
                    print(f"Training data saved to '{train_pkl_filename}'")
                except Exception as e: print(f"Error saving training data to PKL {train_pkl_filename}: {e}")
                if PANDAS_AVAILABLE: # CSV
                    try:
                        # Use helper function, extracting required arrays from train_data
                        train_csv_dict = prepare_csv_dict(
                            train_data['t'], train_data['x'], train_data['x_dot'],
                            train_data['theta_wrapped'], # Pass wrapped angle
                            train_data['omega'], train_data['x_dot_dot'],
                            train_data['omega_dot'], train_data['F'], train_data['mass_matrix']
                        )
                        df_train = pd.DataFrame(train_csv_dict)
                        df_train.to_csv(train_csv_filename, index=False, float_format='%.6f')
                        print(f"Training data saved to '{train_csv_filename}'")
                    except Exception as e: print(f"Error saving training data to CSV {train_csv_filename}: {e}")
                else: print("Warning: pandas not installed. Training CSV file could not be saved.")

                # --- Process Test Data ---
                print("Processing test data...")
                test_data = prepare_split_dict(test_indices)
                np.savez(test_npz_filename, **test_data) # NPZ
                print(f"Test data saved to '{test_npz_filename}'")
                try: # PKL
                    with open(test_pkl_filename, 'wb') as f: pickle.dump(test_data, f)
                    print(f"Test data saved to '{test_pkl_filename}'")
                except Exception as e: print(f"Error saving test data to PKL {test_pkl_filename}: {e}")
                if PANDAS_AVAILABLE: # CSV
                    try:
                        test_csv_dict = prepare_csv_dict(
                            test_data['t'], test_data['x'], test_data['x_dot'],
                            test_data['theta_wrapped'], # Pass wrapped angle
                            test_data['omega'], test_data['x_dot_dot'],
                            test_data['omega_dot'], test_data['F'], test_data['mass_matrix']
                        )
                        df_test = pd.DataFrame(test_csv_dict)
                        df_test.to_csv(test_csv_filename, index=False, float_format='%.6f')
                        print(f"Test data saved to '{test_csv_filename}'")
                    except Exception as e: print(f"Error saving test data to CSV {test_csv_filename}: {e}")
                else: print("Warning: pandas not installed. Test CSV file could not be saved.")

            except Exception as e:
                 print(f"Error during train-test split or saving: {e}")

    # --- Plotting / Animation ---
    plot_requested = not args.no_plot
    animation_requested = args.animate

    # Check if data exists (relevant if loading failed or split aborted early)
    if 't' not in locals() or theta_sim_wrapped is None:
        print("Error: No data available for plotting or animation.")
        return

    if plot_requested:
        print("Generating static plots...")
        # Pass the WRAPPED angle to the plotting function
        plot_results(t, x, x_dot, theta_sim_wrapped, omega, x_dot_dot, omega_dot, F_applied)
        print("Static plot window generated.")

    if animation_requested:
        print("Generating animation...")
        # Animation still uses the RAW simulation theta (theta_sim)
        animate_pendulum(t, x, theta_sim, anim_params, dt)
        print("Animation window closed.")

    # --- Cleanup/Exit Logic [NO CHANGE] ---
    if not plot_requested and not animation_requested:
        print("No plotting or animation requested.")
    if plot_requested and not animation_requested:
        print("Close the 'Simulation State Variables' plot window to exit.")
        plt.show(block=True) # Make static plots blocking
    print("Script finished.")

# --- Entry Point ---
if __name__ == "__main__":
    main()

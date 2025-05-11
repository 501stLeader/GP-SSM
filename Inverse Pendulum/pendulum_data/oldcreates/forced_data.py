# %% Modifiziertes create_data.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation # For animation
import matplotlib.patches as patches     # For drawing shapes
import argparse
import os
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
                      theta=0 is downward vertical.
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

    # Denominator term (common in the coupled equations)
    # D = M + m * s**2 # Simplified denominator often seen, derived from ml^2(M+ms^2)
    D = M + m * s**2 # Check if this is correct or if it should be ml^2(M+ms^2)
                     # The matrix inversion gives ml^2(M+ms^2)
                     # Let's use the derived x_dot_dot and omega_dot directly
    
    common_denominator = M + m * s**2

    if np.isclose(common_denominator, 0):
        print(f"Warning: Denominator near zero at t={t}, theta={theta}. Clamping accelerations.")
        # This happens if M=0 and sin(theta)=0, which is unlikely
        x_dot_dot = 0
        omega_dot = 0
    else:
        # Calculate accelerations using the derived coupled equations
        # x_dot_dot = (F + m*l*s*omega**2 - m*g*s*c) / (M + m*s**2) # From direct derivation earlier
        # omega_dot = (-F*c - m*l*s*c*omega**2 + (M+m)*g*s) / (l * (M + m*s**2)) # From direct derivation earlier

        # Let's re-check the source of equations in the original script
        # Original x_dot_dot = (m*g*s*c - m*l*omega**2*s) / (M + m*s**2)
        # This is missing the Force F. Add F to the numerator.
        x_dot_dot = (F + m*g*s*c - m*l*omega**2*s) / common_denominator

        # Original omega_dot = (x_dot_dot * c + g*s) / l
        # Use the newly calculated x_dot_dot (which includes F) here.
        # This assumes the pivot point relationship holds even with F.
        omega_dot = (x_dot_dot * c + g*s) / l

        # Alternative check with the fully derived omega_dot:
        # omega_dot_alt = (-F*c - m*l*s*c*omega**2 + (M+m)*g*s) / (l * common_denominator)
        # It's crucial these match. Let's test the original script's structure first.

    return np.array([x_dot, x_dot_dot, omega, omega_dot])

# --- Plotting Function (Static Plots) ---
# (Remains the same as before, assuming data includes force now)
def plot_results(t, x, x_dot, theta, omega, F=None): # Add optional F
    """Generates the plot panel of the simulation results."""
    theta_deg = np.degrees(theta) # Convert theta to degrees for plotting

    num_plots = 4 if F is None else 5
    plt.figure("Simulation State Variables", figsize=(12, 10 if num_plots == 5 else 8)) # Adjust size if F is plotted

    plt.subplot(num_plots // 2 + num_plots % 2, 2, 1) # Adjust subplot layout
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position vs. Time")
    plt.grid(True)

    plt.subplot(num_plots // 2 + num_plots % 2, 2, 2)
    plt.plot(t, x_dot)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Velocity (m/s)")
    plt.title("Cart Velocity vs. Time")
    plt.grid(True)

    plt.subplot(num_plots // 2 + num_plots % 2, 2, 3)
    plt.plot(t, theta_deg)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (deg from downward vertical)")
    plt.title("Pendulum Angle vs. Time")
    plt.grid(True)

    plt.subplot(num_plots // 2 + num_plots % 2, 2, 4)
    plt.plot(t, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angular Velocity (rad/s)")
    plt.title("Pendulum Angular Velocity vs. Time")
    plt.grid(True)

    if F is not None:
        plt.subplot(num_plots // 2 + num_plots % 2, 2, 5)
        plt.plot(t, F)
        plt.xlabel("Time (s)")
        plt.ylabel("Applied Force (N)")
        plt.title("Applied Force vs. Time")
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Inverted Pendulum Simulation Results (Forced)", y=1.02)
    plt.show(block=False)

# --- Animation Function ---
# (Remains the same as before)
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
    # For animation purposes, it's often more intuitive if theta=0 is UP.
    # Let's adjust theta for animation display IF the simulation uses 0=down.
    # The simulation dynamics here use theta=0 DOWNWARDS.
    # The animation code needs theta=0 UPWARDS.
    theta_vec_anim = theta_vec_sim + np.pi # Adjust for animation

    num_frames = len(x_vec)
    # Calculate interval intelligently, aiming for ~30-60 fps if possible
    real_time_duration = t_vec[-1] - t_vec[0]
    target_fps = 30
    desired_interval_ms = 1000 / target_fps
    # Adjust playback speed if simulation is very long or short
    # speed_multiplier = real_time_duration / (num_frames * desired_interval_ms / 1000) # Estimated playback speed vs real time
    INTERVAL_MS = max(10, int(dt * 1000)) # Use simulation dt, but ensure minimum interval

    # --- Setup the Animation Plot ---
    fig_anim, ax_anim = plt.subplots(num="Pendulum Animation", figsize=(10, 6)) # Give figure a name

    # Determine plot limits dynamically
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

    # Draw static ground line
    ground_line, = ax_anim.plot([min_x, max_x], [GROUND_Y, GROUND_Y], 'k-', lw=2)

    # --- Define Drawing Elements (Initial State) ---
    x0 = x_vec[0]
    th0_anim = theta_vec_anim[0] # Use adjusted angle for animation start

    cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2
    cart_center_y = cart_y_bottom + CART_HEIGHT / 2 # Not used directly for rect y
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
    pivot_y = cart_y_bottom + CART_HEIGHT # Pivot at the top center of the cart

    # Pendulum calculation using adjusted angle (theta_anim=0 is UP)
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
        current_theta_anim = theta_vec_anim[frame] # Use adjusted angle

        # Update Cart, Wheels
        cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2
        cart_rect.set_x(current_x - CART_WIDTH / 2)
        cart_rect.set_y(cart_y_bottom)
        wheel_y = GROUND_Y + WHEEL_RADIUS
        wheel1.center = (current_x - wheel_offset_x, wheel_y)
        wheel2.center = (current_x + wheel_offset_x, wheel_y)

        # Update Pendulum
        pivot_x = current_x
        pivot_y = cart_y_bottom + CART_HEIGHT
        pendulum_x_end = pivot_x + PENDULUM_LENGTH * np.sin(current_theta_anim)
        pendulum_y_end = pivot_y + PENDULUM_LENGTH * np.cos(current_theta_anim)
        pendulum_line.set_data([pivot_x, pendulum_x_end], [pivot_y, pendulum_y_end])
        pendulum_bob.center = (pendulum_x_end, pendulum_y_end)

        # Update time text
        current_time = t_vec[frame]
        time_text.set_text(f'Time: {current_time:.2f}s')

        return cart_rect, wheel1, wheel2, pendulum_line, pendulum_bob, time_text

    # --- Create and Run Animation ---
    ani = animation.FuncAnimation(fig_anim, update, frames=num_frames,
                                  interval=INTERVAL_MS, blit=True, repeat=False)

    plt.show() # Show the animation window

# --- Main Function (Modified) ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Simulate (with force), load, plot, and animate an inverted pendulum.")
    parser.add_argument('--load', action='store_true', help='Load data instead of simulating.')
    parser.add_argument('--no-plot', action='store_true', help='Skip static plotting.')
    parser.add_argument('--animate', action='store_true', help='Show pendulum animation.')
    parser.add_argument('--filename', type=str, default='inversePendulum_forced.npz', help='File path for saving/loading data.') # Changed default name
    parser.add_argument('--duration', type=float, default=20.0, help='Simulation duration (seconds).')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation time step (seconds).')
    parser.add_argument('--force-sines', type=int, default=5, help='Number of sinusoids in force profile.')
    parser.add_argument('--force-max-amp', type=float, default=10.0, help='Max total amplitude for force profile.')
    parser.add_argument('--force-max-freq', type=float, default=1.5, help='Max frequency (Hz) for force profile.')
    parser.add_argument('--init-theta-deg', type=float, default=170.0, help='Initial pendulum angle in degrees (0=down, 180=up).')


    args = parser.parse_args()

    # --- System & Simulation Parameters ---
    M = 1.0   # Mass of the cart (kg)
    m = 0.5   # Mass of the pendulum bob (kg)
    l = 0.6   # Length of the pendulum rod (m) - Used in simulation & animation
    g = 9.81  # Gravity (m/s^2)
    sim_params = {'M': M, 'm': m, 'l': l, 'g': g}

    # --- Animation Parameters --- (Make consistent with simulation)
    anim_params = {
        'l': l,                 # Use same pendulum length
        'cart_w': l * 0.6,      # Example: cart width relative to pendulum length
        'cart_h': l * 0.3,      # Example: cart height relative to pendulum length
        'wheel_r': l * 0.08     # Example: wheel radius relative to pendulum length
    }

    # --- Simulation/Loading Logic ---
    if args.load:
        print(f"Attempting to load data from '{args.filename}'...")
        if os.path.exists(args.filename):
            try:
                with np.load(args.filename) as data:
                    t = data['t']
                    x = data['x']
                    x_dot = data['x_dot']
                    theta_sim = data['theta'] # Raw theta from simulation/file
                    omega = data['omega']
                    x_dot_dot = data['x_dot_dot'] # Load accelerations
                    omega_dot = data['omega_dot'] # Load accelerations
                    F = data['F']             # Load force
                    dt = data['dt']           # Load dt
                print(f"Data loaded successfully. Detected dt = {dt:.4f}s")
            except Exception as e:
                print(f"Error loading data from {args.filename}: {e}")
                return
        else:
            print(f"Error: File '{args.filename}' not found. Cannot load.")
            return
    else:
        print("Generating force profile...")
        force_profile = generate_sinusoidal_force(
            num_sines=args.force_sines,
            max_amp=args.force_max_amp,
            max_freq=args.force_max_freq
        )

        print("Running simulation with external force...")
        # Initial state (theta=0 is DOWNWARD vertical)
        # Convert initial angle from degrees (0=down) to radians
        theta_0_rad = np.radians(args.init_theta_deg)
        # Start near upright equilibrium if desired, e.g., init_theta_deg = 175
        x_0, x_dot_0, omega_0 = 0.0, 0.0, 0.0
        q0 = np.array([x_0, x_dot_0, theta_0_rad, omega_0])

        t_span = (0, args.duration)  # Simulation time interval
        dt = args.dt                 # Time step for evaluation points
        steps = int((t_span[1] - t_span[0]) / dt) + 1
        t_eval = np.linspace(t_span[0], t_span[1], steps)

        # Wrap the dynamics function to include parameters and the force function
        def dynamics_for_solver(t, q):
            # M, m, l, g are accessible from the outer scope (main)
            # force_profile is also accessible
            return inverse_pendulum_dynamics_forced(t, q, M, m, l, g, force_profile)

        sol = solve_ivp(
            fun=dynamics_for_solver,
            t_span=t_span, y0=q0, method='RK45', t_eval=t_eval,
            # dense_output=True # Can be useful but t_eval is often sufficient
            rtol=1e-6, atol=1e-8 # Stricter tolerances might be needed
        )

        if not sol.success:
            print(f"Warning: Solver failed! Message: {sol.message}")
            # Decide how to handle failure - maybe exit or try different parameters

        t = sol.t
        q = sol.y.T

        # Check for shape mismatch which can happen if solve_ivp fails early
        if q.shape[0] != len(t):
             print(f"Warning: Mismatch between t ({len(t)}) and q ({q.shape[0]}) shapes. Truncating t.")
             t = t[:q.shape[0]] # Truncate t to match output length
             t_eval = t_eval[:q.shape[0]]

        x = q[:, 0]
        x_dot = q[:, 1]
        theta_sim = q[:, 2] # Raw theta from simulation (0 is down)
        omega = q[:, 3]

        # Calculate accelerations and forces at the evaluation points t_eval
        print("Recalculating accelerations and forces at evaluation points...")
        x_dot_dot = np.zeros_like(t)
        omega_dot = np.zeros_like(t)
        F_applied = np.zeros_like(t)
        for i in range(len(t)):
            # Use the state *at that time* from the solution
            current_q = q[i, :]
            current_t = t[i]
            # Calculate derivatives using the state and force at t_i
            derivatives = inverse_pendulum_dynamics_forced(current_t, current_q, M, m, l, g, force_profile)
            x_dot_dot[i] = derivatives[1]
            omega_dot[i] = derivatives[3]
            # Calculate the force applied at that time
            F_applied[i] = force_profile(current_t)

        print("Simulation complete.")

        # --- Save Results (including F, accelerations, and dt) ---
        try:
            np.savez(args.filename,
                     t=t, x=x, x_dot=x_dot, theta=theta_sim, omega=omega, # States
                     x_dot_dot=x_dot_dot, omega_dot=omega_dot, # Accelerations
                     F=F_applied, # Applied Force
                     dt=dt, M=M, m=m, l=l, g=g # Parameters
                    )
            npz_filename = args.filename
            csv_filename = os.path.splitext(npz_filename)[0] + ".csv"
            # Importiere pandas (am besten am Anfang des Skripts)
            import pandas as pd

            # Erstelle ein Dictionary mit den Zeitreihendaten
            # (Stelle sicher, dass alle Arrays die gleiche Länge haben!)
            csv_data = {
                'time_s': t,
                'cart_pos_m': x,
                'cart_vel_mps': x_dot,
                'pendulum_angle_rad': theta_sim, # Beachte: Winkel in Radiant
                'pendulum_ang_vel_radps': omega,
                'cart_accel_mps2': x_dot_dot,
                'pendulum_ang_accel_radps2': omega_dot,
                'applied_force_N': F_applied
            }

            # Erstelle einen pandas DataFrame
            df = pd.DataFrame(csv_data)

            # Speichere den DataFrame als CSV
            # index=False -> Schreibt die DataFrame-Indexspalte nicht in die CSV
            df.to_csv(csv_filename, index=False, float_format='%.6f') # Format für Float-Zahlen
            print(f"Simulation data also saved to '{csv_filename}'.")

        except ImportError:
             print("Warnung: pandas ist nicht installiert. CSV-Datei konnte nicht gespeichert werden.")
             print("Bitte installieren Sie pandas: pip install pandas")
        except Exception as e:
            print(f"Error saving data to CSV {csv_filename}: {e}")
            print(f"Simulation data saved to '{args.filename}'.")
        except Exception as e:
            print(f"Error saving data to {args.filename}: {e}")

            

        
    # --- Plotting / Animation ---
    plot_requested = not args.no_plot
    animation_requested = args.animate

    # Ensure data exists before trying to plot or animate
    if 't' not in locals():
        print("Error: No data available for plotting or animation.")
        return

    if plot_requested:
        print("Generating static plots...")
        # Pass force 'F' if it exists (it should unless loaded from old file)
        F_to_plot = F if 'F' in locals() else None
        plot_results(t, x, x_dot, theta_sim, omega, F=F_to_plot)
        print("Static plot window generated.") # Continues after plt.show(block=False)

    if animation_requested:
        print("Generating animation...")
        # Pass raw simulation theta (theta_sim) - adjustment happens inside animate_pendulum
        animate_pendulum(t, x, theta_sim, anim_params, dt)
        print("Animation window closed.") # Only prints after animation window is closed

    # If neither plot nor animation requested
    if not plot_requested and not animation_requested:
         print("No plotting or animation requested.")

    # If only static plots were shown non-blockingly, keep script alive until user closes plots
    if plot_requested and not animation_requested:
        print("Close the 'Simulation State Variables' plot window to exit.")
        plt.show(block=True) # Make the static plots blocking now

    print("Script finished.")

# --- Entry Point ---
if __name__ == "__main__":
    main()
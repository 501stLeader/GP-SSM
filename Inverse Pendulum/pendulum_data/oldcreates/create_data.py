# main.py
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation # For animation
import matplotlib.patches as patches     # For drawing shapes
import argparse
import os

# --- Dynamics Function ---
# (Using the dynamics provided in the original script)

def inverse_pendulum_dynamics(t, q, M, m, l, g):
    x, x_dot, theta, omega = q
    denominator_x_accel = M + m * np.sin(theta)**2

    if np.isclose(denominator_x_accel, 0):
        print(f"Warning: Denominator near zero at t={t}, theta={theta}. Clamping accelerations.")
        x_dot_dot = 0
        omega_dot = 0
    else:
        x_dot_dot = (m*g*np.sin(theta)*np.cos(theta)-m*l*omega**2*np.sin(theta))/(M+m*np.sin(theta)**2)
        omega_dot = (x_dot_dot *np.cos(theta)+g*np.sin(theta))/l
    return np.array([x_dot, x_dot_dot, omega, omega_dot])

# --- Plotting Function (Static Plots) ---
def plot_results(t, x, x_dot, theta, omega):
    """Generates the 4-panel plot of the simulation results."""
    theta_deg = np.degrees(theta) # Convert theta to degrees for plotting

    plt.figure("Simulation State Variables", figsize=(12, 8)) # Give figure a name

    plt.subplot(2, 2, 1)
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Position (m)")
    plt.title("Cart Position vs. Time")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t, x_dot)
    plt.xlabel("Time (s)")
    plt.ylabel("Cart Velocity (m/s)")
    plt.title("Cart Velocity vs. Time")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    # Plot angle relative to upward vertical for convention consistency if desired
    # theta_up_deg = np.degrees(theta + np.pi) # Or adjust label accordingly
    plt.plot(t, theta_deg)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (deg from downward vertical)") # Adjusted label
    plt.title("Pendulum Angle vs. Time")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, omega)
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angular Velocity (rad/s)")
    plt.title("Pendulum Angular Velocity vs. Time")
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Inverted Pendulum Simulation Results", y=1.02)
    plt.show(block=False) # Use block=False if animation follows, otherwise True or default

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
    theta_vec_anim = theta_vec_sim 

    num_frames = len(x_vec)
    INTERVAL_MS = int(dt * 1000)

    # --- Setup the Animation Plot ---
    fig_anim, ax_anim = plt.subplots(num="Pendulum Animation", figsize=(10, 6)) # Give figure a name

    min_x = np.min(x_vec) - CART_WIDTH / 2 - PENDULUM_LENGTH * 1.2
    max_x = np.max(x_vec) + CART_WIDTH / 2 + PENDULUM_LENGTH * 1.2
    min_y = GROUND_Y - WHEEL_RADIUS * 2 - 0.5
    # Max Y considers pendulum pointing straight up (L) from top of cart
    max_y = GROUND_Y + WHEEL_RADIUS * 2 + CART_HEIGHT + PENDULUM_LENGTH + 0.5

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
    th0_anim = theta_vec_anim[0] # Use adjusted angle

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

# --- Main Function ---
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Simulate, load, plot, and animate an inverted pendulum.")
    parser.add_argument('--load', action='store_true', help='Load data instead of simulating.')
    parser.add_argument('--no-plot', action='store_true', help='Skip static plotting.')
    parser.add_argument('--animate', action='store_true', help='Show pendulum animation.') # New flag
    parser.add_argument('--filename', type=str, default='inversePendulum.npz', help='File path for saving/loading data.')
    args = parser.parse_args()

    # --- System & Simulation Parameters ---
    M = 1.0   # Mass of the cart (kg)
    m = 1.0   # Mass of the pendulum bob (kg)
    l = 0.5   # Length of the pendulum rod (m) - Used in simulation & animation
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
                    # Try to load dt, otherwise calculate from t
                    if 'dt' in data:
                        dt = data['dt']
                    elif len(t) > 1:
                        dt = np.mean(np.diff(t))
                    else:
                        dt = 0.01 # Fallback dt
                print(f"Data loaded successfully. Detected dt = {dt:.4f}s")
            except Exception as e:
                print(f"Error loading data from {args.filename}: {e}")
                return
        else:
            print(f"Error: File '{args.filename}' not found. Cannot load.")
            return
    else:
        print("Running simulation...")
        # Initial state (theta=0 is DOWNWARD vertical)
        x_0, x_dot_0, theta_0, omega_0 = 0.0, 0.0, 0+0.2*np.pi, 0.0 # Pendulum slightly off bottom
        q0 = np.array([x_0, x_dot_0, theta_0, omega_0])

        t_span = (0, 10)  # Simulation time interval
        dt = 0.01         # Time step for evaluation points
        steps = int((t_span[1] - t_span[0]) / dt) + 1
        t_eval = np.linspace(t_span[0], t_span[1], steps)
        all_q_dots = []
        def augmented_dynamics(t, q):
            q_dot = inverse_pendulum_dynamics(t, q, M, m, l, g)
            all_q_dots.append(q_dot)
            return q_dot
        sol = solve_ivp(
            fun=augmented_dynamics,
            t_span=t_span, y0=q0, method='RK45', t_eval=t_eval
        )

        if not sol.success:
            print(f"Warning: Solver failed! Message: {sol.message}")
        
        t = sol.t
        q = sol.y.T
        if q.shape[0] != len(t):
             print(f"Warning: Mismatch between t ({len(t)}) and q ({q.shape[0]}) shapes.")

        x = q[:, 0]
        x_dot = q[:, 1]
        theta_sim = q[:, 2] # Raw theta from simulation (0 is down)
        omega = q[:, 3]
        
        # Extract the calculated derivatives
        all_q_dots_array = np.array(all_q_dots)
        # Ensure the length matches (solve_ivp might take internal steps)
        if len(all_q_dots_array) == len(t):
            x_dot_dot = all_q_dots_array[:, 1]
            omega_dot = all_q_dots_array[:, 3]
        else:
            print("Warning: Length mismatch in calculated derivatives. Recalculating.")
            x_dot_dot = np.zeros_like(x)
            omega_dot = np.zeros_like(omega)
            for i in range(len(t)):
                _, current_x_dot_dot, _, current_omega_dot = inverse_pendulum_dynamics(t[i], q[i, :], M, m, l, g)
                x_dot_dot[i] = current_x_dot_dot
                omega_dot[i] = current_omega_dot
        print("Simulation complete.")

        # --- Save Results (including dt) ---
        try:
            np.savez(args.filename, t=t, x=x, x_dot=x_dot, theta=theta_sim, omega=omega, x_dot_dot=x_dot_dot, omega_dot=omega_dot) # Save dt and accelerations
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
        plot_results(t, x, x_dot, theta_sim, omega)
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
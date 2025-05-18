import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import matplotlib.patches as patches
import os 

def wrap_angle(angle_rad):
    """Wraps an angle in radians to the interval [-pi, pi)."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def plot_simple_pendulum_results(t, theta_wrapped_rad, omega, omega_dot=None, tau=None, save_dir=None):
    """
    Generates plots for the simple pendulum simulation results and optionally saves them.

    Args:
        t (np.array): Time vector.
        theta_wrapped_rad (np.array): Pendulum angle (wrapped to [-pi, pi)) in radians.
        omega (np.array): Pendulum angular velocity in rad/s.
        omega_dot (np.array, optional): Pendulum angular acceleration in rad/s^2. Defaults to None.
        tau (np.array, optional): Applied torque in Nm. Defaults to None.
        save_dir (str, optional): Directory path to save the plots. If None, plots are not saved.
    """
    theta_wrapped_deg = np.degrees(theta_wrapped_rad)
    num_plots = 2
    if omega_dot is not None: num_plots += 1
    if tau is not None: num_plots += 1
    rows = (num_plots + 1) // 2

    fig = plt.figure("Simple Pendulum Simulation State Variables", figsize=(12, rows * 4))
    plot_idx = 1

    plt.subplot(rows, 2, plot_idx); plot_idx+=1;
    plt.plot(t, theta_wrapped_deg);
    plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angle (Wrapped, deg)\n(0=down, +/-180=up)");
    plt.title("Pendulum Angle (Wrapped to [-180, 180]) vs. Time");
    plt.ylim(-190, 190); plt.grid(True)

    plt.subplot(rows, 2, plot_idx); plot_idx+=1;
    plt.plot(t, omega);
    plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angular Velocity (rad/s)");
    plt.title("Pendulum Angular Velocity vs. Time"); plt.grid(True)

    if omega_dot is not None:
        plt.subplot(rows, 2, plot_idx); plot_idx+=1;
        plt.plot(t, omega_dot);
        plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angular Accel (rad/s^2)");
        plt.title("Pendulum Angular Acceleration vs. Time"); plt.grid(True)

    if tau is not None:
        plt.subplot(rows, 2, plot_idx); plot_idx+=1;
        plt.plot(t, tau);
        plt.xlabel("Time (s)"); plt.ylabel("Applied Torque (Nm)");
        plt.title("Applied Torque vs. Time"); plt.grid(True)

    plt.tight_layout();
    plt.suptitle("Simple Pendulum Simulation Results (Torqued)", y=1.02 if rows > 2 else 1.05);


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "simple_pendulum_state_plots.png")
    print(f"Saving plots to {save_path}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show(block=False)



def animate_simple_pendulum(t_vec, theta_vec_sim, anim_params, dt, save_path=None):
    """
    Generates an animation of the simple pendulum and optionally saves it.

    Args:
        t_vec (np.array): Time vector.
        theta_vec_sim (np.array): Pendulum angle in radians from simulation (0=down).
        anim_params (dict): Dictionary containing animation parameters like 'l' (pendulum length).
        dt (float): Time step between frames (used for animation interval).
        save_path (str, optional): File path to save the animation (e.g., 'animation.gif', 'animation.mp4'). If None, animation is not saved.
    """
    # Switch backend to 'Agg' for saving if a save_path is provided and not showing interactively
    # Check if a non-interactive backend is already in use.
    current_backend = matplotlib.get_backend()
    if  'interactive' not in current_backend.lower():
         pass # Already a non-interactive backend, good for saving
    else:
        matplotlib.use('Agg')
        pass 

    PENDULUM_LENGTH = anim_params['l']
    theta_vec_anim = theta_vec_sim 

    num_frames = len(t_vec)
    
    INTERVAL_MS = max(10, int(dt * 1000))

    fig_anim, ax_anim = plt.subplots(num="Simple Pendulum Animation", figsize=(8, 8))

    # Determine animation limits based on pendulum swing
    max_pendulum_reach = PENDULUM_LENGTH * 1.1
    pivot_x_fixed = 0.0 # Fixed pivot point X coordinate
    pivot_y_fixed = 0.0 # Fixed pivot point Y coordinate

    ax_anim.set_xlim(pivot_x_fixed - max_pendulum_reach, pivot_x_fixed + max_pendulum_reach)
    ax_anim.set_ylim(pivot_y_fixed - max_pendulum_reach, pivot_y_fixed + max_pendulum_reach)
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.grid(True)
    ax_anim.set_xlabel("X Position (m)")
    ax_anim.set_ylabel("Y Position (m)")
    ax_anim.set_title("Simple Pendulum Animation")

    # Draw the fixed pivot point
    ax_anim.plot(pivot_x_fixed, pivot_y_fixed, 'ko', markersize=8)

    # Initial pendulum position
    th0_anim = theta_vec_anim[0]
    pendulum_x_end = pivot_x_fixed + PENDULUM_LENGTH * np.sin(th0_anim)
    # Note: positive y is upward in standard plots, but in physics theta=0 is often down.
    # If theta=0 is downward, the y coordinate of the bob is pivot_y - L*cos(theta).
    pendulum_y_end = pivot_y_fixed - PENDULUM_LENGTH * np.cos(th0_anim)


    pendulum_line, = ax_anim.plot([pivot_x_fixed, pendulum_x_end], [pivot_y_fixed, pendulum_y_end], 'r-', lw=3, solid_capstyle='round')
    pendulum_bob = patches.Circle((pendulum_x_end, pendulum_y_end), PENDULUM_LENGTH * 0.08, fc='darkred', ec='black')
    ax_anim.add_patch(pendulum_bob)

    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, verticalalignment='top', fontsize=10)

    def update(frame):
        current_theta_anim = theta_vec_anim[frame]

        # Calculate bob position relative to the fixed pivot
        current_pendulum_x_end = pivot_x_fixed + PENDULUM_LENGTH * np.sin(current_theta_anim)
        current_pendulum_y_end = pivot_y_fixed - PENDULUM_LENGTH * np.cos(current_theta_anim) # Negative because theta=0 is down

        pendulum_line.set_data([pivot_x_fixed, current_pendulum_x_end], [pivot_y_fixed, current_pendulum_y_end])
        pendulum_bob.center = (current_pendulum_x_end, current_pendulum_y_end)

        current_time = t_vec[frame]
        time_text.set_text(f'Time: {current_time:.2f}s')

        return pendulum_line, pendulum_bob, time_text # Return objects that were modified

    # Create the animation
    ani = animation.FuncAnimation(fig_anim, update, frames=num_frames, interval=INTERVAL_MS, blit=True, repeat=False)


    print(f"Saving animation to {save_path}")
    # Determine the writer based on file extension
    if save_path.lower().endswith('.gif'):
        writer = 'pillow' # Requires Pillow library
    elif save_path.lower().endswith('.mp4'):
        writer = 'ffmpeg' # Requires ffmpeg installed and in PATH, or matplotlib.rcParams['animation.ffmpeg_path'] set
    else:
        print("Warning: Unsupported animation save format. Please use .gif or .mp4")
        writer = None

    if writer:
        try:
            ani.save(save_path, writer=writer)
            print("Animation saved successfully.")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Please ensure you have the necessary writer installed (Pillow for .gif, ffmpeg for .mp4).")
    if not save_path or (save_path and writer is None): # Show if not saving or if saving failed due to unsupported format
         plt.show() # This will block execution until the animation window is closed.
    elif save_path and writer:
        # Close the figure to free memory after saving if not showing interactively
        plt.close(fig_anim)

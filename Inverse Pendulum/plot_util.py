import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation # For animation
import matplotlib.patches as patches      # For drawing shapes


def wrap_angle(angle_rad):
    """Wraps an angle in radians to the interval [-pi, pi)."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi
# --- Plotting Function (Static Plots) ---
def plot_results(t, x, x_dot, theta_wrapped_rad, omega, x_dot_dot=None, omega_dot=None, F=None):
    """Generates the plot panel of the simulation results, using pre-wrapped angle."""
    theta_wrapped_deg = np.degrees(theta_wrapped_rad)
    num_plots = 4
    if x_dot_dot is not None: num_plots += 1
    if omega_dot is not None: num_plots += 1
    if F is not None: num_plots += 1
    rows = (num_plots + 1) // 2
    plt.figure("Simulation State Variables", figsize=(12, rows * 4))
    plot_idx = 1
    plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, x); plt.xlabel("Time (s)"); plt.ylabel("Cart Position (m)"); plt.title("Cart Position vs. Time"); plt.grid(True)
    plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, x_dot); plt.xlabel("Time (s)"); plt.ylabel("Cart Velocity (m/s)"); plt.title("Cart Velocity vs. Time"); plt.grid(True)
    plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, theta_wrapped_deg); plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angle (Wrapped, deg)\n(0=down, +/-180=up)"); plt.title("Pendulum Angle (Wrapped to [-180, 180]) vs. Time"); plt.ylim(-190, 190); plt.grid(True)
    plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, omega); plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angular Velocity (rad/s)"); plt.title("Pendulum Angular Velocity vs. Time"); plt.grid(True)
    if x_dot_dot is not None: plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, x_dot_dot); plt.xlabel("Time (s)"); plt.ylabel("Cart Acceleration (m/s^2)"); plt.title("Cart Acceleration vs. Time"); plt.grid(True)
    if omega_dot is not None: plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, omega_dot); plt.xlabel("Time (s)"); plt.ylabel("Pendulum Angular Accel (rad/s^2)"); plt.title("Pendulum Angular Acceleration vs. Time"); plt.grid(True)
    if F is not None: plt.subplot(rows, 2, plot_idx); plot_idx+=1; plt.plot(t, F); plt.xlabel("Time (s)"); plt.ylabel("Applied Force (N)"); plt.title("Applied Force vs. Time"); plt.grid(True)
    plt.tight_layout(); plt.suptitle("Inverted Pendulum Simulation Results (Forced)", y=1.02 if rows > 2 else 1.05); plt.show(block=False)

# --- Animation Function ---
def animate_pendulum(t_vec, x_vec, theta_vec_sim, anim_params, dt):
    PENDULUM_LENGTH = anim_params['l']; CART_WIDTH = anim_params['cart_w']; CART_HEIGHT = anim_params['cart_h']; WHEEL_RADIUS = anim_params['wheel_r']; GROUND_Y = 0.0
    theta_vec_anim = theta_vec_sim + np.pi
    num_frames = len(x_vec); INTERVAL_MS = max(10, int(dt * 1000))
    fig_anim, ax_anim = plt.subplots(num="Pendulum Animation", figsize=(10, 6))
    max_pendulum_reach_x = PENDULUM_LENGTH * 1.1; max_pendulum_reach_y = PENDULUM_LENGTH * 1.1
    min_x = np.min(x_vec) - CART_WIDTH / 2 - max_pendulum_reach_x; max_x = np.max(x_vec) + CART_WIDTH / 2 + max_pendulum_reach_x
    min_y = GROUND_Y - WHEEL_RADIUS * 2 - 0.5; max_y = GROUND_Y + WHEEL_RADIUS * 2 + CART_HEIGHT + max_pendulum_reach_y + 0.5
    ax_anim.set_xlim(min_x, max_x); ax_anim.set_ylim(min_y, max_y); ax_anim.set_aspect('equal', adjustable='box'); ax_anim.grid(True); ax_anim.set_xlabel("Cart Position (x)"); ax_anim.set_ylabel("Y Position"); ax_anim.set_title("Inverted Pendulum Animation")
    ground_line, = ax_anim.plot([min_x, max_x], [GROUND_Y, GROUND_Y], 'k-', lw=2)
    x0 = x_vec[0]; th0_anim = theta_vec_anim[0]
    cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2
    cart_rect = patches.Rectangle((x0 - CART_WIDTH / 2, cart_y_bottom), CART_WIDTH, CART_HEIGHT, fc='royalblue', ec='black'); ax_anim.add_patch(cart_rect)
    wheel_y = GROUND_Y + WHEEL_RADIUS; wheel_offset_x = CART_WIDTH / 4
    wheel1 = patches.Circle((x0 - wheel_offset_x, wheel_y), WHEEL_RADIUS, fc='gray', ec='black'); ax_anim.add_patch(wheel1)
    wheel2 = patches.Circle((x0 + wheel_offset_x, wheel_y), WHEEL_RADIUS, fc='gray', ec='black'); ax_anim.add_patch(wheel2)
    pivot_x = x0; pivot_y = cart_y_bottom + CART_HEIGHT
    pendulum_x_end = pivot_x + PENDULUM_LENGTH * np.sin(th0_anim); pendulum_y_end = pivot_y + PENDULUM_LENGTH * np.cos(th0_anim)
    pendulum_line, = ax_anim.plot([pivot_x, pendulum_x_end], [pivot_y, pendulum_y_end], 'r-', lw=3, solid_capstyle='round')
    pendulum_bob = patches.Circle((pendulum_x_end, pendulum_y_end), PENDULUM_LENGTH * 0.08, fc='darkred', ec='black'); ax_anim.add_patch(pendulum_bob)
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes, verticalalignment='top', fontsize=10)
    def update(frame):
        current_x = x_vec[frame]; current_theta_anim = theta_vec_anim[frame]
        cart_y_bottom = GROUND_Y + WHEEL_RADIUS * 2; cart_rect.set_x(current_x - CART_WIDTH / 2); cart_rect.set_y(cart_y_bottom)
        wheel_y = GROUND_Y + WHEEL_RADIUS; wheel1.center = (current_x - wheel_offset_x, wheel_y); wheel2.center = (current_x + wheel_offset_x, wheel_y)
        pivot_x = current_x; pivot_y = cart_y_bottom + CART_HEIGHT
        pendulum_x_end = pivot_x + PENDULUM_LENGTH * np.sin(current_theta_anim); pendulum_y_end = pivot_y + PENDULUM_LENGTH * np.cos(current_theta_anim)
        pendulum_line.set_data([pivot_x, pendulum_x_end], [pivot_y, pendulum_y_end]); pendulum_bob.center = (pendulum_x_end, pendulum_y_end)
        current_time = t_vec[frame]; time_text.set_text(f'Time: {current_time:.2f}s')
        return cart_rect, wheel1, wheel2, pendulum_line, pendulum_bob, time_text
    ani = animation.FuncAnimation(fig_anim, update, frames=num_frames, interval=INTERVAL_MS, blit=True, repeat=False)
    plt.show()

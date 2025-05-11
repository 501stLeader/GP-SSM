# %% Modifiziertes create_data.py
import numpy as np
from scipy.integrate import solve_ivp
import argparse
import os
try:
    import pandas as pd # For CSV saving
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
# Added for train-test split and PKL
from sklearn.model_selection import train_test_split
import pickle
from plot_util import plot_results, animate_pendulum, wrap_angle
from pendulum_dynamics import (
    generate_sinusoidal_force,
    inverse_pendulum_dynamics_forced,
    calculate_mass_matrix,
    calculate_gravity_vector
)

def save_mass_matrix_standalone(mass_matrix_data, base_filename_pkl, base_filename_csv):
    """Saves the mass matrix data to separate PKL and CSV files."""
    if mass_matrix_data is None:
        print(f"Warning: Mass matrix data is None. Skipping save for {base_filename_pkl} and {base_filename_csv}.")
        return

 
    # Save to CSV
    if PANDAS_AVAILABLE:
        if mass_matrix_data.ndim == 3 and mass_matrix_data.shape[1:] == (2, 2):
            try:
                df_data = {
                    'm11': mass_matrix_data[:, 0, 0],
                    'm12': mass_matrix_data[:, 0, 1],
                    'm21': mass_matrix_data[:, 1, 0],
                    'm22': mass_matrix_data[:, 1, 1]
                }
                df = pd.DataFrame(df_data)
                df.to_csv(base_filename_csv, index=False, float_format='%.6f')
                print(f"Mass matrix saved to '{base_filename_csv}'.")
                df.to_pickle(base_filename_pkl)
                print(f"Mass matrix saved to '{base_filename_pkl}'.")
            except Exception as e:
                print(f"Error saving mass matrix to CSV {base_filename_csv}: {e}")
        else:
            print(f"Warning: Unexpected mass matrix shape {mass_matrix_data.shape} for CSV saving. Skipping CSV save for {base_filename_csv}.")
    else:
        print(f"Warning: pandas not installed. Mass matrix CSV file '{base_filename_csv}' could not be saved.")
#    # Save to PKL
#     try:
#         with open(base_filename_pkl, 'wb') as f:
#             pickle.dump(mass_matrix_data, f)
#         print(f"Mass matrix saved to '{base_filename_pkl}'.")
#     except Exception as e:
#         print(f"Error saving mass matrix to PKL {base_filename_pkl}: {e}")

    
def save_gravity_vector_standalone(gravity_vector_data, base_filename_pkl, base_filename_csv):
    """Saves the gravity vector data to separate PKL and CSV files."""
    if gravity_vector_data is None:
        print(f"Warning: Gravity vector data is None. Skipping save for {base_filename_pkl} and {base_filename_csv}.")
        return

    # # Save to PKL
    # try:
    #     with open(base_filename_pkl, 'wb') as f:
    #         pickle.dump(gravity_vector_data, f)
    #     print(f"Gravity vector saved to '{base_filename_pkl}'.")
    # except Exception as e:
    #     print(f"Error saving gravity vector to PKL {base_filename_pkl}: {e}")

    # Save to CSV
    if PANDAS_AVAILABLE:
        if gravity_vector_data.ndim == 2 and gravity_vector_data.shape[1] == 2:
            try:
                df_data = {
                    'g1': gravity_vector_data[:, 0],
                    'g2': gravity_vector_data[:, 1]
                }
                df = pd.DataFrame(df_data)
                df.to_csv(base_filename_csv, index=False, float_format='%.6f')
                print(f"Gravity vector saved to '{base_filename_csv}'.")
                df.to_pickle(base_filename_pkl)
                print(f"Gravity vector saved to '{base_filename_pkl}'.")
            except Exception as e:
                print(f"Error saving gravity vector to CSV {base_filename_csv}: {e}")
        else:
            print(f"Warning: Unexpected gravity vector shape {gravity_vector_data.shape} for CSV saving. Skipping CSV save for {base_filename_csv}.")
    else:
        print(f"Warning: pandas not installed. Gravity vector CSV file '{base_filename_csv}' could not be saved.")


def prepare_csv_dict(t_subset, x_subset, x_dot_subset, theta_wrapped_rad_subset, omega_subset,
                     x_dot_dot_subset, omega_dot_subset, F_subset, tau_pendulum_subset):
    """Prepares a dictionary suitable for saving time-series data (excluding mass/gravity matrices) to CSV."""
    csv_subset_dict = {
        't': t_subset,
        'q_1': x_subset,                      # Cart position
        'dq_1': x_dot_subset,                 # Cart velocity
        'q_2': theta_wrapped_rad_subset,       # Pendulum angle (wrapped)
        'dq_2': omega_subset,                  # Pendulum angular velocity
        'ddq_1': x_dot_dot_subset,             # Cart acceleration
        'ddq_2': omega_dot_subset,             # Pendulum angular acceleration
        'output_1': F_subset,                 # Generalized force on x (cart force)
        'output_2': tau_pendulum_subset,      # Generalized force on theta (pendulum torque, usually 0)
    }
    return csv_subset_dict

# --- Main Function (Modified) ---
def main():
    parser = argparse.ArgumentParser(description="Simulate (with force), load, plot, animate an inverted pendulum, and save data splits.")
    parser.add_argument('--load', action='store_true', help='Load data instead of simulating.')
    parser.add_argument('--no-plot', action='store_true', help='Skip static plotting.')
    parser.add_argument('--animate', action='store_true', help='Show pendulum animation.')
    parser.add_argument('--save-dir', type=str, default='pendulum_data', help='Directory to save output files.')
    parser.add_argument('--base-filename', type=str, default='forced_pendulum_data', help='Base name for output files.')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (seconds).')
    parser.add_argument('--dt', type=float, default=0.02, help='Simulation time step (seconds).')
    parser.add_argument('--force-sines', type=int, default=5, help='Number of sinusoids in force profile.')
    parser.add_argument('--force-max-amp', type=float, default=10.0, help='Max total amplitude for force profile.')
    parser.add_argument('--force-max-freq', type=float, default=1.5, help='Max frequency (Hz) for force profile.')
    parser.add_argument('--init-theta-deg', type=float, default=0.0, help='Initial pendulum angle in degrees (0=down, 180=up).')
    parser.add_argument('--split', action='store_true', help='Perform train-test split (80/20) and save.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data for the test set.')


    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created save directory: {args.save_dir}")

    # Filenames
    full_npz_filename = os.path.join(args.save_dir, f"{args.base_filename}_full.npz")
    full_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_full.csv")

    # Filenames for Mass Matrix (PKL and CSV)
    full_mass_matrix_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_full_mass_matrix.pkl")
    full_mass_matrix_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_full_mass_matrix.csv")
    train_mass_matrix_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_train_mass_matrix.pkl")
    train_mass_matrix_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_train_mass_matrix.csv")
    test_mass_matrix_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_test_mass_matrix.pkl")
    test_mass_matrix_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_test_mass_matrix.csv")

    # Filenames for Gravity Vector (PKL and CSV)
    full_gravity_vector_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_full_gravity_vector.pkl")
    full_gravity_vector_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_full_gravity_vector.csv")
    train_gravity_vector_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_train_gravity_vector.pkl")
    train_gravity_vector_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_train_gravity_vector.csv")
    test_gravity_vector_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_test_gravity_vector.pkl")
    test_gravity_vector_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_test_gravity_vector.csv")

    train_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_train.pkl")
    test_pkl_filename = os.path.join(args.save_dir, f"{args.base_filename}_test.pkl")
    train_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_train.csv")
    test_csv_filename = os.path.join(args.save_dir, f"{args.base_filename}_test.csv")

    M_param = 2.0; m_param = 0.5; l_param = 0.6; g_const_param = 9.81
    anim_params = {'l': l_param, 'cart_w': l_param * 0.6, 'cart_h': l_param * 0.3, 'wheel_r': l_param * 0.08}

    theta_sim_wrapped = None
    t, x, x_dot, theta_sim, omega, x_dot_dot, omega_dot, F_applied, tau_pendulum_applied, mass_matrix_seq, gravity_vector_seq, dt_loaded_sim = [None]*12


    if args.load:
        print(f"Attempting to load data...")
        loaded_successfully = False
        if os.path.exists(full_npz_filename):
            try:
                with np.load(full_npz_filename, allow_pickle=True) as data:
                    t = data['t']
                    x = data['x']
                    x_dot = data['x_dot']
                    theta_sim = data['theta']
                    omega = data['omega']
                    x_dot_dot = data['x_dot_dot']
                    omega_dot = data['omega_dot']
                    F_applied = data['F']
                    dt_loaded_sim = data['dt'].item()
                    M_param = data['M'].item(); m_param = data['m'].item(); l_param = data['l'].item(); g_const_param = data['g'].item()
                    anim_params = {'l': l_param, 'cart_w': l_param * 0.6, 'cart_h': l_param * 0.3, 'wheel_r': l_param * 0.08}
                    if 'theta_wrapped' in data: theta_sim_wrapped = data['theta_wrapped']
                    else: theta_sim_wrapped = wrap_angle(theta_sim)
                    if 'tau_pendulum' in data: tau_pendulum_applied = data['tau_pendulum']
                    else: tau_pendulum_applied = np.zeros_like(t)
                print(f"Base data loaded from '{full_npz_filename}'.")
                loaded_successfully = True
            except Exception as e:
                print(f"Error loading base data from {full_npz_filename}: {e}")
        else:
            print(f"Error: Base data file '{full_npz_filename}' not found.")

        # Attempt to load mass matrix from its PKL file
        if os.path.exists(full_mass_matrix_pkl_filename):
            try:
                with open(full_mass_matrix_pkl_filename, 'rb') as f:
                    mass_matrix_seq = pickle.load(f)
                print(f"Loaded mass_matrix_seq from '{full_mass_matrix_pkl_filename}'.")
            except Exception as e:
                print(f"Error loading mass matrix from {full_mass_matrix_pkl_filename}: {e}. Will be None.")
                mass_matrix_seq = None
        else:
            print(f"Mass matrix file '{full_mass_matrix_pkl_filename}' not found. mass_matrix_seq will be None.")
            mass_matrix_seq = None

        # Attempt to load gravity vector from its PKL file
        if os.path.exists(full_gravity_vector_pkl_filename):
            try:
                with open(full_gravity_vector_pkl_filename, 'rb') as f:
                    gravity_vector_seq = pickle.load(f)
                print(f"Loaded gravity_vector_seq from '{full_gravity_vector_pkl_filename}'.")
            except Exception as e:
                print(f"Error loading gravity vector from {full_gravity_vector_pkl_filename}: {e}.")
                gravity_vector_seq = None # Fallback handled below
        else:
            print(f"Gravity vector file '{full_gravity_vector_pkl_filename}' not found.")
            gravity_vector_seq = None # Fallback handled below

        if loaded_successfully and gravity_vector_seq is None: # If base data loaded but gravity pkl failed/missing
            if all(v is not None for v in [theta_sim, m_param, l_param, g_const_param]):
                print("Calculating gravity vector from loaded data (theta_sim, m, l, g)...")
                num_points_loaded = len(theta_sim)
                gravity_vector_seq = np.zeros((num_points_loaded, 2))
                for i in range(num_points_loaded):
                    gravity_vector_seq[i] = calculate_gravity_vector(theta_sim[i], m_param, l_param, g_const_param)
            else:
                gravity_vector_seq = np.zeros((len(t), 2)) if t is not None else None
                print("WARNING: Could not calculate gravity_vector from loaded data. Initialized to zeros or None.")

        if not loaded_successfully:
             print("Failed to load essential data. Exiting.")
             return
        print(f"Data loading process complete. dt_loaded_sim = {dt_loaded_sim:.4f}s. Params: M={M_param:.2f}, m={m_param:.2f}, l={l_param:.2f}, g={g_const_param:.2f}")

    else: #
        print("Generating force profile...")
        force_profile = generate_sinusoidal_force(args.force_sines, args.force_max_amp, args.force_max_freq)
        print("Running simulation with external force...")
        theta_0_rad = np.radians(args.init_theta_deg)
        q0 = np.array([0.0, 0.0, theta_0_rad, 0.0]) # x, x_dot, theta, omega
        t_span = (0, args.duration)
        dt_loaded_sim = args.dt # dt_loaded_sim will store the simulation dt
        steps = int((t_span[1] - t_span[0]) / dt_loaded_sim) + 1
        t_eval = np.linspace(t_span[0], t_span[1], steps)

        def dynamics_for_solver(t_curr, q_curr):
            return inverse_pendulum_dynamics_forced(t_curr, q_curr, M_param, m_param, l_param, g_const_param, force_profile)

        sol = solve_ivp(dynamics_for_solver, t_span, y0=q0, method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8)

        if not sol.success: print(f"Warning: Solver failed! Message: {sol.message}")
        t = sol.t
        q_sol = sol.y.T
        if q_sol.shape[0] != len(t):
            print(f"Warning: Mismatch shapes. Truncating t and solution.")
            min_len = min(q_sol.shape[0], len(t))
            t = t[:min_len]; q_sol = q_sol[:min_len, :]; t_eval = t_eval[:min_len]

        x = q_sol[:, 0]; x_dot = q_sol[:, 1]; theta_sim = q_sol[:, 2]; omega = q_sol[:, 3]

        print("Calculating accelerations, forces, mass matrix, and gravity vector...")
        num_points = len(t)
        x_dot_dot = np.zeros(num_points); omega_dot = np.zeros(num_points)
        F_applied = np.zeros(num_points); tau_pendulum_applied = np.zeros(num_points)
        mass_matrix_seq = np.zeros((num_points, 2, 2)); gravity_vector_seq = np.zeros((num_points, 2))

        for i in range(num_points):
            derivs = inverse_pendulum_dynamics_forced(t[i], q_sol[i, :], M_param, m_param, l_param, g_const_param, force_profile)
            x_dot_dot[i] = derivs[1]; omega_dot[i] = derivs[3]
            F_applied[i] = force_profile(t[i])
            mass_matrix_seq[i] = calculate_mass_matrix(q_sol[i, 2], M_param, m_param, l_param)
            gravity_vector_seq[i] = calculate_gravity_vector(q_sol[i, 2], m_param, l_param, g_const_param)

        theta_sim_wrapped = wrap_angle(theta_sim)
        print("Simulation complete.")

        save_mass_matrix_standalone(mass_matrix_seq, full_mass_matrix_pkl_filename, full_mass_matrix_csv_filename)
        save_gravity_vector_standalone(gravity_vector_seq, full_gravity_vector_pkl_filename, full_gravity_vector_csv_filename)

        if PANDAS_AVAILABLE:
            try:
                # Main CSV does not include mass/gravity matrices anymore
                full_csv_dict = prepare_csv_dict(t, x, x_dot, theta_sim_wrapped, omega,
                                                 x_dot_dot, omega_dot, F_applied,
                                                 tau_pendulum_applied)
                df = pd.DataFrame(full_csv_dict)
                df.to_csv(full_csv_filename, index=False, float_format='%.6f')
                print(f"Full simulation data (excluding mass/gravity matrices) also saved to '{full_csv_filename}'.")
            except Exception as e:
                print(f"Error saving full data to CSV {full_csv_filename}: {e}")
        else:
            print("Warning: pandas not installed. Full CSV file (main data) could not be saved.")

    # Post-load/simulation checks
    if t is None or theta_sim_wrapped is None:
        print("Error: No data available (either from loading or simulation). Exiting.")
        return
    # Warnings for mass_matrix_seq or gravity_vector_seq being None will be handled by save/split functions

    if args.split:
        print(f"Performing train-test split ({1-args.test_size:.0%}/{args.test_size:.0%})...")
        # Ensure all necessary variables for splitting the main data are present
        # mass_matrix_seq and gravity_vector_seq are handled separately
        required_vars_for_split_main = {
            't': t, 'x': x, 'x_dot': x_dot, 'theta_sim': theta_sim,
            'theta_sim_wrapped': theta_sim_wrapped, 'omega': omega,
            'x_dot_dot': x_dot_dot, 'omega_dot': omega_dot,
            'F_applied': F_applied, 'tau_pendulum_applied': tau_pendulum_applied
        }
        missing_vars = [name for name, var in required_vars_for_split_main.items() if var is None]
        if missing_vars:
            print(f"Error: Cannot perform split. Essential variables for main data missing: {', '.join(missing_vars)}")
        else:
            try:
                indices = np.arange(t.shape[0])
                train_indices, test_indices = train_test_split(indices, test_size=args.test_size, shuffle=False)

                def prepare_split_dict_for_pkl(subset_indices):
                    data_dict = {
                        'q_1': x[subset_indices],
                        'q_2': theta_sim_wrapped[subset_indices],
                        'dq_1': x_dot[subset_indices],
                        'dq_2': omega[subset_indices],
                        'dqq_1': x_dot_dot[subset_indices],
                        'dqq_2': omega_dot[subset_indices],
                        'output_1': F_applied[subset_indices],
                        'output_2': tau_pendulum_applied[subset_indices],}
                    return data_dict

                # Training Data
                print("Processing training data...")
                train_data_pkl_main = prepare_split_dict_for_pkl(train_indices)
                try:
                    with open(train_pkl_filename, 'wb') as f: pickle.dump(train_data_pkl_main, f)
                    print(f"Training data (PKL, main) saved to '{train_pkl_filename}'")

                    # Save train mass matrix and gravity vector separately
                    if mass_matrix_seq is not None:
                        save_mass_matrix_standalone(mass_matrix_seq[train_indices], train_mass_matrix_pkl_filename, train_mass_matrix_csv_filename)
                    else:
                        print(f"Warning: mass_matrix_seq is None. Skipping save for train mass matrix files.")
                    if gravity_vector_seq is not None:
                        save_gravity_vector_standalone(gravity_vector_seq[train_indices], train_gravity_vector_pkl_filename, train_gravity_vector_csv_filename)
                    else:
                        print(f"Warning: gravity_vector_seq is None. Skipping save for train gravity vector files.")

                except Exception as e: print(f"Error saving training data (PKL main or separate matrices): {e}")

                if PANDAS_AVAILABLE:
                    try:
                        train_csv_dict_main = prepare_csv_dict(
                            t[train_indices], x[train_indices], x_dot[train_indices],
                            theta_sim_wrapped[train_indices], omega[train_indices],
                            x_dot_dot[train_indices], omega_dot[train_indices], F_applied[train_indices],
                            tau_pendulum_applied[train_indices]
                        )
                        df_train_main = pd.DataFrame(train_csv_dict_main)
                        df_train_main.to_csv(train_csv_filename, index=False, float_format='%.6f')
                        print(f"Training data (CSV, main) saved to '{train_csv_filename}'")
                    except Exception as e: print(f"Error saving training data to CSV (main) {train_csv_filename}: {e}")
                else: print("Warning: pandas not installed. Training CSV file (main) could not be saved.")


                print("Processing test data...")
                test_data_pkl_main = prepare_split_dict_for_pkl(test_indices)
                try:
                    with open(test_pkl_filename, 'wb') as f: pickle.dump(test_data_pkl_main, f)
                    print(f"Test data (PKL, main) saved to '{test_pkl_filename}'")

                    # Save test mass matrix and gravity vector separately
                    if mass_matrix_seq is not None:
                        save_mass_matrix_standalone(mass_matrix_seq[test_indices], test_mass_matrix_pkl_filename, test_mass_matrix_csv_filename)
                    else:
                        print(f"Warning: mass_matrix_seq is None. Skipping save for test mass matrix files.")
                    if gravity_vector_seq is not None:
                        save_gravity_vector_standalone(gravity_vector_seq[test_indices], test_gravity_vector_pkl_filename, test_gravity_vector_csv_filename)
                    else:
                        print(f"Warning: gravity_vector_seq is None. Skipping save for test gravity vector files.")

                except Exception as e: print(f"Error saving test data (PKL main or separate matrices): {e}")

                if PANDAS_AVAILABLE:
                    try:
                        test_csv_dict_main = prepare_csv_dict(
                            t[test_indices], x[test_indices], x_dot[test_indices],
                            theta_sim_wrapped[test_indices], omega[test_indices],
                            x_dot_dot[test_indices], omega_dot[test_indices], F_applied[test_indices],
                            tau_pendulum_applied[test_indices]
                        )
                        df_test_main = pd.DataFrame(test_csv_dict_main)
                        df_test_main.to_csv(test_csv_filename, index=False, float_format='%.6f')
                        print(f"Test data (CSV, main) saved to '{test_csv_filename}'")
                    except Exception as e: print(f"Error saving test data to CSV (main) {test_csv_filename}: {e}")
                else: print("Warning: pandas not installed. Test CSV file (main) could not be saved.")
            except Exception as e:
                print(f"Error during train-test split or saving process: {e}")


    if not args.no_plot:
        print("Generating static plots...")
        if all(v is not None for v in [t, x, x_dot, theta_sim_wrapped, omega]):
            plot_results(t, x, x_dot, theta_sim_wrapped, omega, x_dot_dot, omega_dot, F_applied)
            print("Static plot window generated.")
        else:
            print("Skipping static plots due to missing essential data (t, x, x_dot, theta_sim_wrapped, omega).")


    if args.animate:
        if dt_loaded_sim is None:
            print("Warning: dt for animation is unknown, defaulting to args.dt or 0.02s. Animation speed might be off.")
            anim_dt = args.dt if hasattr(args, 'dt') and args.dt is not None else 0.02
        else:
            anim_dt = dt_loaded_sim
        print("Generating animation...")
        if all(v is not None for v in [t, x, theta_sim]):
            animate_pendulum(t, x, theta_sim, anim_params, anim_dt)
            print("Animation window closed.")
        else:
            print("Skipping animation due to missing essential data (t, x, theta_sim).")

    print("Script finished.")

if __name__ == "__main__":
    main()

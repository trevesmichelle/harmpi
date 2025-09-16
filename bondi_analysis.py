import harm_script as hs  # Import harm_script to access its functions and variables
import matplotlib.pyplot as plt  # For plotting
import matplotlib.animation as animation
import numpy as np
import os
import argparse
from matplotlib.colors import LogNorm  # For LogNorm color scaling
from harm_script import plc


# Load grid and simulation data for multiple dump files
def load_data(grid_file, dump_file, cache={}):
    if "grid" not in cache:
        hs.rg(grid_file)
        cache["grid"] = True
        print(f"Loaded grid from {grid_file}.")
    hs.rd(dump_file)
    print(f"Loaded data from {dump_file}.")
    print(f"Simulation time: {hs.t:.6f}")


def analyze_bondi_acc(dump_files, dim=1, cache={}):
    sonic_surface_results = []
    all_radii = []
    all_rho = []
    all_times = []
    sonic_radii_over_time = []

    for dump_file in dump_files:
        try:
            load_data("gdump", dump_file, cache=cache)
            r = hs.r.squeeze()
            rho = hs.rho.squeeze()
            v1p = hs.v1p.squeeze()

            if dim == 2:
                rho = rho.mean(axis=1)
                r = r[:, 0]
                v1p = v1p.mean(axis=1)

            sonic_surface_index = (v1p[:-1] * v1p[1:] < 0).nonzero()[0]

            if len(sonic_surface_index) > 0:
                sonic_radius = r[sonic_surface_index[0]]
                sonic_surface_results.append((dump_file, float(sonic_radius)))
                print(f"Sonic surface detected at radius r = {float(sonic_radius):.6f} for {dump_file}")
            else:
                sonic_radius = None
                sonic_surface_results.append((dump_file, None))
                print(f"Sonic surface not detected for {dump_file}")

            all_radii.append(r)
            all_rho.append(rho)
            all_times.append(hs.t)
            sonic_radii_over_time.append(sonic_radius)

        except Exception as e:
            print(f"An error occurred while analyzing {dump_file}: {e}")

    if all_times:
        plot_density_profiles(all_radii, all_rho, all_times, sonic_surface_results, dim)

    if sonic_radii_over_time and all_times:
        plot_sonic_surface_evolution(all_times, sonic_radii_over_time)

    return sonic_surface_results


# Plot density profiles
def plot_density_profiles(all_radii, all_rho, all_times, sonic_surface_results, dim, savefig=False, filename=None, show=True):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    colormap = plt.cm.viridis
    norm = plt.Normalize(min(all_times), max(all_times))

    for i, (r, rho) in enumerate(zip(all_radii, all_rho)):
        color = colormap(norm(all_times[i]))
        ax.loglog(r, rho, label=f"t = {all_times[i]:.2f}", color=color)
        if sonic_surface_results[i][1] is not None:
            idx = np.abs(r - sonic_surface_results[i][1]).argmin()
            ax.plot(sonic_surface_results[i][1], rho[idx],
                    'o', color=color, markersize=6, label=f"Sonic Surface at t = {all_times[i]:.2f}")

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time')

    ax.set_xlabel("Radius (r)")
    ax.set_ylabel("Density (rho)")
    ax.set_title(f"Density Profiles with Sonic Surfaces (Time Evolution) - {dim}D")
    ax.grid(True)

    if savefig and filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")

    if show:
        plt.show()
    plt.close()


# Plot sonic surface evolution over time
def plot_sonic_surface_evolution(times, sonic_radii):
    plt.figure(figsize=(10, 6))
    plt.plot(times, sonic_radii, marker='o', linestyle='None', color='b', label="Sonic Radius")
    plt.xlabel("Time (t)")
    plt.ylabel("Sonic Radius (r)")
    plt.title("Sonic Surface Evolution Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()


# Compute Cartesian coordinates
def compute_cartesian_coordinates():
    r = hs.r.squeeze()
    theta = hs.h.squeeze()
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y


# Fixed animate_density_distribution function
def animate_density_distribution(grid_file, dump_files, output_file="bondi_density.mp4", fps=10):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Load first frame to set up the plot structure
    load_data(grid_file, dump_files[0])
    rho = hs.rho.squeeze()
    x, y = compute_cartesian_coordinates()
    rho_masked = np.ma.masked_less_equal(rho, 0)
    
    # Find global min/max across all frames for consistent scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Computing global density range...")
    for i, dump_file in enumerate(dump_files[::10]):  # Sample every 10th file for speed
        load_data(grid_file, dump_file)
        rho_temp = hs.rho.squeeze()
        rho_temp_masked = rho_temp[rho_temp > 0]  # Only positive values
        if len(rho_temp_masked) > 0:
            global_min = min(global_min, rho_temp_masked.min())
            global_max = max(global_max, rho_temp_masked.max())
    
    print(f"Global density range: {global_min:.2e} to {global_max:.2e}")
    
    # Create initial plot with global scaling
    pcm = ax.pcolormesh(
        x, y, rho_masked,
        shading='auto', cmap='viridis',
        norm=LogNorm(vmin=global_min, vmax=global_max)
    )
    cbar = fig.colorbar(pcm, ax=ax, label="Density (log scale)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    title = ax.set_title(f"2D Density Distribution at t = {hs.t:.3f}")

    def update(frame):
        dump_file = dump_files[frame]
        load_data(grid_file, dump_file)
        rho = hs.rho.squeeze()
        print(f"Frame {frame}: density min={rho.min():.2e}, max={rho.max():.2e}, time={hs.t:.3f}")
        
        rho_masked = np.ma.masked_less_equal(rho, 0)
        pcm.set_array(rho_masked.ravel())
        # Keep the global normalization - this ensures consistent color scaling
        title.set_text(f"2D Density Distribution at t = {hs.t:.3f}")
        return pcm, title

    ani = animation.FuncAnimation(fig, update, frames=len(dump_files), blit=False, interval=100)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    print(f"Saved density animation to {output_file}")
    plt.close(fig)

# Fixed animate_density_contours function
def animate_density_contours(grid_file, dump_files, levels=20, output_file="bondi_density_contours.mp4", fps=10):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find global density range for consistent scaling
    global_min = float('inf')
    global_max = float('-inf')
    
    print("Computing global density range for contours...")
    for dump_file in dump_files[::10]:  # Sample every 10th file
        load_data(grid_file, dump_file)
        rho = hs.rho.squeeze()
        positive_rho = rho[rho > 0]
        if len(positive_rho) > 0:
            global_min = min(global_min, positive_rho.min())
            global_max = max(global_max, positive_rho.max())
    
    # Set up coordinate system
    load_data(grid_file, dump_files[0])
    r, z = compute_cartesian_coordinates()
    
    # Create level values that will be consistent across frames
    log_levels = np.logspace(np.log10(global_min), np.log10(global_max), levels)
    
    def update(frame):
        ax.clear()  # Clear the entire axes for each frame
        
        dump_file = dump_files[frame]
        load_data(grid_file, dump_file)
        rho = hs.rho.squeeze()
        
        # Create contour plot with consistent levels
        contours = ax.contourf(r, z, rho, levels=log_levels, cmap='viridis', 
                              norm=LogNorm(vmin=global_min, vmax=global_max), extend='both')
        
        ax.set_xlabel("R")
        ax.set_ylabel("z")
        ax.set_title(f"Density Contours at t = {hs.t:.3f}")
        ax.set_aspect('equal')
        
        # return contours.collections
        return []  # Return empty list instead of contours.collections

    # Create initial plot for colorbar
    load_data(grid_file, dump_files[0])
    rho = hs.rho.squeeze()
    contours = ax.contourf(r, z, rho, levels=log_levels, cmap='viridis',
                          norm=LogNorm(vmin=global_min, vmax=global_max), extend='both')
    cbar = fig.colorbar(contours, ax=ax, label='Density (log scale)')

    ani = animation.FuncAnimation(fig, update, frames=len(dump_files), blit=False, interval=100)
    ani.save(output_file, writer='ffmpeg', fps=fps)
    print(f"Saved density contour animation to {output_file}")
    plt.close(fig)

# Enhanced 2D sonic radius analysis function
def analyze_2d_sonic_surface(dump_files, cache={}, debug=False):
    """
    Analyze sonic surface in 2D - track min, max, and average sonic radii
    Fixed to properly collect times from each dump file
    """
    sonic_analysis_results = []
    
    if debug:
        print("=== SONIC SURFACE ANALYSIS DEBUG ===")
    
    for i, dump_file in enumerate(dump_files):
        try:
            # Load data for this specific dump file
            load_data("gdump", dump_file, cache=cache)
            
            # NOW hs.t contains the correct time for this dump file
            current_time = float(hs.t)  # Capture the time immediately after loading
            
            if debug and i < 5:  # Debug first 5 files
                print(f"\n{dump_file}:")
                print(f"  Time: {current_time}")
                print(f"  Grid shape: r={hs.r.shape}, v1p={hs.v1p.shape}")
            
            r = hs.r.squeeze()
            v1p = hs.v1p.squeeze()
            
            # Find sonic surface for each theta slice
            sonic_radii_at_angles = []
            
            # Check middle slice for debugging
            if debug and i < 3:
                mid_j = r.shape[1] // 2
                r_slice = r[:, mid_j]
                v1p_slice = v1p[:, mid_j]
                print(f"  Mid-slice v1p range: {v1p_slice.min():.6f} to {v1p_slice.max():.6f}")
                sign_changes = np.where(v1p_slice[:-1] * v1p_slice[1:] < 0)[0]
                print(f"  Sign changes in mid-slice: {len(sign_changes)} at indices {sign_changes}")
            
            for j in range(r.shape[1]):  # Loop over theta direction
                r_slice = r[:, j]
                v1p_slice = v1p[:, j]
                
                # Find where v1p changes sign (sonic point)
                sign_changes = np.where(v1p_slice[:-1] * v1p_slice[1:] < 0)[0]
                
                if len(sign_changes) > 0:
                    # Take the first sign change (closest to BH)
                    sonic_idx = sign_changes[0]
                    sonic_r = r_slice[sonic_idx]
                    sonic_radii_at_angles.append(sonic_r)
            
            if len(sonic_radii_at_angles) > 0:
                min_sonic_r = min(sonic_radii_at_angles)
                max_sonic_r = max(sonic_radii_at_angles)
                avg_sonic_r = np.mean(sonic_radii_at_angles)
                
                result = {
                    'dump_file': dump_file,
                    'time': current_time,  # Use the captured time
                    'min_sonic_r': min_sonic_r,
                    'max_sonic_r': max_sonic_r,
                    'avg_sonic_r': avg_sonic_r,
                    'all_sonic_r': sonic_radii_at_angles
                }
                
                print(f"{dump_file}: t={current_time:.3f}, sonic r: min={min_sonic_r:.3f}, max={max_sonic_r:.3f}, avg={avg_sonic_r:.3f}")
            else:
                result = {
                    'dump_file': dump_file,
                    'time': current_time,  # Use the captured time
                    'min_sonic_r': None,
                    'max_sonic_r': None,
                    'avg_sonic_r': None,
                    'all_sonic_r': []
                }
                print(f"{dump_file}: t={current_time:.3f}, No sonic surface detected")
            
            sonic_analysis_results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {dump_file}: {e}")
    
    return sonic_analysis_results

# Plot 2D sonic radius evolution
def plot_2d_sonic_evolution(sonic_analysis_results):
    """
    Plot min, max, and average sonic radii over time
    Updated to work with the corrected data structure
    """
    # Extract data, filtering out None values
    valid_results = [r for r in sonic_analysis_results if r['avg_sonic_r'] is not None]
    
    if not valid_results:
        print("No valid sonic surface data found!")
        return
    
    times = [r['time'] for r in valid_results]
    min_radii = [r['min_sonic_r'] for r in valid_results]
    max_radii = [r['max_sonic_r'] for r in valid_results]
    avg_radii = [r['avg_sonic_r'] for r in valid_results]
    
    print(f"Plotting {len(times)} data points from t={min(times):.3f} to t={max(times):.3f}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, min_radii, 'b-', label='Min Sonic Radius', alpha=0.7)
    plt.plot(times, max_radii, 'r-', label='Max Sonic Radius', alpha=0.7)
    plt.plot(times, avg_radii, 'g-', label='Average Sonic Radius', linewidth=2)
    plt.fill_between(times, min_radii, max_radii, alpha=0.2, color='gray', label='Sonic Surface Range')
    plt.xlabel('Time')
    plt.ylabel('Sonic Radius')
    plt.title('2D Sonic Surface Evolution')
    plt.legend()
    plt.grid(True)
    
    # Plot sonic surface shape variation over time
    plt.subplot(2, 1, 2)
    shape_variation = [(max_r - min_r) / avg_r if avg_r > 0 else 0 
                      for min_r, max_r, avg_r in zip(min_radii, max_radii, avg_radii)]
    plt.plot(times, shape_variation, 'purple', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('(R_max - R_min) / R_avg')
    plt.title('Sonic Surface Shape Deviation from Spherical')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with theoretical values
    if avg_radii:
        final_avg = avg_radii[-1]
        print(f"Final average sonic radius: {final_avg:.3f}")
        
        # For Bondi accretion with γ=5/3, theoretical sonic radius is rs = 5GM/(4c²)
        # In HARM units where GM/c²=0.5 (typical), rs = 5/4 * 0.5 = 0.625
        # But this depends on your specific setup - check your Rin, Rout, and mass parameters
        print(f"Check your HARM parameters (a, Rin, Rout) to compare with theoretical sonic radius")

# Plot Sonic Surface with plc overlaying Mach contour #NOTE: is this function still relevant??
def plot_sonic_surface(grid_file, dump_file, savefig=False, filename=None, show=True):
    try:
        load_data(grid_file, dump_file)
        rho = hs.rho.squeeze()
        vx = hs.v1p.squeeze()
        vy = hs.v2p.squeeze()

        cs = np.sqrt(4.0 / 3.0 * (rho ** (4.0 / 3.0 - 1)))
        v = np.sqrt(vx**2 + vy**2)
        mach = v / cs

        plc(rho, xy=1, cb=1, isfilled=1, cmap="viridis")
        x, y = compute_cartesian_coordinates()
        plt.contour(x, y, mach, levels=[1], colors="red", linewidths=2, linestyles="--")
        plt.title(f"Sonic Surface (Mach=1) at t = {hs.t:.2f}")

        if savefig and filename:
            plt.savefig(filename)
            print(f"Sonic surface plot saved as {filename}")

        if show:
            plt.show()

    except Exception as e:
        print(f"Error in plot_sonic_surface with plc: {e}")


# Plot Velocity Fields (unchanged)
def plot_velocity_field(grid_file, dump_file, savefig=False, filename=None, show=True):
    try:
        load_data(grid_file, dump_file)
        rho = hs.rho.squeeze()
        x, y = compute_cartesian_coordinates()
        vx = hs.v1p.squeeze()
        vy = hs.v2p.squeeze()

        v_magnitude = np.sqrt(vx**2 + vy**2)

        skip = (slice(None, None, 5), slice(None, None, 5))
        x_downsampled, y_downsampled = x[skip], y[skip]
        vx_downsampled, vy_downsampled = vx[skip], vy[skip]
        v_magnitude_downsampled = v_magnitude[skip]

        plt.clf()
        q = plt.quiver(
            x_downsampled, y_downsampled, vx_downsampled, vy_downsampled,
            v_magnitude_downsampled, cmap='viridis', scale=20, headwidth=5, alpha=0.8
        )
        cbar = plt.colorbar(q)
        cbar.set_label("Velocity Magnitude")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Velocity Field at t = {hs.t:.2f}")

        if savefig and filename:
            plt.savefig(filename)
            print(f"Velocity field plot saved as {filename}")

        if show:
            plt.show()

    except Exception as e:
        print(f"An error occurred while plotting velocity field for {dump_file}: {e}")

# Fixed main execution for 2D analysis
def main_2d_analysis_corrected(dump_files, args):
    """
    Corrected main analysis function for 2D Bondi problem
    """
    cache = {}
    
    print("Step 1: Analyzing sonic surfaces with corrected time collection...")
    sonic_analysis_2d = analyze_2d_sonic_surface(dump_files, cache=cache, debug=True)
    
    # Verify we have different times
    times = [result['time'] for result in sonic_analysis_2d if result['time'] is not None]
    print(f"\nTime range: {min(times):.3f} to {max(times):.3f}")
    print(f"Number of unique times: {len(set(times))}")
    
    # Plot sonic evolution with the corrected data
    print("Step 2: Plotting sonic surface evolution...")
    plot_2d_sonic_evolution(sonic_analysis_2d)
    
    # Fixed animations
    animate_density_distribution("gdump", dump_files,
                                output_file=os.path.join(args.output, "bondi_density.mp4"), fps=10)
    animate_density_contours("gdump", dump_files, levels=30,
                            output_file=os.path.join(args.output, "bondi_density_contours.mp4"), fps=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Bondi Accretion")
    parser.add_argument("--dim", type=int, default=2, help="Simulation dimension (1 for 1D, 2 for 2D)")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing them")
    parser.add_argument("--output", type=str, default="./bondi_plots", help="Output folder for saved plots")
    args = parser.parse_args()

    dump_folder = "/home/michelle/harmpi/dumps"
    all_dump_files = sorted([f for f in os.listdir(dump_folder) if f.startswith("dump")])
    selected_dump_files = ["dump021", "dump042", "dump069", "dump107", "dump216", "dump283", "dump311", "dump415", "dump459", "dump501"]

    if args.save and not os.path.exists(args.output):
        os.makedirs(args.output)

    cache = {}

    if args.dim == 1:
        sonic_surface_results_1D = analyze_bondi_acc(selected_dump_files, dim=1, cache=cache)
        r1d_list = []
        rho1d_list = []
        time1d = []
        for dump_file in selected_dump_files:
            load_data("gdump", dump_file, cache=cache)
            r1d_list.append(hs.r.squeeze())
            rho1d_list.append(hs.rho.squeeze())
            time1d.append(hs.t)
        plot_density_profiles(r1d_list, rho1d_list, time1d, sonic_surface_results_1D, dim=1)
        plot_sonic_surface_evolution(time1d, [r for _, r in sonic_surface_results_1D])

    else:
        # Original analysis
        # sonic_surface_results_2D = analyze_bondi_acc(all_dump_files, dim=2, cache=cache)
        # Plot sonic surface evolution
        # times_2d = [hs.t for _ in all_dump_files]  # this created a bug
        # sonic_radii_2d = [r for _, r in sonic_surface_results_2D]
        # plot_sonic_surface_evolution(times_2d, sonic_radii_2d)

        main_2d_analysis_corrected(all_dump_files, args)



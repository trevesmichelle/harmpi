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


# Original density animation function (unchanged from old version)
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


# Original contour animation function (unchanged from old version)
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
        
        return []

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
    FIXED: Now properly collects times from each dump file
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
            upstream_sonic_r = None
            downstream_sonic_r = None
            
            # Check middle slice for debugging
            if debug and i < 3:
                mid_j = r.shape[1] // 2
                r_slice = r[:, mid_j]
                v1p_slice = v1p[:, mid_j]
                print(f"  Mid-slice v1p range: {v1p_slice.min():.6f} to {v1p_slice.max():.6f}")
                sign_changes = np.where(v1p_slice[:-1] * v1p_slice[1:] < 0)[0]
                print(f"  Sign changes in mid-slice: {len(sign_changes)} at indices {sign_changes}")
            
            n_theta = r.shape[1]
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
                    
                    # Add upstream/downstream tracking
                    theta_fraction = j / n_theta
                    if theta_fraction < 0.25:  # "Upstream" region
                        if upstream_sonic_r is None or sonic_r < upstream_sonic_r:
                            upstream_sonic_r = sonic_r
                    elif theta_fraction > 0.75:  # "Downstream" region  
                        if downstream_sonic_r is None or sonic_r > downstream_sonic_r:
                            downstream_sonic_r = sonic_r
            
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
                    'upstream_sonic_r': upstream_sonic_r,
                    'downstream_sonic_r': downstream_sonic_r,
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
                    'upstream_sonic_r': None,
                    'downstream_sonic_r': None,
                    'all_sonic_r': []
                }
                print(f"{dump_file}: t={current_time:.3f}, No sonic surface detected")
            
            sonic_analysis_results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {dump_file}: {e}")
    
    return sonic_analysis_results


# Plot 2D sonic radius evolution
def plot_2d_sonic_evolution(sonic_analysis_results, title_suffix=""):
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
    
    # Add upstream/downstream if available
    upstream_radii = [r['upstream_sonic_r'] for r in valid_results if r['upstream_sonic_r'] is not None]
    downstream_radii = [r['downstream_sonic_r'] for r in valid_results if r['downstream_sonic_r'] is not None]
    upstream_times = [r['time'] for r in valid_results if r['upstream_sonic_r'] is not None]
    downstream_times = [r['time'] for r in valid_results if r['downstream_sonic_r'] is not None]
    
    if upstream_radii:
        plt.plot(upstream_times, upstream_radii, 'b--', alpha=0.8, linewidth=2, label='Upstream Sonic Radius')
    if downstream_radii:
        plt.plot(downstream_times, downstream_radii, 'r--', alpha=0.8, linewidth=2, label='Downstream Sonic Radius')
    
    plt.xlabel('Time')
    plt.ylabel('Sonic Radius')
    plt.title(f'2D Sonic Surface Evolution {title_suffix}')
    plt.legend()
    plt.grid(True)
    
    # Plot sonic surface shape variation over time
    plt.subplot(2, 1, 2)
    shape_variation = [(max_r - min_r) / avg_r if avg_r > 0 else 0 
                      for min_r, max_r, avg_r in zip(min_radii, max_radii, avg_radii)]
    plt.plot(times, shape_variation, 'purple', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('(R_max - R_min) / R_avg')
    plt.title(f'Sonic Surface Shape Deviation from Spherical {title_suffix}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Plot Sonic Surface with plc overlaying Mach contour
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


# Wind velocity detection function
def detect_wind_velocity(dump_files, cache={}, sample_size=5):
    """
    Detect if simulation has significant uniform velocity (wind)
    """
    total_vy = 0
    count = 0
    
    # Sample a few files to detect wind
    sample_files = dump_files[:sample_size] if len(dump_files) >= sample_size else dump_files
    
    for dump_file in sample_files:
        try:
            load_data("gdump", dump_file, cache=cache)
            if hasattr(hs, 'vu') and len(hs.vu) > 2:
                vy = hs.vu[2].squeeze()
                total_vy += np.abs(vy).mean()
                count += 1
        except:
            continue
    
    avg_vy = total_vy / count if count > 0 else 0
    has_wind = avg_vy > 0.01  # Threshold for significant wind
    
    print(f"Average |v_theta|: {avg_vy:.4f} -> {'Wind detected' if has_wind else 'No wind (pure Bondi)'}")
    return has_wind


# Original velocity field animation with density overlay (from old version)
def animate_density_with_velocity(grid_file, dump_files, output_file="bondi_density_velocity.mp4", fps=10, skip_factor=3):
    """
    Create animation showing density contours with velocity field overlay
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Find global ranges for consistent scaling
    print("Computing global ranges for density+velocity animation...")
    global_rho_min, global_rho_max = float('inf'), float('-inf')
    global_v_max = 0
    
    sample_files = dump_files[::max(1, len(dump_files)//20)]
    for dump_file in sample_files:
        try:
            load_data(grid_file, dump_file)
            rho = hs.rho.squeeze()
            
            # Use vu if available, otherwise fall back to v1p, v2p
            if hasattr(hs, 'vu') and len(hs.vu) > 2:
                vx, vy = hs.vu[1].squeeze(), hs.vu[2].squeeze()
            else:
                vx, vy = hs.v1p.squeeze(), hs.v2p.squeeze()
            
            positive_rho = rho[rho > 0]
            if len(positive_rho) > 0:
                global_rho_min = min(global_rho_min, positive_rho.min())
                global_rho_max = max(global_rho_max, positive_rho.max())
            
            v_mag = np.sqrt(vx**2 + vy**2)
            global_v_max = max(global_v_max, v_mag.max())
        except Exception as e:
            print(f"Warning: Could not process {dump_file} for range calculation: {e}")
    
    print(f"Density range: {global_rho_min:.2e} to {global_rho_max:.2e}")
    print(f"Max velocity: {global_v_max:.3f}")
    
    # Set up coordinate system
    load_data(grid_file, dump_files[0])
    x, y = compute_cartesian_coordinates()
    
    # Create density contour levels
    rho_levels = np.logspace(np.log10(global_rho_min), np.log10(global_rho_max), 25)
    
    def update(frame):
        ax.clear()
        
        dump_file = dump_files[frame]
        load_data(grid_file, dump_file)
        
        rho = hs.rho.squeeze()
        
        # Use available velocity data
        if hasattr(hs, 'vu') and len(hs.vu) > 2:
            vx, vy = hs.vu[1].squeeze(), hs.vu[2].squeeze()
        else:
            vx, vy = hs.v1p.squeeze(), hs.v2p.squeeze()
        
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Density contour plot
        try:
            contours = ax.contourf(x, y, rho, levels=rho_levels, cmap='viridis',
                                  norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max),
                                  alpha=0.7, extend='both')
        except:
            # Fallback if LogNorm fails
            contours = ax.contourf(x, y, rho, levels=25, cmap='viridis', alpha=0.7)
        
        # Velocity field overlay (downsampled)
        skip = (slice(None, None, skip_factor), slice(None, None, skip_factor))
        x_skip, y_skip = x[skip], y[skip]
        vx_skip, vy_skip = vx[skip], vy[skip]
        v_mag_skip = v_mag[skip]
        
        # Normalize arrows for better visibility
        v_norm = np.sqrt(vx_skip**2 + vy_skip**2)
        v_norm = np.where(v_norm > 0, v_norm, 1)  # Avoid division by zero
        
        # Only draw arrows where velocity is significant
        significant_v = v_mag_skip > 0.01 * global_v_max
        if np.any(significant_v):
            quiver = ax.quiver(x_skip[significant_v], y_skip[significant_v], 
                              (vx_skip/v_norm)[significant_v], (vy_skip/v_norm)[significant_v],
                              v_mag_skip[significant_v], cmap='plasma', scale=30, headwidth=3,
                              alpha=0.8, width=0.003)
        
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Z", fontsize=14)
        ax.set_title(f"Density + Velocity Field at t = {hs.t:.3f}", fontsize=16)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        max_extent = min(50, max(x.max(), y.max()) if x.size > 0 and y.size > 0 else 50)
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        
        return []
    
    # Create colorbar using first frame
    load_data(grid_file, dump_files[0])
    rho = hs.rho.squeeze()
    
    try:
        contours = ax.contourf(x, y, rho, levels=rho_levels, cmap='viridis',
                              norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max),
                              alpha=0.7, extend='both')
        cbar1 = fig.colorbar(contours, ax=ax, shrink=0.8, pad=0.15)
        cbar1.set_label('Density (log scale)', fontsize=12)
    except Exception as e:
        print(f"Warning: Colorbar creation failed: {e}")
    
    ani = animation.FuncAnimation(fig, update, frames=len(dump_files),
                                 blit=False, interval=100, repeat=True)
    
    try:
        ani.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
        print(f"Saved density+velocity animation to {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Make sure ffmpeg is installed: sudo apt install ffmpeg")
    
    plt.close(fig)


# Enhanced sonic surface analysis with wind effects (from old version)
def analyze_sonic_surface_with_wind(dump_files, cache={}, debug=False):
    """
    Enhanced sonic surface analysis that also tracks flow asymmetry
    FIXED: Now properly collects times from each dump file
    """
    results = []
    
    for i, dump_file in enumerate(dump_files):
        try:
            load_data("gdump", dump_file, cache=cache)
            current_time = float(hs.t)  # FIXED: Capture time immediately after loading
            
            r = hs.r.squeeze()
            v1p = hs.v1p.squeeze()  # Radial velocity (lab frame)
            
            # Analyze sonic surface shape and location
            sonic_radii_at_angles = []
            upstream_sonic_r = None
            downstream_sonic_r = None
            
            n_theta = r.shape[1]
            for j in range(n_theta):
                r_slice = r[:, j]
                v1p_slice = v1p[:, j]
                
                # Find sonic points (where v1p changes sign)
                sign_changes = np.where(v1p_slice[:-1] * v1p_slice[1:] < 0)[0]
                
                if len(sign_changes) > 0:
                    sonic_r = r_slice[sign_changes[0]]
                    sonic_radii_at_angles.append(sonic_r)
                    
                    # Track upstream vs downstream (simplified geometry)
                    theta_fraction = j / n_theta
                    if theta_fraction < 0.25:  # "Upstream" region
                        if upstream_sonic_r is None or sonic_r < upstream_sonic_r:
                            upstream_sonic_r = sonic_r
                    elif theta_fraction > 0.75:  # "Downstream" region  
                        if downstream_sonic_r is None or sonic_r > downstream_sonic_r:
                            downstream_sonic_r = sonic_r
            
            # Flow asymmetry metrics
            if len(sonic_radii_at_angles) > 0:
                min_sonic_r = min(sonic_radii_at_angles)
                max_sonic_r = max(sonic_radii_at_angles)
                avg_sonic_r = np.mean(sonic_radii_at_angles)
                asymmetry = (max_sonic_r - min_sonic_r) / avg_sonic_r if avg_sonic_r > 0 else 0
                
                result = {
                    'dump_file': dump_file,
                    'time': current_time,
                    'min_sonic_r': min_sonic_r,
                    'max_sonic_r': max_sonic_r,
                    'avg_sonic_r': avg_sonic_r,
                    'asymmetry': asymmetry,
                    'upstream_sonic_r': upstream_sonic_r,
                    'downstream_sonic_r': downstream_sonic_r,
                    'all_sonic_r': sonic_radii_at_angles
                }
                
                if debug and i < 10:
                    print(f"{dump_file}: t={current_time:.3f}, avg_sonic_r={avg_sonic_r:.3f}, asymmetry={asymmetry:.3f}")
            else:
                result = {
                    'dump_file': dump_file,
                    'time': current_time,
                    'min_sonic_r': None,
                    'max_sonic_r': None,
                    'avg_sonic_r': None,
                    'asymmetry': 0,
                    'upstream_sonic_r': None,
                    'downstream_sonic_r': None,
                    'all_sonic_r': []
                }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {dump_file}: {e}")
    
    return results


# Enhanced plotting for wind effects (from old version)
def plot_wind_effects(sonic_results, title_suffix="", show_asymmetry=True):
    """
    Plot sonic surface evolution with wind effects
    """
    valid_results = [r for r in sonic_results if r['avg_sonic_r'] is not None]
    
    if not valid_results:
        print("No valid sonic surface data!")
        return
    
    times = [r['time'] for r in valid_results]
    avg_radii = [r['avg_sonic_r'] for r in valid_results]
    
    if show_asymmetry:
        asymmetries = [r['asymmetry'] for r in valid_results]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot 1: Sonic radius evolution
    ax1.plot(times, avg_radii, 'g-', linewidth=2, label='Average Sonic Radius')
    
    # Add upstream/downstream if available
    upstream_times, downstream_times = [], []
    upstream_radii, downstream_radii = [], []
    
    for r in valid_results:
        if r['upstream_sonic_r'] is not None:
            upstream_times.append(r['time'])
            upstream_radii.append(r['upstream_sonic_r'])
        if r['downstream_sonic_r'] is not None:
            downstream_times.append(r['time'])
            downstream_radii.append(r['downstream_sonic_r'])
    
    if upstream_radii:
        ax1.plot(upstream_times, upstream_radii, 'b--', alpha=0.7, label='Upstream Sonic Radius')
    if downstream_radii:
        ax1.plot(downstream_times, downstream_radii, 'r--', alpha=0.7, label='Downstream Sonic Radius')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Sonic Radius')
    ax1.set_title(f'Sonic Surface Evolution {title_suffix}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Flow asymmetry (if requested)
    if show_asymmetry:
        ax2.plot(times, asymmetries, 'purple', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Flow Asymmetry (R_max - R_min)/R_avg')
        ax2.set_title(f'Sonic Surface Asymmetry {title_suffix}')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


# Enhanced scenario detection (from old version)
def detect_flow_regime(dump_files, cache={}, sample_size=5, force_regime=None):
    """
    Detect what type of Bondi modification we're dealing with
    IMPROVED: Multi-criteria detection to avoid misclassification
    
    Args:
        force_regime: Optional manual override ("random_velocity", "density_gradient", "wind", "angular_momentum")
    """
    # Manual override if specified
    if force_regime:
        regime_map = {
            "random_velocity": "Bondi with Random Velocity",
            "density_gradient": "Bondi with Density Gradient", 
            "wind": "Bondi-Hoyle-Lyttleton (Wind)",
            "angular_momentum": "Bondi with Angular Momentum"
        }
        if force_regime.lower() in regime_map:
            forced_regime = regime_map[force_regime.lower()]
            print(f"MANUAL OVERRIDE: Forcing regime to {forced_regime}")
            return forced_regime, {'manual_override': True}
    
    total_vy = 0  # Wind velocity
    total_angular = 0  # Angular momentum
    velocity_variance = 0  # Random velocity
    density_gradient = 0  # Density variation
    velocity_spatial_variance = 0  # Spatial variation in velocity field
    radial_density_profile_strength = 0  # Strength of radial density profile
    count = 0
    
    sample_files = dump_files[:sample_size]
    
    for dump_file in sample_files:
        try:
            load_data("gdump", dump_file, cache=cache)
            
            # Check for uniform wind
            if hasattr(hs, 'vu') and len(hs.vu) > 2:
                vy = hs.vu[2].squeeze()
                total_vy += np.abs(vy).mean()
            
            # Check for angular momentum - FIXED
            if hasattr(hs, 'vu') and len(hs.vu) > 3:
                vphi = hs.vu[3].squeeze()
                total_angular += np.abs(vphi).mean()
            elif hasattr(hs, 'up'):  # Alternative way to access phi velocity
                up = hs.up.squeeze()
                total_angular += np.abs(up).mean()
            
            # Check velocity field randomness (temporal)
            vr = hs.vu[1].squeeze() if hasattr(hs, 'vu') else hs.v1p.squeeze()
            velocity_variance += np.var(vr)
            
            # NEW: Check spatial velocity variation (key for random velocity)
            if hasattr(hs, 'vu') and len(hs.vu) > 2:
                vx, vy = hs.vu[1].squeeze(), hs.vu[2].squeeze()
            else:
                vx, vy = hs.v1p.squeeze(), hs.v2p.squeeze()
            
            # Measure spatial variation in velocity field
            vx_spatial_var = np.var(vx)
            vy_spatial_var = np.var(vy)
            velocity_spatial_variance += (vx_spatial_var + vy_spatial_var)
            
            # Check density gradient - but also check if it's primarily radial
            rho = hs.rho.squeeze()
            r = hs.r.squeeze()
            
            # Standard density gradient check
            rho_profile = rho.mean(axis=1) if rho.ndim > 1 else rho
            r_profile = r[:, 0] if r.ndim > 1 else r
            if len(r_profile) > 10:
                density_gradient += np.abs(np.gradient(np.log(rho_profile), np.log(r_profile))).mean()
            
            # NEW: Check if density profile is primarily radial (true density gradient)
            # vs random/turbulent (random velocity case)
            if rho.ndim > 1:
                # Compare radial profile smoothness vs angular variation
                rho_angular_var = np.var(rho, axis=1).mean()  # Variation in theta direction
                rho_radial_var = np.var(rho_profile)  # Variation in radial direction
                if rho_radial_var > 0:
                    radial_density_profile_strength += rho_radial_var / (rho_radial_var + rho_angular_var)
                
            count += 1
        except:
            continue
    
    if count == 0:
        return "Unknown", {}
    
    # Average the metrics
    avg_vy = total_vy / count
    avg_angular = total_angular / count
    avg_variance = velocity_variance / count
    avg_gradient = density_gradient / count
    avg_spatial_var = velocity_spatial_variance / count
    avg_radial_profile = radial_density_profile_strength / count
    
    # IMPROVED: Multi-criteria classification with priority logic
    regime_type = "Pure Bondi"
    characteristics = {}
    
    print(f"Detection values:")
    print(f"  gradient={avg_gradient:.3f}, radial_profile={avg_radial_profile:.3f}")
    print(f"  wind={avg_vy:.6f}, angular={avg_angular:.6f}")
    print(f"  velocity_variance={avg_variance:.3f}, spatial_variance={avg_spatial_var:.3f}")
    
    # Random velocity: Adjusted thresholds based on actual simulation data
    # Check for combination of wind + measurable spatial variation
    if avg_spatial_var > 0.005 and avg_variance > 0.0005 and avg_vy > 0.01:
        regime_type = "Bondi with Random Velocity"
        characteristics['velocity_variance'] = avg_variance
        characteristics['spatial_variance'] = avg_spatial_var
        characteristics['wind_component'] = avg_vy  # Also record the baseline wind
    # Density gradient: High gradient + strong radial profile + low spatial velocity variation
    elif avg_gradient > 1.0 and avg_radial_profile > 0.7 and avg_spatial_var < 0.02:
        regime_type = "Bondi with Density Gradient"
        characteristics['density_gradient_strength'] = avg_gradient
        characteristics['radial_profile_strength'] = avg_radial_profile
        characteristics['wind_component'] = avg_vy  # Also record wind component
    # BHL wind: High uniform wind WITHOUT significant turbulence
    elif avg_vy > 0.01 and avg_spatial_var < 0.02:
        regime_type = "Bondi-Hoyle-Lyttleton (Wind)"
        characteristics['wind_velocity'] = avg_vy
    # Angular momentum: Significant angular velocity
    elif avg_angular > 0.01:
        regime_type = "Bondi with Angular Momentum"
        characteristics['angular_velocity'] = avg_angular
    # Fallback: If wind detected but criteria unclear
    elif avg_vy > 0.01:
        regime_type = "Mixed Wind/Turbulent Flow"
        characteristics['wind_velocity'] = avg_vy
        characteristics['spatial_variance'] = avg_spatial_var
    
    print(f"Flow regime detected: {regime_type}")
    for key, value in characteristics.items():
        print(f"  {key}: {value:.4f}")
    
    return regime_type, characteristics


# Fixed main execution for 2D analysis (from old version but with time fix)
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
    
    # Fixed animations - USE ALL DUMP FILES, no subsampling
    animate_density_distribution("gdump", dump_files,
                                output_file=os.path.join(args.output, "bondi_density.mp4"), fps=10)
    animate_density_contours("gdump", dump_files, levels=30,
                            output_file=os.path.join(args.output, "bondi_density_contours.mp4"), fps=10)


# Modified main execution for enhanced 2D analysis (from old version but with time fix)
def main_2d_analysis_enhanced(dump_files, args):
    """
    Enhanced main analysis function that detects flow regime and adapts accordingly
    """
    cache = {}
    
    print("=== ENHANCED 2D BONDI ANALYSIS ===")
    
    # Debug: Check what data we actually have
    print(f"Total dump files found: {len(dump_files)}")
    print(f"First 5 files: {dump_files[:5]}")
    print(f"Last 5 files: {dump_files[-5:]}")
    
    # Check time range
    load_data("gdump", dump_files[0], cache=cache)
    t_start = hs.t
    load_data("gdump", dump_files[-1], cache=cache)  
    t_end = hs.t
    print(f"Time range: {t_start:.1f} to {t_end:.1f}")
    
    # Step 1: Enhanced regime detection with manual override option
    regime_type, characteristics = detect_flow_regime(dump_files, cache=cache, force_regime="random_velocity")
    
    # Create safe filename from regime type
    safe_regime_name = regime_type.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    
    print(f"Analysis type: {regime_type}")
    
    # Step 2: Run appropriate sonic analysis based on detected regime
    if "Wind" in regime_type or "Angular Momentum" in regime_type:
        print("Running enhanced sonic analysis with asymmetry tracking...")
        sonic_results = analyze_sonic_surface_with_wind(dump_files, cache=cache, debug=True)
        plot_wind_effects(sonic_results, title_suffix=f"({regime_type})", show_asymmetry=True)
        create_velocity_animation = True  # These regimes benefit from velocity visualization
    elif "Random Velocity" in regime_type:
        print("Running enhanced sonic analysis for turbulent flow...")
        sonic_results = analyze_sonic_surface_with_wind(dump_files, cache=cache, debug=True)
        plot_wind_effects(sonic_results, title_suffix=f"({regime_type})", show_asymmetry=True)
        create_velocity_animation = True  # Show velocity structure
    elif "Density Gradient" in regime_type:
        print("Running standard sonic analysis with density gradient...")
        sonic_results = analyze_2d_sonic_surface(dump_files, cache=cache, debug=True)
        plot_2d_sonic_evolution(sonic_results, title_suffix=f"({regime_type})")
        create_velocity_animation = True  # Enable velocity animation for density gradient
    else:
        print("Running standard 2D sonic analysis...")
        sonic_results = analyze_2d_sonic_surface(dump_files, cache=cache, debug=True)
        plot_2d_sonic_evolution(sonic_results, title_suffix=f"({regime_type})")
        create_velocity_animation = False  # Pure Bondi case
    
    # Step 3: Create appropriate animations - USE ALL DUMP FILES
    print("Creating animations...")
    
    print(f"Creating animations with {len(dump_files)} frames covering t≈{t_start:.1f} to t≈{t_end:.1f}")
    
    # Always create density animation
    animate_density_distribution("gdump", dump_files,
                                output_file=os.path.join(args.output, f"bondi_density_{safe_regime_name}.mp4"), 
                                fps=12)
    
    # Create density+velocity animation for regimes with interesting velocity fields
    if create_velocity_animation:
        animate_density_with_velocity("gdump", dump_files,
                                     output_file=os.path.join(args.output, f"bondi_density_velocity_{safe_regime_name}.mp4"),
                                     fps=12, skip_factor=3)
        print(f"Created velocity field animation for {regime_type}")
    
    # Create contour animation
    animate_density_contours("gdump", dump_files,
                            levels=25,
                            output_file=os.path.join(args.output, f"bondi_contours_{safe_regime_name}.mp4"), 
                            fps=10)
    
    # Print summary of what was detected and analyzed
    print(f"\n=== {regime_type.upper()} ANALYSIS COMPLETE ===")
    if characteristics:
        print("Detected characteristics:")
        for key, value in characteristics.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"Generated animations:")
    print(f"  - Density: bondi_density_{safe_regime_name}.mp4")
    if create_velocity_animation:
        print(f"  - Density+Velocity: bondi_density_velocity_{safe_regime_name}.mp4")
    print(f"  - Contours: bondi_contours_{safe_regime_name}.mp4")


# Updated main block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Bondi Accretion")
    parser.add_argument("--dim", type=int, default=2, help="Simulation dimension (1 for 1D, 2 for 2D)")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing them")
    parser.add_argument("--output", type=str, default="./bondi_plots", help="Output folder for saved plots")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced analysis with automatic wind detection")
    args = parser.parse_args()

    dump_folder = "/home/michelle/harmpi/dumps"
    all_dump_files = sorted([f for f in os.listdir(dump_folder) if f.startswith("dump")])
    selected_dump_files = ["dump021", "dump042", "dump069", "dump107", "dump216", "dump283", "dump311", "dump415", "dump459", "dump501"]

    # Always ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

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
        if args.enhanced:
            # Use the new enhanced analysis
            main_2d_analysis_enhanced(all_dump_files, args)
        else:
            # Use the original corrected analysis
            main_2d_analysis_corrected(all_dump_files, args)
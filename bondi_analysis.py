import harm_script as hs  # Import harm_script to access its functions and variables
import matplotlib.pyplot as plt  # For plotting
import numpy as np
import os

# Step 1: Load grid and simulation data for multiple dump files
def load_data(grid_file, dump_file):
    """ 
    Loads the grid and dump file using harm_script. 
    Args:
        grid_file (str): Name of the grid file (e.g., "gdump").
        dump_file (str): Name of the dump file (e.g., "dump021").
    """
    hs.rg(grid_file)  # Read grid information
    hs.rd(dump_file)  # Read dump file (populates variables like t, rho, etc.)
    print(f"Loaded grid from {grid_file} and data from {dump_file}.")
    print(f"Simulation time: {hs.t:.6f}")

# Step 2: Analyze Bondi accretion for multiple dump files
def analyze_bondi_acc(dump_files, dim=1):
    """ 
    Analyzes Bondi accretion for the given dump files. 
    Args:
        dump_files (list): List of dump file names to analyze.
        dim (int): Dimension of the simulation (1 for 1D, 2 for 2D).
    """
    sonic_surface_results = []  # To store results for sonic surface
    all_radii = []  # To store all radius values for plotting
    all_rho = []  # To store all density values for plotting
    all_times = []  # To store all times for plotting
    sonic_radii_over_time = []  # To track sonic radius evolution

    for dump_file in dump_files:
        try:
            load_data("gdump", dump_file)  # Load grid and data for each dump file

            r = hs.r.squeeze()  # Radial coordinates
            rho = hs.rho.squeeze()  # Density profile
            v1p = hs.v1p.squeeze()  # Outgoing wave velocity

            if dim == 2:
                # For 2D, consider analyzing along a specific axis or average over angles
                rho = rho.mean(axis=1)  # Average density over theta
                r = r[:, 0]  # Radial values (1D slice)
                v1p = v1p.mean(axis=1)  # Average velocity over theta

            # Find the sonic surface (where v1p changes sign)
            sonic_surface_index = (v1p[:-1] * v1p[1:] < 0).nonzero()[0]  # Sign change detection

            sonic_radius = None  # Initialize sonic_radius
            if len(sonic_surface_index) > 0:
                sonic_radius = r[sonic_surface_index[0]]  # First crossing
                sonic_surface_results.append((dump_file, float(sonic_radius)))
                print(f"Sonic surface detected at radius r = {float(sonic_radius):.6f} for {dump_file}")
            else:
                sonic_surface_results.append((dump_file, None))
                print(f"Sonic surface not detected for {dump_file}")

            # Add data to lists for plotting
            all_radii.append(r)
            all_rho.append(rho)
            all_times.append(hs.t)  # Store time for labeling
            sonic_radii_over_time.append(sonic_radius)

        except Exception as e:
            print(f"An error occurred while analyzing {dump_file}: {e}")

    # Plot all density profiles
    if all_times:
        plot_density_profiles(all_radii, all_rho, all_times, sonic_surface_results, dim)

    # Plot sonic surface evolution over time
    if sonic_radii_over_time and all_times:
        plot_sonic_surface_evolution(all_times, sonic_radii_over_time)

    return sonic_surface_results

# Step 3: Plot density profiles
def plot_density_profiles(all_radii, all_rho, all_times, sonic_surface_results, dim):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get the current axes
    colormap = plt.cm.viridis  # Choose a colormap (e.g., viridis, plasma, etc.)
    norm = plt.Normalize(min(all_times), max(all_times))  # Normalize times for coloring

    for i, (r, rho) in enumerate(zip(all_radii, all_rho)):
        color = colormap(norm(all_times[i]))  # Map time to a color
        ax.plot(r, rho, label=f"t = {all_times[i]:.2f}", color=color)  # Add label with time
            
        # If sonic surface exists for this file, plot it
        if sonic_surface_results[i][1] is not None:
            ax.axvline(sonic_surface_results[i][1], color='r', linestyle='--', 
                        label=f"Sonic Surface at r = {sonic_surface_results[i][1]:.2f}")

    # Add a colorbar to show the time scale
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Required for the colorbar to work
    cbar = plt.colorbar(sm, ax=ax)  # Add the colorbar to the current axes
    cbar.set_label('Time')

    # Add labels, title, and legend
    ax.set_xlabel("Radius (r)")
    ax.set_ylabel("Density (rho)")
    ax.set_title(f"Density Profiles with Sonic Surfaces (Time Evolution) - {dim}D")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()

# Step 4: Plot sonic surface evolution over time
def plot_sonic_surface_evolution(times, sonic_radii):
    plt.figure(figsize=(10, 6))
    plt.plot(times, sonic_radii, marker='o', linestyle='None', color='b', label="Sonic Radius")
    plt.xlabel("Time (t)")
    plt.ylabel("Sonic Radius (r)")
    plt.title("Sonic Surface Evolution Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

# Step 5: Helper function to compute Cartesian coordinates
def compute_cartesian_coordinates():
    """
    Computes the Cartesian grid (x, y) from the spherical grid.
    Returns:
        x (ndarray): X-coordinates of the grid.
        y (ndarray): Y-coordinates of the grid.
    """
    r = hs.r.squeeze()  # Radial coordinate
    theta = hs.h.squeeze()  # Polar angle

    x = r * np.sin(theta)  # X-coordinate
    y = r * np.cos(theta)  # Y-coordinate

    return x, y

# Step 6: Plot 2D Density Distribution
def plot_density_distribution(grid_file, dump_file):
    """
    Plots the 2D density distribution.
    Args:
        grid_file (str): Name of the grid file (e.g., "gdump").
        dump_file (str): Name of the dump file (e.g., "dump021").
    """
    try:
        load_data(grid_file, dump_file)  # Load grid and data
        rho = hs.rho.squeeze()  # Density field
        x, y = compute_cartesian_coordinates()  # Compute Cartesian coordinates

        # Create a 2D density plot
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, y, rho, shading="auto", cmap="viridis")
        plt.colorbar(label="Density")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"2D Density Distribution at t = {hs.t:.2f}")
        # plt.savefig(f"density_distribution_{dump_file}.png")
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting density distribution for {dump_file}: {e}")


# Step 7: Plot 2D Velocity Field
def plot_velocity_field(grid_file, dump_file):
    """
    Plots the 2D velocity field over the density distribution.
    Args:
        grid_file (str): Name of the grid file (e.g., "gdump").
        dump_file (str): Name of the dump file (e.g., "dump021").
    """
    try:
        load_data(grid_file, dump_file)  # Load grid and data
        rho = hs.rho.squeeze()  # Density field
        x, y = compute_cartesian_coordinates()  # Compute Cartesian coordinates
        vx = hs.v1p.squeeze()  # X-velocity
        vy = hs.v2p.squeeze()  # Y-velocity

        # Create a density plot with velocity quivers
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, y, rho, shading="auto", cmap="viridis", alpha=0.8)
        plt.colorbar(label="Density")
        plt.quiver(x, y, vx, vy, color="white", scale=20, alpha=0.7)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Velocity Field at t = {hs.t:.2f}")
        # plt.savefig(f"velocity_field_{dump_file}.png")
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting velocity field for {dump_file}: {e}")


# Step 8: Plot Sonic Surface in 2D
def plot_sonic_surface(grid_file, dump_file):
    """
    Plots the sonic surface (Mach = 1 contour) over the 2D density distribution.
    Args:
        grid_file (str): Name of the grid file (e.g., "gdump").
        dump_file (str): Name of the dump file (e.g., "dump021").
    """
    try:
        load_data(grid_file, dump_file)  # Load grid and data
        rho = hs.rho.squeeze()  # Density field
        x, y = compute_cartesian_coordinates()  # Compute Cartesian coordinates
        vx = hs.v1p.squeeze()  # X-velocity
        vy = hs.v2p.squeeze()  # Y-velocity

        # Calculate Mach number
        cs = np.sqrt(4.0 / 3.0 * (rho ** (4.0 / 3.0 - 1)))  # Sound speed
        v = np.sqrt(vx**2 + vy**2)  # Velocity magnitude
        mach = v / cs  # Mach number

        # Create a density plot with sonic surface contour
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, y, rho, shading="auto", cmap="viridis", alpha=0.8)
        plt.colorbar(label="Density")
        plt.contour(x, y, mach, levels=[1], colors="red", linewidths=2, linestyles="--")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Sonic Surface (Mach=1) at t = {hs.t:.2f}")
        # plt.savefig(f"sonic_surface_{dump_file}.png")
        plt.show()

    except Exception as e:
        print(f"An error occurred while plotting sonic surface for {dump_file}: {e}")


# Step 9: Main function enhancements
if __name__ == "__main__":
    dump_folder = "/home/michelle/harmpi/dumps"
    all_dump_files = [f for f in os.listdir(dump_folder) if f.startswith("dump")]
    selected_dump_files = ["dump021", "dump042", "dump069", "dump107", "dump216", "dump283", "dump311", "dump415", "dump459", "dump501"]

    # Analyze for 2D simulation
    dim = 2  # Change to 1 for 1D simulation
    sonic_surface_results_2D = analyze_bondi_acc(selected_dump_files, dim=dim)

    # Print out results for sonic surface detection
    print("\nSonic Surface Results:")
    for dump_file, sonic_radius in sonic_surface_results_2D:
        if sonic_radius is not None:
            print(f"{dump_file}: Sonic surface at r = {sonic_radius:.6f}")
        else:
            print(f"{dump_file}: No sonic surface detected.")

    # Plotting for a single dump file
    for dump_file in selected_dump_files:
        plot_density_distribution("gdump", dump_file)
        plot_velocity_field("gdump", dump_file)
        plot_sonic_surface("gdump", dump_file)

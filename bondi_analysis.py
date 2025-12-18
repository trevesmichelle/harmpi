import harm_script as hs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/cluster use
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import argparse
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import glob

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
GAMMA = 5./3.           # Adiabatic index
R_IN = 2.0              # Initial boundary (r_g) - FIXED from 10.0
R_SONIC_THEORY_BONDI = 5.0    # Theoretical sonic radius for pure Bondi (γ=5/3)

# BHL theoretical parameters (from init.c)
V_Z_WIND = 0.1          # Wind velocity
C_S_INF = np.sqrt(0.2)  # Sound speed at infinity
R_ACC_BHL = 1.0 / (C_S_INF**2 + V_Z_WIND**2)  # BHL accretion radius ≈ 4.76
R_SONIC_THEORY = R_SONIC_THEORY_BONDI  # Default to Bondi value 

# =============================================================================
# BONDI ANALYSIS CLASS
# =============================================================================

class BondiAnalysis:
    """
    Comprehensive Bondi accretion analysis class.
    Handles 1D and 2D scenarios with proper coordinate transformations.
    
    CRITICAL: This class expects to be run from ~/harmpi directory with
    a 'dumps' symlink pointing to the data folder.
    """
    def __init__(self, output_dir="./bondi_plots"):
        self.output_dir = output_dir
        self.cache = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, grid_file, dump_file):
        """
        Load grid and dump data.
        
        CRITICAL: Pass BARE filenames only! harm_script automatically 
        prepends "dumps/" to all paths.
        
        Args:
            grid_file: Usually "gdump" (NOT "dumps/gdump")
            dump_file: e.g. "dump000" (NOT "dumps/dump000")
        """
        # Load grid only once
        if "grid" not in self.cache:
            hs.rg(grid_file)  # harm_script looks for dumps/gdump
            self.cache["grid"] = True
        
        # Load dump file
        hs.rd(dump_file)  # harm_script looks for dumps/dump000
        return hs.t


    def mirror_domain(self, r, theta, data):
        """
        Mirror the axisymmetric data to create full circle visualization.
        
        Parameters:
        -----------
        r : array
            Radial coordinate (nx, ny)
        theta : array
            Theta coordinate (nx, ny)  
        data : array
            Data to mirror (nx, ny)
            
        Returns:
        --------
        x_full, z_full, data_full : mirrored arrays
        """
        x = r * np.sin(theta)
        z = r * np.cos(theta)
        
        # Mirror across the midplane
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        data_full = np.concatenate([data[:, ::-1], data], axis=1)
        
        return x_full, z_full, data_full

    # =========================================================================
    # 1D ANALYSIS & DOCUMENTATION
    # =========================================================================
    
    def analyze_1d_sonic_surface(self, dump_files):
        """
        Analyze sonic surface in 1D with full documentation.
        
        Returns:
            list: Results with time, r_sonic, mach_number for each dump
        """
        results = []
        
        for dump_file in dump_files:
            t = self.load_data("gdump", dump_file)
            
            r = hs.r.squeeze()
            v1p = hs.v1p.squeeze()
            
            # Calculate sound speed (documented)
            P = hs.pg.squeeze()
            rho = hs.rho.squeeze()
            c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
            
            # Method: Find where Mach ≈ 1
            mach = np.abs(v1p) / c_s
            sonic_idx = np.argmin(np.abs(mach - 1.0))
            
            results.append({
                'time': t,
                'r_sonic': float(r[sonic_idx]),
                'mach': float(mach[sonic_idx]),
                'dump': dump_file
            })
        
        return results
    
    def plot_1d_density_profiles(self, dump_files, sonic_results):
        """
        Plot density profiles with initial conditions marked.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Color map
        times = [r['time'] for r in sonic_results]
        norm = plt.Normalize(min(times), max(times))
        cmap = plt.cm.viridis
        
        for i, (dump_file, result) in enumerate(zip(dump_files, sonic_results)):
            self.load_data("gdump", dump_file)
            r = hs.r.squeeze()
            rho = hs.rho.squeeze()
            
            color = cmap(norm(result['time']))
            ax.loglog(r, rho, alpha=0.7, color=color)
            
            # Mark sonic point
            if result['r_sonic']:
                ax.axvline(result['r_sonic'], color=color, alpha=0.3, linestyle='--')
        
        # Mark initial boundary
        ax.axvline(R_IN, color='red', linestyle=':', linewidth=2, 
                  label=f'Initial boundary (r={R_IN})')
        ax.axvline(R_SONIC_THEORY, color='orange', linestyle=':', linewidth=2,
                  label=f'Theory sonic radius (r ≈ {R_SONIC_THEORY})')
        
        ax.set_xlabel('Radius (r_g)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('1D Density Profiles Evolution', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Time')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "density_1d_profiles.png"), dpi=150)
        plt.close()
    
    def plot_1d_sonic_evolution(self, sonic_results):
        """
        Plot how sonic radius evolves over time.
        """
        times = [r['time'] for r in sonic_results]
        r_sonic = [r['r_sonic'] for r in sonic_results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, r_sonic, 'b-', linewidth=2, label='Sonic radius')
        ax.axhline(R_SONIC_THEORY, color='orange', linestyle='--', 
                  label=f'Theory (γ=5/3): r ≈ {R_SONIC_THEORY}')
        
        ax.set_xlabel('Time (t/tg)', fontsize=12)
        ax.set_ylabel('Sonic Radius (r_g)', fontsize=12)
        ax.set_title('Sonic Surface Evolution (1D)', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sonic_evolution_1d.png"), dpi=150)
        plt.close()
    
    # =========================================================================
    # 2D ANALYSIS
    # =========================================================================
    
    def analyze_2d_sonic_surface(self, dump_files):
        """
        Analyze 2D sonic surface tracking min, max, avg radii, 
        plus upstream and downstream radii for BHL scenarios.
        Now includes proper handling for angular momentum (v_phi).
        """
        results = []
        
        for dump_file in dump_files:
            self.load_data("gdump", dump_file)
            
            r = hs.r.squeeze()
            v1p = hs.v1p.squeeze()
            
            # Check if we have azimuthal velocity (angular momentum case)
            has_vphi = hasattr(hs, 'v3p') and hs.v3p is not None
            if has_vphi:
                v3p = hs.v3p.squeeze()
            
            # Calculate sound speed for Mach number
            P = hs.pg.squeeze()
            rho = hs.rho.squeeze()
            c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
            
            # Find sonic point at each angle
            sonic_radii = []
            upstream_sonic_r = None
            downstream_sonic_r = None
            n_theta = r.shape[1]
            
            for j in range(n_theta):
                r_slice = r[:, j]
                v1p_slice = v1p[:, j]
                c_s_slice = c_s[:, j]
                
                if has_vphi:
                    # Angular momentum case: use total Mach number
                    v3p_slice = v3p[:, j]
                    v_total = np.sqrt(v1p_slice**2 + v3p_slice**2)
                    mach_slice = v_total / c_s_slice
                    
                    # Find where Mach crosses 1
                    mach_diff = mach_slice - 1.0
                    sign_changes = np.where(mach_diff[:-1] * mach_diff[1:] < 0)[0]
                else:
                    # Pure Bondi/BHL case: use radial velocity sign change
                    sign_changes = np.where(v1p_slice[:-1] * v1p_slice[1:] < 0)[0]
                
                if len(sign_changes) > 0:
                    sonic_r = r_slice[sign_changes[0]]
                    sonic_radii.append(sonic_r)
                    
                    # Track upstream/downstream sonic radii
                    theta_fraction = j / n_theta
                    if theta_fraction < 0.25:  # "Upstream" region (θ ~ 0, +z)
                        if upstream_sonic_r is None or sonic_r < upstream_sonic_r:
                            upstream_sonic_r = sonic_r
                    elif theta_fraction > 0.75:  # "Downstream" region (θ ~ π, -z)
                        if downstream_sonic_r is None or sonic_r > downstream_sonic_r:
                            downstream_sonic_r = sonic_r
            
            if len(sonic_radii) > 0:
                results.append({
                    'time': float(hs.t),
                    'min_r': float(np.min(sonic_radii)),
                    'max_r': float(np.max(sonic_radii)),
                    'avg_r': float(np.mean(sonic_radii)),
                    'upstream_r': float(upstream_sonic_r) if upstream_sonic_r is not None else None,
                    'downstream_r': float(downstream_sonic_r) if downstream_sonic_r is not None else None,
                    'dump': dump_file
                })
            else:
                results.append({
                    'time': float(hs.t),
                    'min_r': None,
                    'max_r': None,
                    'avg_r': None,
                    'upstream_r': None,
                    'downstream_r': None,
                    'dump': dump_file
                })
        
        return results
    
    def plot_2d_sonic_evolution(self, sonic_results, scenario=""):
        """
        Plot sonic surface statistics over time, including upstream/downstream
        tracking for BHL scenarios.
        """
        times = [r['time'] for r in sonic_results if r['avg_r'] is not None]
        min_r = [r['min_r'] for r in sonic_results if r['avg_r'] is not None]
        max_r = [r['max_r'] for r in sonic_results if r['avg_r'] is not None]
        avg_r = [r['avg_r'] for r in sonic_results if r['avg_r'] is not None]
        
        # Extract upstream/downstream data if available
        upstream_times = [r['time'] for r in sonic_results if r.get('upstream_r') is not None]
        upstream_r = [r['upstream_r'] for r in sonic_results if r.get('upstream_r') is not None]
        downstream_times = [r['time'] for r in sonic_results if r.get('downstream_r') is not None]
        downstream_r = [r['downstream_r'] for r in sonic_results if r.get('downstream_r') is not None]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Main curves
        ax.fill_between(times, min_r, max_r, alpha=0.2, color='gray', label='Min-Max Range')
        ax.plot(times, avg_r, 'b-', linewidth=2, label='Average Sonic Radius')
        
        # Upstream/downstream curves (for BHL scenarios)
        if upstream_r and len(upstream_r) > 0:
            ax.plot(upstream_times, upstream_r, 'c--', linewidth=2, alpha=0.8,
                   label='Upstream (θ~0, +z)', marker='o', markersize=3)
        if downstream_r and len(downstream_r) > 0:
            ax.plot(downstream_times, downstream_r, 'm--', linewidth=2, alpha=0.8,
                   label='Downstream (θ~π, -z)', marker='s', markersize=3)
        
        # Theory reference - scenario-dependent
        if "Bondi-Hoyle-Lyttleton" in scenario or "BHL" in scenario:
            # For BHL, plot the accretion radius as reference
            ax.axhline(R_ACC_BHL, color='orange', linestyle=':', linewidth=2,
                      label=f'BHL accretion radius: r≈{R_ACC_BHL:.1f}')
            # Add note about asymmetry
            ax.text(0.02, 0.98, 'Note: BHL sonic surface is highly asymmetric\n' +
                   'Upstream: typically no sonic point (supersonic flow)\n' +
                   'Downstream: sonic point at bow shock',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # For pure Bondi, use the standard sonic radius
            ax.axhline(R_SONIC_THEORY_BONDI, color='orange', linestyle=':', linewidth=2,
                      label=f'Theory (γ=5/3): r≈{R_SONIC_THEORY_BONDI:.1f}')
        
        ax.set_xlabel('Time (t/tg)', fontsize=12)
        ax.set_ylabel('Sonic Radius (r_g)', fontsize=12)
        title = 'Sonic Surface Evolution (2D)'
        if scenario:
            title += f' - {scenario}'
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"sonic_evolution_2d_{scenario.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    
    def plot_2d_sonic_surface_map(self, dump_file, scenario=""):
        """
        Create 2D map with sonic surface contour.
        Shows full circle with mirrored domain.
        Now supports angular momentum scenarios.
        """
        self.load_data("gdump", dump_file)
        
        # Check if we have azimuthal velocity (angular momentum case)
        has_vphi = hasattr(hs, 'v3p') and hs.v3p is not None
        
        # Calculate velocity components
        v1p = hs.v1p.squeeze()  # Lab frame radial velocity
        u_r = (hs.uu[1] / hs.uu[0]).squeeze()
        u_theta = (hs.uu[2] / hs.uu[0]).squeeze()
        
        # Calculate velocity magnitude (include v_phi if present)
        if has_vphi:
            v_phi = hs.v3p.squeeze()
            v_mag = np.sqrt(u_r**2 + u_theta**2 + v_phi**2)
        else:
            v_mag = np.sqrt(u_r**2 + u_theta**2)
        
        # Calculate sound speed and Mach number
        P = hs.pg.squeeze()
        rho = hs.rho.squeeze()
        c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
        mach = v_mag / c_s
        
        # Cartesian coordinates
        r = hs.r.squeeze()
        theta = hs.h.squeeze()
        
        # Mirror domain for full circle visualization
        x_full, z_full, v_mag_full = self.mirror_domain(r, theta, v_mag)
        _, _, mach_full = self.mirror_domain(r, theta, mach)
        _, _, v1p_full = self.mirror_domain(r, theta, v1p)
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # VELOCITY MAGNITUDE background (makes physical sense with sonic surface!)
        v_mag_pos = v_mag_full.copy()
        v_mag_pos[~np.isfinite(v_mag_pos)] = np.nan
        
        # Use percentiles for better color distribution
        vmin_v = np.nanpercentile(v_mag_pos, 5)
        vmax_v = np.nanpercentile(v_mag_pos, 95)
        
        im = ax.pcolormesh(x_full, z_full, v_mag_pos,
                        vmin=vmin_v, vmax=vmax_v,
                        cmap='plasma', alpha=0.8, shading='auto')
        cbar = plt.colorbar(im, ax=ax, label='Velocity Magnitude (c)', pad=0.02)
        
        # SONIC SURFACE DETECTION
        # For angular momentum: use Mach=1
        # For pure Bondi/BHL: use v1p=0 (proven method)
        
        # DIAGNOSTIC: Check ranges
        v1p_min, v1p_max = np.nanmin(v1p_full), np.nanmax(v1p_full)
        v1p_near_0 = np.sum((v1p_full > -0.1) & (v1p_full < 0.1))
        v_mag_min, v_mag_max = np.nanmin(v_mag_full), np.nanmax(v_mag_full)
        mach_min, mach_max = np.nanmin(mach_full), np.nanmax(mach_full)
        mach_near_1 = np.sum((mach_full > 0.9) & (mach_full < 1.1))
        
        print(f"\n=== SONIC SURFACE PLOT DIAGNOSTICS ===")
        print(f"Scenario: {scenario}")
        print(f"Angular momentum (v_phi): {'YES' if has_vphi else 'NO'}")
        print(f"Background: Velocity magnitude")
        print(f"  Velocity range: [{v_mag_min:.3f}, {v_mag_max:.3f}] c")
        print(f"  Mach range: [{mach_min:.3f}, {mach_max:.3f}]")
        print(f"  Points with 0.9 < Mach < 1.1: {mach_near_1}")
        
        if has_vphi:
            print(f"Sonic surface detection: Mach = 1")
            contour_field = mach_full
            contour_level = 1.0
            contour_label = 'SONIC (M=1)'
        else:
            print(f"Sonic surface detection: v1p = 0")
            print(f"  v1p range: [{v1p_min:.3f}, {v1p_max:.3f}]")
            print(f"  Points with -0.1 < v1p < 0.1: {v1p_near_0}")
            print(f"  v1p=0 exists: {np.any((v1p_full > -0.01) & (v1p_full < 0.01))}")
            contour_field = v1p_full
            contour_level = 0.0
            contour_label = 'SONIC'
        
        # Layer 1: Thick black base for contrast
        try:
            cs_base = ax.contour(x_full, z_full, contour_field, levels=[contour_level],
                            colors='black', linewidths=10, alpha=0.6, zorder=3)
            print(f"✓ Black base contour ({contour_label}) plotted successfully")
        except Exception as e:
            print(f"✗ Black contour failed: {e}")
        
        # Layer 2: Medium red contour
        try:
            cs_red = ax.contour(x_full, z_full, contour_field, levels=[contour_level],
                            colors='red', linewidths=6, alpha=1.0, zorder=4)
            print(f"✓ Red contour ({contour_label}) plotted successfully")
        except Exception as e:
            print(f"✗ Red contour failed: {e}")
        
        # Layer 3: Thin white highlight
        try:
            cs_white = ax.contour(x_full, z_full, contour_field, levels=[contour_level],
                                colors='white', linewidths=2, alpha=0.8, zorder=5)
            # Add label
            ax.clabel(cs_white, inline=True, fontsize=11, fmt=contour_label, 
                    manual=[(0, np.max(z_full)*0.6)])
            print(f"✓ White contour ({contour_label}) plotted successfully")
        except Exception as e:
            print(f"✗ White contour failed: {e}")
        print("=================================\n")
        
        # Directional indicator - ONLY for asymmetric scenarios
        if "Bondi-Hoyle-Lyttleton" in scenario or "BHL" in scenario.upper():
            # Add simple arrow showing wind direction
            arrow_props = dict(arrowstyle='->', lw=3, color='cyan', alpha=0.8)
            ax.annotate('', xy=(0, np.max(z_full)*0.75), xytext=(0, np.max(z_full)*0.95),
                    arrowprops=arrow_props)
            ax.text(0, np.max(z_full)*0.97, 'Wind', ha='center', fontsize=10,
                weight='bold', color='cyan',
                bbox=dict(boxstyle='round', fc='black', alpha=0.7, pad=0.3))
        
        # Add subsonic/supersonic region labels
        ax.text(0.02, 0.98, 'Subsonic\n(v < c_s)', 
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.98, 0.02, 'Supersonic\n(v > c_s)', 
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Horizon
        r_h = 2.0
        horizon = plt.Circle((0, 0), r_h, color='black', fill=True)
        ax.add_patch(horizon)
        
        ax.set_xlabel('x (r_g)', fontsize=12)
        ax.set_ylabel('z (r_g)', fontsize=12)
        title = f'Sonic Surface Map (t={hs.t:.1f})'
        if scenario:
            title += f' - {scenario}'
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        filename = f"sonic_map_2d_{scenario.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()
    # =========================================================================
    # VELOCITY FIELD ANALYSIS
    # =========================================================================
    
    def verify_velocity_transformation(self, dump_file):
        """
        Verify velocity transformation from spherical to Cartesian.
        Tests the init.c implementation.
        """
        self.load_data("gdump", dump_file)
        
        u_r = hs.uu[1] / hs.uu[0]
        u_theta = hs.uu[2] / hs.uu[0]
        theta = hs.h.squeeze()
        
        # Transform to Cartesian
        v_x = u_r * np.sin(theta) + u_theta * np.cos(theta)
        v_z = u_r * np.cos(theta) - u_theta * np.sin(theta)
        
        stats = {
            'v_z_mean': float(np.mean(v_z)),
            'v_z_std': float(np.std(v_z)),
            'v_x_mean': float(np.mean(v_x)),
            'v_x_std': float(np.std(v_x))
        }
        
        print(f"\nVelocity Statistics:")
        print(f"  v_z: mean={stats['v_z_mean']:.4f}, std={stats['v_z_std']:.4f}")
        print(f"  v_x: mean={stats['v_x_mean']:.4f}, std={stats['v_x_std']:.4f}")
        
        return stats
    
    def plot_velocity_field(self, dump_file, scenario=""):
        """
        Plot velocity field with LARGE VISIBLE arrows (magnetized_analysis_v3 pattern).
        Shows full circle with mirrored domain.
        """
        self.load_data("gdump", dump_file)
        
        # Transform to Cartesian
        u_r = (hs.uu[1] / hs.uu[0]).squeeze()
        u_theta = (hs.uu[2] / hs.uu[0]).squeeze()
        r = hs.r.squeeze()
        theta = hs.h.squeeze()
        rho = hs.rho.squeeze()
        
        v_x = u_r * np.sin(theta) + u_theta * np.cos(theta)
        v_z = u_r * np.cos(theta) - u_theta * np.sin(theta)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta)
        z = r * np.cos(theta)
        
        # Mirror domain - note v_x changes sign, v_z doesn't
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        v_x_full = np.concatenate([-v_x[:, ::-1], v_x], axis=1)  
        v_z_full = np.concatenate([v_z[:, ::-1], v_z], axis=1)
        u_r_full = np.concatenate([u_r[:, ::-1], u_r], axis=1)  # For color-coding
        
        # Mirror density for background
        rho_full = np.concatenate([rho[:, ::-1], rho], axis=1)
        rho_full[rho_full <= 0] = np.nan
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add density background (lighter)
        im = ax.pcolormesh(x_full, z_full, rho_full,
                          norm=LogNorm(vmin=np.nanpercentile(rho_full, 5),
                                      vmax=np.nanpercentile(rho_full, 95)),
                          cmap='gray', shading='auto', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Density (log)', shrink=0.8)
        
        # CRITICAL FIX: Use magnetized_analysis_v3.py arrow parameters!
        skip = 6  # More arrows
        
        # Flatten for color-coding
        x_flat = x_full[::skip, ::skip].flatten()
        z_flat = z_full[::skip, ::skip].flatten()
        vx_flat = v_x_full[::skip, ::skip].flatten()
        vz_flat = v_z_full[::skip, ::skip].flatten()
        ur_flat = u_r_full[::skip, ::skip].flatten()
        
        # Color-code by radial direction
        inward_mask = ur_flat < 0  # Infall
        outward_mask = ur_flat >= 0  # Outflow
        
        # INWARD arrows (BLACK) - MUCH MORE VISIBLE!
        if np.any(inward_mask):
            ax.quiver(x_flat[inward_mask], z_flat[inward_mask],
                     vx_flat[inward_mask], vz_flat[inward_mask],
                     scale=0.3, scale_units='xy',  # KEY CHANGE!
                     width=0.008,  # Thicker (was 0.003)
                     headwidth=6, headlength=7,  # Bigger heads
                     color='black', alpha=0.95, zorder=5)
        
        # OUTWARD arrows (GRAY)
        if np.any(outward_mask):
            ax.quiver(x_flat[outward_mask], z_flat[outward_mask],
                     vx_flat[outward_mask], vz_flat[outward_mask],
                     scale=0.3, scale_units='xy',
                     width=0.008,
                     headwidth=6, headlength=7,
                     color='lightgray', edgecolors='gray', linewidths=0.5,
                     alpha=0.95, zorder=5)
        
        # Legend
        legend_elements = [
            Line2D([0], [0], marker='>', color='w', markerfacecolor='black',
                   markersize=12, label='Inflow (ur<0)'),
            Line2D([0], [0], marker='>', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markersize=12, label='Outflow (ur≥0)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
        ax.set_xlabel('x (r_g)', fontsize=12)
        ax.set_ylabel('z (r_g)', fontsize=12)
        title = f'Velocity Field (t={hs.t:.1f})'
        if scenario:
            title += f' - {scenario}'
        ax.set_title(title, fontsize=13, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"velocity_field_{scenario.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()
    
    def plot_angular_velocity_field(self, dump_file, scenario=""):
        """
        Plot velocity field specifically for angular momentum scenarios.
        Shows both radial (v_r) and azimuthal (v_phi) components.
        """
        self.load_data("gdump", dump_file)
        
        # Get velocity components
        v_r = hs.v1p
        
        # Check for azimuthal velocity
        if not (hasattr(hs, 'v3p') and hs.v3p is not None):
            print("Warning: No azimuthal velocity found. Using regular velocity plot.")
            return self.plot_velocity_field(dump_file, scenario)
        
        v_phi = hs.v3p
        rho = hs.rho
        
        # Calculate total velocity magnitude
        v_total = np.sqrt(v_r**2 + v_phi**2)
        
        # Coordinates
        r = hs.r.squeeze()
        theta = hs.h.squeeze()
        
        # Mirror domain
        x_full, z_full, v_r_full = self.mirror_domain(r, theta, v_r.squeeze())
        _, _, v_phi_full = self.mirror_domain(r, theta, v_phi.squeeze())
        _, _, v_total_full = self.mirror_domain(r, theta, v_total.squeeze())
        _, _, rho_full = self.mirror_domain(r, theta, rho.squeeze())
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Density background for all plots
        rho_pos = rho_full.copy()
        rho_pos[rho_pos <= 0] = np.nan
        
        # Plot 1: Radial velocity
        im1 = axes[0].pcolormesh(x_full, z_full, rho_pos, norm=LogNorm(),
                                 cmap='gray', alpha=0.3, shading='auto')
        v_r_plot = axes[0].pcolormesh(x_full, z_full, v_r_full,
                                      cmap='RdBu_r', alpha=0.7, shading='auto',
                                      vmin=-np.max(np.abs(v_r_full)),
                                      vmax=np.max(np.abs(v_r_full)))
        axes[0].set_title('Radial Velocity (v_r)', fontsize=12, weight='bold')
        axes[0].set_xlabel('x (r_g)')
        axes[0].set_ylabel('z (r_g)')
        axes[0].set_aspect('equal')
        plt.colorbar(v_r_plot, ax=axes[0], label='v_r')
        
        # Add horizon
        horizon1 = plt.Circle((0, 0), 2.0, color='black', fill=True)
        axes[0].add_patch(horizon1)
        
        # Plot 2: Azimuthal velocity
        im2 = axes[1].pcolormesh(x_full, z_full, rho_pos, norm=LogNorm(),
                                 cmap='gray', alpha=0.3, shading='auto')
        v_phi_plot = axes[1].pcolormesh(x_full, z_full, v_phi_full,
                                        cmap='PRGn', alpha=0.7, shading='auto',
                                        vmin=-np.max(np.abs(v_phi_full)),
                                        vmax=np.max(np.abs(v_phi_full)))
        axes[1].set_title('Azimuthal Velocity (v_φ)', fontsize=12, weight='bold')
        axes[1].set_xlabel('x (r_g)')
        axes[1].set_ylabel('z (r_g)')
        axes[1].set_aspect('equal')
        plt.colorbar(v_phi_plot, ax=axes[1], label='v_φ')
        
        # Add horizon
        horizon2 = plt.Circle((0, 0), 2.0, color='black', fill=True)
        axes[1].add_patch(horizon2)
        
        # Plot 3: Total velocity magnitude
        im3 = axes[2].pcolormesh(x_full, z_full, rho_pos, norm=LogNorm(),
                                 cmap='gray', alpha=0.3, shading='auto')
        v_total_plot = axes[2].pcolormesh(x_full, z_full, v_total_full,
                                          cmap='viridis', alpha=0.7, shading='auto')
        axes[2].set_title('Total Velocity |v|', fontsize=12, weight='bold')
        axes[2].set_xlabel('x (r_g)')
        axes[2].set_ylabel('z (r_g)')
        axes[2].set_aspect('equal')
        plt.colorbar(v_total_plot, ax=axes[2], label='|v|')
        
        # Add horizon
        horizon3 = plt.Circle((0, 0), 2.0, color='black', fill=True)
        axes[2].add_patch(horizon3)
        
        # Super title
        fig.suptitle(f'Angular Momentum Velocity Field (t={hs.t:.1f}) - {scenario}',
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        filename = f"angular_velocity_field_{scenario.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()
        print(f"✓ Saved: {filename}")

    # =========================================================================
    # ANIMATIONS WITH PROPER COLOR SCALING
    # =========================================================================
    
    def animate_density(self, dump_files, scenario="", fps=12):
        """
        Create density animation with percentile-based color scaling.
        """
        print(f"Creating density animation ({len(dump_files)} frames)...")
        
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Compute global color limits (percentile-based)
        print("  Computing global color limits...")
        all_rho = []
        for dump_file in dump_files[::5]:  # Sample
            self.load_data("gdump", dump_file)
            rho_pos = hs.rho.squeeze()[hs.rho.squeeze() > 0]
            all_rho.extend(rho_pos.flatten())
        
        # Use narrower percentiles (1-99 instead of 5-95) for better evolution visibility
        vmin = np.percentile(all_rho, 5)
        vmax = np.percentile(all_rho, 95)
        print(f"  Color range: [{vmin:.2e}, {vmax:.2e}]")
        
        # Also print time range to verify evolution
        self.load_data("gdump", dump_files[0])
        t_start = hs.t
        self.load_data("gdump", dump_files[-1])
        t_end = hs.t
        print(f"  Time range: {t_start:.2f} to {t_end:.2f}")
        
        # Setup coordinates
        self.load_data("gdump", dump_files[0])
        r = hs.r.squeeze()
        theta = hs.h.squeeze()
        x = r * np.sin(theta)
        z = r * np.cos(theta)
        
        def update(frame):
            ax.clear()
            self.load_data("gdump", dump_files[frame])
            rho = hs.rho.squeeze()
            rho[rho <= 0] = np.nan
            
            # Mirror domain for full circle
            r_current = hs.r.squeeze()
            theta_current = hs.h.squeeze()
            x_full, z_full, rho_full = self.mirror_domain(r_current, theta_current, rho)
            
            ax.pcolormesh(x_full, z_full, rho_full, norm=LogNorm(vmin=vmin, vmax=vmax),
                        cmap='viridis', shading='auto')
            ax.set_xlabel('x (r_g)', fontsize=12)
            ax.set_ylabel('z (r_g)', fontsize=12)
            ax.set_title(f'Density Evolution (t={hs.t:.1f})', fontsize=14)
            ax.set_aspect('equal')
            
            return []
        
        # Initial frame for colorbar
        self.load_data("gdump", dump_files[0])
        rho = hs.rho.squeeze()
        rho[rho <= 0] = np.nan
        im = ax.pcolormesh(x, z, rho, norm=LogNorm(vmin=vmin, vmax=vmax),
                          cmap='viridis', shading='auto')
        cbar = fig.colorbar(im, ax=ax, label='Density (log)')
        
        ani = animation.FuncAnimation(fig, update, frames=len(dump_files),
                                     interval=100, blit=False)
        
        filename = f"density_anim_{scenario.lower().replace(' ', '_')}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        
        # TRY-EXCEPT BLOCK for animation saving (Issue #2 fix)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"\n✓ Saved: {filename}")
        except Exception as e:
            print(f"\n✗ Error saving animation: {e}")
            print("  Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
        finally:
            plt.close(fig)
    
    def animate_sonic_surface(self, dump_files, scenario="", fps=10):
        """
        Animate sonic surface evolution with Mach number coloring.
        Shows full circle with mirrored domain.
        """
        print(f"Creating sonic surface animation ({len(dump_files)} frames)...")
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        # Sample files for reasonable animation length (inline, no helper function)
        if len(dump_files) > 100:
            # Sample every Nth file to get ~100 frames
            sample_every = max(1, len(dump_files) // 100)
            sampled_files = dump_files[::sample_every]
        else:
            sampled_files = dump_files
        
        print(f"  Using {len(sampled_files)} frames")
        
        # Compute global Mach number range
        print("  Computing global Mach range...")
        all_mach = []
        for dump_file in sampled_files[::5]:  # Sample
            self.load_data("gdump", dump_file)
            rho = hs.rho.squeeze()
            v_r = (hs.uu[1] / hs.uu[0]).squeeze()
            P = hs.pg.squeeze()
            c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
            mach = (np.abs(v_r) / c_s).squeeze()
            
            mach_valid = mach[np.isfinite(mach)]
            all_mach.extend(mach_valid.flatten())
        
        vmin_mach = np.percentile(all_mach, 5)
        vmax_mach = np.percentile(all_mach, 95)
        print(f"  Mach range: [{vmin_mach:.2f}, {vmax_mach:.2f}]")
        
        def update(frame):
            ax.clear()
            self.load_data("gdump", sampled_files[frame])
            
            # Compute fields
            rho = hs.rho.squeeze()
            v_r = (hs.uu[1] / hs.uu[0]).squeeze()
            P = hs.pg.squeeze()
            c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
            mach = (np.abs(v_r) / c_s).squeeze()
            
            r = hs.r.squeeze()
            theta = hs.h.squeeze()
            
            # Mirror domain for full circle
            x_full, z_full, mach_full = self.mirror_domain(r, theta, mach)
            
            # Plot Mach number with colormap
            mach_full[~np.isfinite(mach_full)] = np.nan
            im = ax.pcolormesh(x_full, z_full, mach_full,
                            vmin=vmin_mach, vmax=vmax_mach,
                            cmap='RdYlBu_r', shading='auto', alpha=0.8)
            
            # Overlay sonic surface contour (Mach = 1) - TRIPLE LAYER!
            try:
                # Base layer
                ax.contour(x_full, z_full, mach_full, levels=[1.0],
                        colors='black', linewidths=8, alpha=0.7, zorder=3)
                # Main layer
                ax.contour(x_full, z_full, mach_full, levels=[1.0],
                        colors='red', linewidths=5, alpha=1.0, zorder=4)
                # Highlight
                ax.contour(x_full, z_full, mach_full, levels=[1.0],
                        colors='white', linewidths=2, alpha=0.9, zorder=5)
            except:
                # Silently fail if Mach=1 doesn't exist at this time
                pass
            
            # Add subsonic/supersonic labels
            ax.text(0.02, 0.98, 'Subsonic\n(M < 1)',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            ax.text(0.98, 0.98, 'Supersonic\n(M > 1)',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            ax.set_xlabel('x (r_g)', fontsize=12)
            ax.set_ylabel('z (r_g)', fontsize=12)
            ax.set_title(f'Sonic Surface Evolution (t={hs.t:.1f})', fontsize=14)
            ax.set_aspect('equal')
            
            # Add horizon circle
            horizon = plt.Circle((0, 0), 1.0, color='black', fill=False,
                            linestyle='--', linewidth=1.5, alpha=0.5)
            ax.add_patch(horizon)
            
            return [im]
        
        # Create initial frame for colorbar
        self.load_data("gdump", sampled_files[0])
        rho = hs.rho.squeeze()
        v_r = (hs.uu[1] / hs.uu[0]).squeeze()
        P = hs.pg.squeeze()
        c_s = np.sqrt(GAMMA * P / (rho + P / (GAMMA - 1)))
        mach = (np.abs(v_r) / c_s).squeeze()
        r = hs.r.squeeze()
        theta = hs.h.squeeze()
        
        x_full, z_full, mach_full = self.mirror_domain(r, theta, mach)
        mach_full[~np.isfinite(mach_full)] = np.nan
        
        im = ax.pcolormesh(x_full, z_full, mach_full,
                        vmin=vmin_mach, vmax=vmax_mach,
                        cmap='RdYlBu_r', shading='auto', alpha=0.8)
        cbar = fig.colorbar(im, ax=ax, label='Mach Number', pad=0.02)
        cbar.ax.axhline(y=1.0, color='black', linewidth=2, linestyle='-')
        cbar.ax.text(0.5, 1.0, 'Sonic', transform=cbar.ax.transAxes,
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        ani = animation.FuncAnimation(fig, update, frames=len(sampled_files),
                                    interval=100, blit=False)
        
        filename = f"sonic_anim_{scenario.lower().replace(' ', '_')}.mp4"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save with error handling
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"\nâœ“ Saved: {filename}")
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"  Size: {file_size:.2f} MB, Duration: {len(sampled_files)/fps:.1f}s")
        except Exception as e:
            print(f"\nâœ— Error saving animation: {e}")
            print("  Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
        finally:
            plt.close(fig)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main analysis function with clean argument handling.
    
    IMPORTANT: This script expects a 'dumps' symlink in the current directory
    pointing to your data. The --scenario flag is ONLY for plot labeling.
    """
    
    parser = argparse.ArgumentParser(
        description='Bondi Accretion Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SETUP INSTRUCTIONS:
  Before running, create a symlink pointing to your data:
    cd ~/harmpi
    ln -sf dumps_bondi_2d_pure_128 dumps
    
  The 'dumps' symlink must exist and point to a folder containing:
    - gdump (grid file)
    - dump000, dump001, ... (dump files)

EXAMPLES:
  # 1D Pure Bondi analysis
  ln -sf dumps_bondi_1d_pure dumps
  python bondi_analysis.py --dim 1
  
  # If you DON'T want animations
  ln -sf dumps_bondi_2d_bhl_128 dumps
  python bondi_analysis.py --scenario bhl --no-animations
  
  # 2D gradient with velocity check
  ln -sf dumps_bondi_2d_gradient_128 dumps
  python bondi_analysis.py --scenario gradient --velocity-check

NOTE: The --scenario flag is ONLY for labeling plots. The actual data 
      comes from whatever the 'dumps' symlink points to!
        """
    )
    
    # Core options
    parser.add_argument('--scenario', type=str, default='pure',
                       choices=['pure', 'bhl', 'gradient', 'angular', 'random'],
                       help='Scenario name (for plot labels only)')
    parser.add_argument('--dim', type=int, default=2, choices=[1, 2],
                       help='Dimension (1D or 2D)')
    parser.add_argument('--output', type=str, default='./bondi_plots',
                       help='Output directory')
    
    # Analysis flags
    parser.add_argument('--no-animations', action='store_true',
                       help='Skip animations (they run by default)')
    parser.add_argument('--velocity-check', action='store_true',
                       help='Verify velocity transformation (Session 2)')
    parser.add_argument('--sonic-map', action='store_true',
                       help='Generate 2D sonic surface map')
    parser.add_argument('--fps', type=int, default=12,
                       help='Animation FPS')
    
    args = parser.parse_args()
    
    # Scenario names for plot labels
    scenario_names = {
        'pure': 'Pure Bondi',
        'bhl': 'Bondi-Hoyle-Lyttleton',
        'gradient': 'Density Gradient',
        'angular': 'Angular Momentum',
        'random': 'Random Velocity'
    }
    scenario_name = scenario_names[args.scenario]
    
    # Check working directory
    if not os.path.exists("dumps"):
        print("ERROR: 'dumps' symlink not found!")
        print("\nPlease create a symlink to your data folder:")
        print("  cd ~/harmpi")
        print("  ln -sf dumps_bondi_2d_pure_128 dumps")
        print("\nThen run this script again.")
        return
    
    if not os.path.islink("dumps"):
        print("WARNING: 'dumps' exists but is not a symlink.")
        print("This script expects 'dumps' to be a symlink pointing to your data.")
    
    # Check for gdump
    if not os.path.exists("dumps/gdump"):
        print("ERROR: dumps/gdump not found!")
        print("Make sure your dumps folder contains the gdump file.")
        return
    
    # Get dump files using glob - returns BARE BASENAMES
    dump_files = sorted(glob.glob("dumps/dump[0-9][0-9][0-9]"))
    dump_files = [os.path.basename(f) for f in dump_files]  # Just "dump000", not "dumps/dump000"
    
    if not dump_files:
        print("ERROR: No dump files found in dumps/")
        return
    
    # Initialize analyzer
    analyzer = BondiAnalysis(output_dir=args.output)
    
    print(f"\n{'='*70}")
    print(f"BONDI ANALYSIS: {scenario_name.upper()} ({args.dim}D)")
    print(f"{'='*70}")
    print(f"Data source: {os.readlink('dumps') if os.path.islink('dumps') else 'dumps/'}")
    print(f"Dumps found: {len(dump_files)} files (dump000 to dump{len(dump_files)-1:03d})")
    print(f"Output: {args.output}/")
    print()
    
    # =========================================================================
    # ANALYSIS EXECUTION
    # =========================================================================

    if args.dim == 1:
        # 1D Analysis
        print("Running 1D sonic surface analysis...")
        sonic_results = analyzer.analyze_1d_sonic_surface(dump_files[::5])
        
        print("Generating 1D plots...")
        analyzer.plot_1d_density_profiles(dump_files[::5], sonic_results)
        analyzer.plot_1d_sonic_evolution(sonic_results)
        
        print(f"\n✓ 1D Analysis complete!")
        print(f"  - density_1d_profiles.png")
        print(f"  - sonic_evolution_1d.png")
    
    else:
        # 2D Analysis
        print("Running 2D sonic surface analysis...")
        sonic_results = analyzer.analyze_2d_sonic_surface(dump_files)
        
        print("Generating 2D sonic evolution plot...")
        analyzer.plot_2d_sonic_evolution(sonic_results, scenario_name)
        
        # Velocity verification
        if args.velocity_check or args.scenario in ['bhl', 'gradient', 'random', 'angular']:
            print("\nVerifying velocity transformation...")
            stats = analyzer.verify_velocity_transformation(dump_files[-1])
            
            # Use angular-specific plot for angular momentum scenario
            if args.scenario == 'angular':
                print("Generating angular momentum velocity field plot...")
                analyzer.plot_angular_velocity_field(dump_files[len(dump_files)//2], scenario_name)
            else:
                analyzer.plot_velocity_field(dump_files[len(dump_files)//2], scenario_name)
        
        # Sonic surface map
        if args.sonic_map or True:  # Always generate
            print("\nGenerating sonic surface map...")
            analyzer.plot_2d_sonic_surface_map(dump_files[len(dump_files)//2], scenario_name)
        
        # Animations (run by default unless --no-animations specified)
        if not args.no_animations:
            print(f"\n{'='*70}")
            print(f"GENERATING ANIMATIONS FOR {scenario_name.upper()}")
            print(f"{'='*70}")
            analyzer.animate_density(dump_files, scenario_name, fps=args.fps)
            analyzer.animate_sonic_surface(dump_files, scenario_name, fps=args.fps)
        
        print(f"\n✓ 2D Analysis complete!")
        print(f"  - sonic_evolution_2d_{args.scenario}.png")
        print(f"  - sonic_map_2d_{args.scenario}.png")
        if args.velocity_check or args.scenario in ['bhl', 'gradient', 'random', 'angular']:
            print(f"  - velocity_field_{args.scenario}.png")
        if not args.no_animations:
            print(f"  - density_anim_{args.scenario}.mp4 (if ffmpeg worked)")
            print(f"  - sonic_anim_{args.scenario}.mp4 (if ffmpeg worked)")
    
    print(f"\nAll outputs saved to: {args.output}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
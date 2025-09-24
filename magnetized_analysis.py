#!/usr/bin/env python3
"""
Complete Magnetized Black Hole Analysis Script
For HARM simulations - both 1D and 2D monopole problems
"""

import harm_script as hs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import argparse
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import glob


class MagnetizedAnalysis:
    """Class for analyzing magnetized black hole problems"""
    
    def __init__(self, output_dir="./magnetized_plots"):
        self.output_dir = output_dir
        self.cache = {}
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, grid_file, dump_file):
        """Load grid and simulation data with caching"""
        if "grid" not in self.cache:
            hs.rg(grid_file)
            self.cache["grid"] = True
            print(f"Loaded grid from {grid_file}.")
        hs.rd(dump_file)
        print(f"Loaded data from {dump_file}, time: {hs.t:.6f}")
    
    def calculate_magnetization(self):
        """Calculate magnetization parameter sigma = b^2/(4π*rho*c^2) = bsq/rho"""
        if not hasattr(hs, 'bsq') or not hasattr(hs, 'rho'):
            print("Error: bsq or rho not available. Make sure data is loaded.")
            return None
        
        # Avoid division by zero
        sigma = np.where(hs.rho > 0, hs.bsq / hs.rho, 0)
        return sigma
    
    def calculate_lorentz_factor(self):
        """Calculate Lorentz factor gamma = alpha * u^0"""
        if not hasattr(hs, 'alpha') or not hasattr(hs, 'uu'):
            # Calculate alpha if not available
            if hasattr(hs, 'guu'):
                alpha = (-hs.guu[0,0])**(-0.5)
            else:
                print("Error: Cannot calculate Lorentz factor - missing metric data")
                return None
        else:
            alpha = hs.alpha
        
        gamma = alpha * hs.uu[0]
        return gamma
    
    def calculate_omega_ratios(self):
        """Calculate field angular velocity ratios ΩF/ΩH"""
        # Need to compute auxiliary quantities first
        hs.aux()  # This computes omegaf1, omegaf2, and other quantities
        
        if not hasattr(hs, 'omegaf2'):
            print("Error: omegaf2 not computed. aux() may have failed.")
            return None, None
        
        # Black hole angular velocity
        a = hs.a  # Black hole spin
        rhor = 1 + (1 - a**2)**0.5  # Horizon radius
        omega_h = a / (2 * rhor)
        
        # Field angular velocity
        omega_f = hs.omegaf2
        
        # Ratio
        omega_ratio = omega_f / omega_h if omega_h != 0 else 0
        
        return omega_f, omega_ratio
    
    def analyze_1d_monopole(self, dump_files, sample_every=1):
        """Complete analysis of 1D monopole problem"""
        results = {
            'times': [],
            'sigma_horizon': [],
            'sigma_initial': None,
            'lorentz_factors': [],
            'omega_ratios': [],
            'radial_profiles': {
                'r': None,
                'gamma': [],
                'sigma': [],
                'times': []
            }
        }
        
        print("=== 1D MONOPOLE ANALYSIS ===")
        
        # Sample dump files
        sampled_files = dump_files[::sample_every]
        
        for i, dump_file in enumerate(sampled_files):
            try:
                self.load_data("gdump", dump_file)
                current_time = float(hs.t)
                
                # Calculate key quantities
                sigma = self.calculate_magnetization()
                gamma = self.calculate_lorentz_factor()
                omega_f, omega_ratio = self.calculate_omega_ratios()
                
                if sigma is not None and gamma is not None:
                    # Extract 1D profiles (along equatorial plane)
                    r_profile = hs.r.squeeze()
                    sigma_profile = sigma.squeeze()
                    gamma_profile = gamma.squeeze()
                    
                    # Store initial magnetization (from first dump)
                    if results['sigma_initial'] is None:
                        # Find magnetization near horizon (within first few cells)
                        horizon_idx = 5  # First few cells near horizon
                        results['sigma_initial'] = np.mean(sigma_profile[:horizon_idx])
                        results['radial_profiles']['r'] = r_profile
                    
                    # Store current profiles
                    results['times'].append(current_time)
                    results['sigma_horizon'].append(sigma_profile[0])  # At innermost cell
                    results['lorentz_factors'].append(gamma_profile[-1])  # At outermost cell
                    
                    if omega_ratio is not None:
                        # Average over available cells
                        avg_omega_ratio = np.mean(omega_ratio.squeeze()) if hasattr(omega_ratio, 'squeeze') else omega_ratio
                        results['omega_ratios'].append(avg_omega_ratio)
                    
                    # Store profiles for selected times
                    if i % 5 == 0:  # Every 5th file for profile evolution
                        results['radial_profiles']['gamma'].append(gamma_profile)
                        results['radial_profiles']['sigma'].append(sigma_profile)
                        results['radial_profiles']['times'].append(current_time)
                
                print(f"Processed {dump_file}: t={current_time:.3f}")
                
            except Exception as e:
                print(f"Error processing {dump_file}: {e}")
                continue
        
        return results
    
    def analyze_2d_monopole(self, dump_files, sample_every=5):
        """Enhanced analysis for 2D BZ monopole problems"""
        print("=== 2D BZ MONOPOLE ANALYSIS ===")
        
        results = {
            'times': [],
            'power_extraction': [],
            'fast_surface_data': [],
            'omega_theta_profiles': [],
            'horizon_data': []
        }
        
        for i, dump_file in enumerate(dump_files[::sample_every]):
            try:
                self.load_data("gdump", dump_file)
                current_time = float(hs.t)
                
                print(f"Processing {dump_file}: t={current_time:.3f}")
                
                # Calculate auxiliary quantities
                hs.aux()
                
                # Get 2D data
                r_2d = hs.r.squeeze()
                h_2d = hs.h.squeeze()  # theta coordinate
                rho_2d = hs.rho.squeeze()
                
                # Power extraction analysis
                if hasattr(hs, 'Tud'):
                    # Energy flux: -g^(1/2) * T^r_t
                    dEr = -hs.gdet * hs.Tud[1,0] * hs._dx2 * hs._dx3
                    Er = dEr.sum(axis=-1) if dEr.ndim > 2 else dEr.sum(axis=1)  # Sum over phi
                    
                    # FIXED: Average over theta to get single radial profile
                    if Er.ndim > 1:
                        Er_avg = Er.mean(axis=1)  # Average over theta direction
                    else:
                        Er_avg = Er
                    
                    results['power_extraction'].append({
                        'time': current_time,
                        'energy_flux': Er_avg,
                        'r_coord': r_2d[:,0] if r_2d.ndim > 1 else r_2d
                    })
                
                # Fast surface analysis (where u^r = 0)
                if hasattr(hs, 'uu'):
                    ur = hs.uu[1].squeeze()
                    results['fast_surface_data'].append({
                        'time': current_time,
                        'ur': ur,
                        'r': r_2d,
                        'h': h_2d,
                        'rho': rho_2d
                    })
                
                # ΩF/ΩH analysis at horizon
                if hasattr(hs, 'omegaf2') and hasattr(hs, 'a'):
                    omega_f = hs.omegaf2.squeeze()
                    a = hs.a
                    rhor = 1 + (1 - a**2)**0.5
                    omega_h = a / (2 * rhor)
                    
                    # Find horizon index (closest to rhor)
                    r_1d = r_2d[:,0] if r_2d.ndim > 1 else r_2d
                    horizon_idx = np.abs(r_1d - rhor).argmin()
                    
                    # Get ΩF/ΩH profile at horizon
                    if omega_f.ndim > 1:
                        omega_f_horizon = omega_f[horizon_idx, :]
                        theta_horizon = h_2d[horizon_idx, :] if h_2d.ndim > 1 else h_2d
                        omega_ratio_horizon = omega_f_horizon / omega_h if omega_h != 0 else omega_f_horizon * 0
                    else:
                        omega_f_horizon = omega_f
                        theta_horizon = np.pi/2  # equatorial
                        omega_ratio_horizon = omega_f_horizon / omega_h if omega_h != 0 else 0
                    
                    results['omega_theta_profiles'].append({
                        'time': current_time,
                        'theta': theta_horizon,
                        'omega_ratio': omega_ratio_horizon,
                        'horizon_radius': rhor
                    })
                
                results['times'].append(current_time)
                
            except Exception as e:
                print(f"Error in 2D analysis for {dump_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def plot_1d_monopole_results(self, results, show=True):
        """Plot results from 1D monopole analysis - clean and focused"""
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(16, 10))
        
        # Use GridSpec with tighter spacing - reduced gaps between subplots
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25, 
                     left=0.06, right=0.96, top=0.88, bottom=0.10)
        
        # 1. Lorentz factor evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if results['times'] and results['lorentz_factors']:
            ax1.plot(results['times'], results['lorentz_factors'], 'b-', linewidth=3)
            ax1.set_xlabel('Time', fontsize=13)
            ax1.set_ylabel('Lorentz Factor γ', fontsize=13)
            ax1.set_title('Plasma Acceleration', fontsize=14, pad=15)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=12)
            
            # Clean annotation for final value
            final_gamma = results['lorentz_factors'][-1]
            ax1.text(0.05, 0.95, f'Final γ = {final_gamma:.2f}', 
                    transform=ax1.transAxes, fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
        
        # 2. Frame dragging ratio
        ax2 = fig.add_subplot(gs[0, 1])
        if results['times'] and results['omega_ratios']:
            ax2.plot(results['times'], results['omega_ratios'], 'r-', linewidth=3)
            ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.8, linewidth=2)
            ax2.set_xlabel('Time', fontsize=13)
            ax2.set_ylabel('ΩF/ΩH', fontsize=13)
            ax2.set_title('Frame Dragging Efficiency', fontsize=14, pad=15)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=12)
            
            # Clean annotation
            final_omega = np.mean(results['omega_ratios'][-100:])
            ax2.text(0.05, 0.95, f'Final = {final_omega:.3f}\nTheory = 0.500', 
                    transform=ax2.transAxes, fontsize=13, fontweight='bold', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.9))
        
        # 3. Key values comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if results['sigma_initial'] and results['lorentz_factors']:
            categories = ['σ0', '√σ0', 'γ_final']
            values = [results['sigma_initial'], np.sqrt(results['sigma_initial']), results['lorentz_factors'][-1]]
            colors = ['green', 'orange', 'blue']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Value', fontsize=13)
            ax3.set_title('Magnetization vs Acceleration', fontsize=14, pad=15)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(labelsize=12)
            
            # Value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            ax3.set_ylim(0, max(values) * 1.15)
        
        # 4. Radial acceleration profiles with time progression colorbar (FIXED)
        ax4 = fig.add_subplot(gs[1, 0])
        if results['radial_profiles']['r'] is not None and results['radial_profiles']['gamma']:
            r = results['radial_profiles']['r']
            times = results['radial_profiles']['times']
            
            # Create colormap for time evolution
            colormap = plt.cm.plasma
            norm = plt.Normalize(min(times), max(times))
            
            # Show every 5th profile to avoid clutter (FIXED: changed from 4 to 5)
            for i in range(0, len(results['radial_profiles']['gamma']), 5):
                gamma_prof = results['radial_profiles']['gamma'][i]
                time = times[i]
                color = colormap(norm(time))
                ax4.loglog(r, gamma_prof, color=color, alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Radius (r/rg)', fontsize=13)
            ax4.set_ylabel('Lorentz Factor γ', fontsize=13)
            ax4.set_title('Radial Acceleration Evolution', fontsize=14, pad=15)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=12)
            
            # Add colorbar showing time progression (RESTORED)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax4, shrink=0.8, pad=0.02)
            cbar.set_label('Time', fontsize=12)
            cbar.ax.tick_params(labelsize=11)
        
        # 5. Numerical results summary (MORE COMPACT)
        ax5 = fig.add_subplot(gs[1, 1:])  # Span two columns
        ax5.axis('off')
        
        if results['sigma_initial'] and results['lorentz_factors'] and results['omega_ratios']:
            sigma0 = results['sigma_initial']
            gamma_final = results['lorentz_factors'][-1]
            omega_ratio = np.mean(results['omega_ratios'][-100:])
            sqrt_sigma0 = np.sqrt(sigma0)
            efficiency = gamma_final / sqrt_sigma0
            final_velocity = np.sqrt(1 - 1/gamma_final**2)
            
            # More compact results text
            results_text = f"""
KEY SIMULATION RESULTS

Initial Magnetization:        σ0 = {sigma0:.0f}
Theoretical Max γ:            √σ0 = {sqrt_sigma0:.1f}

Final Lorentz Factor:         γ_final = {gamma_final:.2f}
Final Velocity:               v/c = {final_velocity:.3f}
Acceleration Efficiency:      γ/√σ0 = {efficiency:.2f} ({efficiency*100:.0f}%)

Frame Dragging Ratio:         ΩF/ΩH = {omega_ratio:.3f}
Theoretical Prediction:       ΩF/ΩH = 0.500
Deviation from Theory:        {abs(omega_ratio-0.5)/0.5*100:.1f}%
            """
            
            # Smaller text box with tighter padding
            ax5.text(0.15, 0.95, results_text, transform=ax5.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.9, edgecolor='black'))
        
        # Removed erroneous "(t=200)" from title
        fig.suptitle('1D Monopole Magnetosphere Analysis', 
                    fontsize=18, fontweight='bold', y=0.92)
        
        # Save figure
        filename = os.path.join(self.output_dir, "monopole_1d_results.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved results plot: {filename}")
        
        # Generate detailed Lorentz factor plot
        self.plot_lorentz_factor_detailed(results, show)
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_2d_monopole_results(self, results, show=True):
        """Plot 2D BZ monopole analysis results - clean version"""
        if not results['times']:
            print("No results to plot!")
            return
            
        print(f"Plotting results from {len(results['times'])} time steps")
        
        fig = plt.figure(figsize=(16, 12))
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Power extraction evolution - show time evolution with colorbar
        ax1 = fig.add_subplot(gs[0, 0])
        if results['power_extraction']:
            # Show multiple timesteps with color progression
            times = [data['time'] for data in results['power_extraction']]
            colormap = plt.cm.viridis
            norm = plt.Normalize(min(times), max(times))
            
            # Sample every 5th timestep to avoid clutter
            sample_indices = range(0, len(results['power_extraction']), 5)
            
            for i in sample_indices:
                power_data = results['power_extraction'][i]
                r_coord = power_data['r_coord']
                energy_flux = power_data['energy_flux']
                time = power_data['time']
                color = colormap(norm(time))
                
                ax1.loglog(r_coord, np.abs(energy_flux), color=color, alpha=0.7, linewidth=2)
            
            ax1.set_xlabel('Radius (r/rg)', fontsize=12)
            ax1.set_ylabel('|Energy Flux|', fontsize=12)
            ax1.set_title('Power Extraction Evolution', fontsize=13)  # FIXED: Removed (t=200)
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar for time progression
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
            cbar1.set_label('Time', fontsize=11)
            cbar1.ax.tick_params(labelsize=10)
        
        # 2. Full domain fast surface visualization with better density display
        ax2 = fig.add_subplot(gs[0, 1])
        if results['fast_surface_data']:
            latest_data = results['fast_surface_data'][-1]
            r_2d = latest_data['r']
            h_2d = latest_data['h']
            ur_2d = latest_data['ur']
            rho_2d = latest_data['rho']
            
            # Convert to Cartesian - FIXED: Full circle display
            x = r_2d * np.sin(h_2d)
            z = r_2d * np.cos(h_2d)
            
            # Create mirrored data for full circle visualization
            x_mirror = -x
            z_mirror = z
            rho_mirror = rho_2d
            ur_mirror = ur_2d
            
            # Combine original and mirrored
            x_full = np.concatenate([x, x_mirror], axis=1)
            z_full = np.concatenate([z, z_mirror], axis=1)
            rho_full = np.concatenate([rho_2d, rho_mirror], axis=1)
            ur_full = np.concatenate([ur_2d, ur_mirror], axis=1)
            
            # Better density normalization for visibility
            rho_positive = rho_full[rho_full > 0]
            if len(rho_positive) > 0:
                vmin = np.percentile(rho_positive, 1)  # 1st percentile
                vmax = np.percentile(rho_positive, 99)  # 99th percentile
            else:
                vmin, vmax = rho_full.min(), rho_full.max()
            
            # Plot density with better contrast
            im = ax2.pcolormesh(x_full, z_full, rho_full, cmap='viridis', 
                               norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.8)
            
            # Overplot fast surface contour
            try:
                ax2.contour(x_full, z_full, ur_full, levels=[0], 
                           colors='red', linewidths=2, alpha=0.9)
            except:
                print("Could not plot fast surface contour")
            
            # Add black hole
            if results['omega_theta_profiles']:
                rh = results['omega_theta_profiles'][-1]['horizon_radius']
                circle = plt.Circle((0, 0), rh, color='black', alpha=1.0, zorder=10)
                ax2.add_patch(circle)
            
            ax2.set_xlabel('X (r_g)', fontsize=12)
            ax2.set_ylabel('Z (r_g)', fontsize=12)
            ax2.set_title('Fast Surface (u^r=0)', fontsize=13)
            ax2.set_xlim(-50, 50)
            ax2.set_ylim(-50, 50)
            ax2.set_aspect('equal')
            
            # Add colorbar
            cbar2 = plt.colorbar(im, ax=ax2, label='log10(density)', shrink=0.8)
        
        # 3. ΩF/ΩH vs θ profile - final timestep only (clean)
        ax3 = fig.add_subplot(gs[1, 0])
        if results['omega_theta_profiles']:
            latest_omega = results['omega_theta_profiles'][-1]  # Final timestep only
            theta = latest_omega['theta']
            omega_ratio = latest_omega['omega_ratio']
            final_time = latest_omega['time']
            
            if hasattr(theta, '__len__') and len(theta) > 1:
                ax3.plot(theta, omega_ratio, 'b-', linewidth=3)
                ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.8, linewidth=2)
                ax3.fill_between(theta, 0.5, omega_ratio, alpha=0.2, color='lightblue')
                ax3.set_xlabel('θ (radians)', fontsize=12)
                ax3.set_ylabel('ΩF/ΩH', fontsize=12)
                ax3.set_title(f'Frame Dragging at Horizon (t={final_time:.0f})', fontsize=13)
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(0, np.pi)
                ax3.set_ylim(0.4, 0.8)
                
                # Add theta labels
                ax3.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                ax3.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
            else:
                ax3.text(0.5, 0.5, f'ΩF/ΩH = {omega_ratio:.3f}', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                ax3.set_title('Frame Dragging Ratio', fontsize=13)
        
        # 4. Clean numerical summary 
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        if results['omega_theta_profiles']:
            latest_omega = results['omega_theta_profiles'][-1]
            omega_ratio = latest_omega['omega_ratio']
            
            if hasattr(omega_ratio, '__len__'):
                avg_omega = np.mean(omega_ratio)
                min_omega = np.min(omega_ratio)
                max_omega = np.max(omega_ratio)
                asymmetry = (max_omega - min_omega) / avg_omega * 100
                
                summary_text = f"""
2D BZ MONOPOLE RESULTS

Frame Dragging Analysis:
Average ΩF/ΩH = {avg_omega:.3f}
Min ΩF/ΩH = {min_omega:.3f}  
Max ΩF/ΩH = {max_omega:.3f}
BZ Theory = 0.500
Deviation = {abs(avg_omega-0.5)/0.5*100:.1f}%
Angular variation = {asymmetry:.1f}%

Black Hole Properties:
Horizon radius = {latest_omega['horizon_radius']:.2f} rg
Estimated spin a ≈ 0.9
                """
            else:
                summary_text = f"""
2D BZ MONOPOLE RESULTS

Frame Dragging:
ΩF/ΩH = {omega_ratio:.3f}
BZ Theory = 0.500
Deviation = {abs(omega_ratio-0.5)/0.5*100:.1f}%

Black Hole Properties:
Horizon radius = {latest_omega['horizon_radius']:.2f} rg

Analysis Summary:
Time span: {results['times'][0]:.1f} - {results['times'][-1]:.1f}
Snapshots: {len(results['times'])}
                """
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        fig.suptitle('2D BZ Monopole Magnetosphere Analysis', fontsize=18, fontweight='bold')
        
        # Save figure
        filename = os.path.join(self.output_dir, "bz_monopole_2d_results.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved enhanced 2D analysis plot: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_2d_density_animation(self, dump_files, output_file="bz_monopole_2d_density.mp4", fps=3):
        """Create enhanced 2D density evolution animation - FIXED for full circle display"""
        print(f"Creating enhanced 2D density animation with {len(dump_files)} frames...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Sample more densely at the beginning, then every 3rd file
        early_files = dump_files[:30]  # First 30 files (early evolution)
        later_files = dump_files[30::4]  # Every 4th file after that
        sampled_files = early_files + later_files
        print(f"Using {len(sampled_files)} frames for animation (dense early sampling)")
        
        # Find global density range with better percentile approach
        global_rho_min, global_rho_max = float('inf'), float('-inf')
        all_positive_rho = []
        
        print("Computing global density range...")
        for dump_file in sampled_files[::5]:  # Sample for range calculation
            try:
                self.load_data("gdump", dump_file)
                rho = hs.rho.squeeze()
                positive_rho = rho[rho > 0]
                if len(positive_rho) > 0:
                    all_positive_rho.extend(positive_rho.flatten())
            except:
                continue
        
        if all_positive_rho:
            # Use percentiles for better contrast
            global_rho_min = np.percentile(all_positive_rho, 1)
            global_rho_max = np.percentile(all_positive_rho, 99)
        else:
            global_rho_min, global_rho_max = 1e-10, 1e-5
        
        print(f"Density range (1-99 percentile): {global_rho_min:.2e} to {global_rho_max:.2e}")
        
        def update(frame):
            ax.clear()
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            
            rho = hs.rho.squeeze()
            r = hs.r.squeeze()
            h = hs.h.squeeze()
            
            # Get radial velocity for fast surface
            ur = hs.uu[1].squeeze() if hasattr(hs, 'uu') else None
            
            # Convert to Cartesian
            x = r * np.sin(h)
            z = r * np.cos(h)
            
            # Create full circle by mirroring across x=0 axis (FIXED)
            x_full = np.concatenate([-x[:, ::-1], x], axis=1)
            z_full = np.concatenate([z[:, ::-1], z], axis=1)
            rho_full = np.concatenate([rho[:, ::-1], rho], axis=1)
            
            # Use pcolormesh with better normalization
            im = ax.pcolormesh(x_full, z_full, rho_full, cmap='plasma',
                              norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max),
                              shading='auto', alpha=0.9)
            
            # Add fast surface if available
            if ur is not None:
                try:
                    ur_full = np.concatenate([ur[:, ::-1], ur], axis=1)
                    ax.contour(x_full, z_full, ur_full, levels=[0], 
                              colors='red', linewidths=2, alpha=0.9)
                except:
                    pass  # Skip if contour fails
            
            # Add black hole
            rhor = 1.35  # For a≈0.9
            circle = plt.Circle((0, 0), rhor, facecolor='black', edgecolor='white', 
                              linewidth=2, alpha=1.0, zorder=10)
            ax.add_patch(circle)
            
            ax.set_xlabel('X (r_g)', fontsize=14)
            ax.set_ylabel('Z (r_g)', fontsize=14)
            ax.set_title(f'2D BZ Monopole: Density + Fast Surface (t = {hs.t:.1f})', fontsize=16)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect('equal')
            
            return im,
        
        # Create colorbar using first frame
        self.load_data("gdump", sampled_files[0])
        rho = hs.rho.squeeze()
        r = hs.r.squeeze()
        h = hs.h.squeeze()
        
        x = r * np.sin(h)
        z = r * np.cos(h)
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        rho_full = np.concatenate([rho[:, ::-1], rho], axis=1)
        
        im = ax.pcolormesh(x_full, z_full, rho_full, cmap='plasma',
                          norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max),
                          shading='auto', alpha=0.9)
        cbar = fig.colorbar(im, ax=ax, label='Density (log scale)')
        
        ani = animation.FuncAnimation(fig, update, frames=len(sampled_files),
                                     blit=False, interval=300, repeat=True)  # Slower for better viewing
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved enhanced 2D density animation: {output_path}")
            print(f"Animation duration: {len(sampled_files)/fps:.1f} seconds")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Make sure ffmpeg is installed")
        
        plt.close(fig)
    
    def create_frame_dragging_animation(self, results, output_file="bz_monopole_frame_dragging.mp4", fps=6):
        """Create longer animation showing ΩF/ΩH(θ) evolution over time"""
        if not results['omega_theta_profiles'] or len(results['omega_theta_profiles']) < 10:
            print("Insufficient data for frame dragging animation")
            return
            
        print(f"Creating frame dragging animation with {len(results['omega_theta_profiles'])} time steps...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use more time steps for longer animation
        sample_indices = np.linspace(0, len(results['omega_theta_profiles'])-1, 
                                   min(80, len(results['omega_theta_profiles']))).astype(int)
        
        def update(frame):
            ax.clear()
            
            omega_data = results['omega_theta_profiles'][sample_indices[frame]]
            theta = omega_data['theta']
            omega_ratio = omega_data['omega_ratio']
            time = omega_data['time']
            
            if hasattr(theta, '__len__') and len(theta) > 1:
                # Plot current profile
                ax.plot(theta, omega_ratio, 'b-', linewidth=4, alpha=0.9)
                
                # Add theory reference
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.8, linewidth=2, 
                          label='BZ Theory = 0.5')
                
                # Fill area
                ax.fill_between(theta, 0.5, omega_ratio, alpha=0.3, color='lightblue')
                
                # Styling
                ax.set_xlabel('θ (radians)', fontsize=14)
                ax.set_ylabel('ΩF/ΩH', fontsize=14)
                ax.set_title(f'Frame Dragging Evolution at Horizon (t = {time:.1f})', fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, np.pi)
                ax.set_ylim(0.4, 0.8)
                
                # Add theta labels
                ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
                
                # Add statistics
                avg_omega = np.mean(omega_ratio)
                min_omega = np.min(omega_ratio)
                max_omega = np.max(omega_ratio)
                
                stats_text = f'Avg: {avg_omega:.3f}\nRange: {min_omega:.3f} - {max_omega:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ani = animation.FuncAnimation(fig, update, frames=len(sample_indices),
                                     blit=False, interval=200, repeat=True)  # Slightly slower
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved frame dragging animation: {output_path}")
            print(f"Animation duration: {len(sample_indices)/fps:.1f} seconds")
        except Exception as e:
            print(f"Error saving frame dragging animation: {e}")
        
        plt.close(fig)
    
    def create_power_extraction_animation(self, results, output_file="bz_monopole_power_extraction.mp4", fps=6):
        """Create animation showing power extraction evolution over time - FIXED"""
        if not results['power_extraction'] or len(results['power_extraction']) < 10:
            print("Insufficient data for power extraction animation")
            return
            
        print(f"Creating power extraction animation with {len(results['power_extraction'])} time steps...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample time steps for animation
        sample_indices = np.linspace(0, len(results['power_extraction'])-1, 
                                   min(60, len(results['power_extraction']))).astype(int)
        
        def update(frame):
            ax.clear()
            
            power_data = results['power_extraction'][sample_indices[frame]]
            r_coord = power_data['r_coord']
            energy_flux = power_data['energy_flux']
            time = power_data['time']
            
            # Ensure we have single arrays (not 2D)
            if energy_flux.ndim > 1:
                energy_flux = energy_flux.mean(axis=0)  # Average if still 2D
            if r_coord.ndim > 1:
                r_coord = r_coord[:, 0]  # Take first column if 2D
            
            # Plot current power profile
            ax.loglog(r_coord, np.abs(energy_flux), 'b-', linewidth=3, alpha=0.9)
            
            # Styling
            ax.set_xlabel('Radius (r/rg)', fontsize=14)
            ax.set_ylabel('|Energy Flux|', fontsize=14)
            ax.set_title(f'Power Extraction Evolution (t = {time:.1f})', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # Add statistics in corner - FIXED numpy format error completely
            try:
                # Ensure all values are 1D numpy arrays
                energy_flux_1d = np.asarray(energy_flux).flatten()
                r_coord_1d = np.asarray(r_coord).flatten()
                
                total_power = np.trapezoid(np.abs(energy_flux_1d), r_coord_1d)
                max_power = np.max(np.abs(energy_flux_1d))
                
                # Convert to regular Python numbers to avoid any numpy formatting issues
                total_power_val = float(total_power) if np.isscalar(total_power) else float(total_power.item())
                max_power_val = float(max_power) if np.isscalar(max_power) else float(max_power.item())
                
                stats_text = f'Total Power: {total_power_val:.2e}\nPeak Flux: {max_power_val:.2e}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            except Exception as e:
                print(f"Debug info - energy_flux shape: {np.asarray(energy_flux).shape}, type: {type(energy_flux)}")
                print(f"Debug info - r_coord shape: {np.asarray(r_coord).shape}, type: {type(r_coord)}")
                # Fallback without statistics
                ax.text(0.02, 0.98, f'Time: {float(time):.1f}', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ani = animation.FuncAnimation(fig, update, frames=len(sample_indices),
                                     blit=False, interval=200, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved power extraction animation: {output_path}")
            print(f"Animation duration: {len(sample_indices)/fps:.1f} seconds")
        except Exception as e:
            print(f"Error saving power extraction animation: {e}")
        
        plt.close(fig)
    
    def plot_lorentz_factor_detailed(self, results, show=True):
        """Create detailed plot focusing on Lorentz factor evolution"""
        
        # Fix the array condition check
        if (results['radial_profiles']['r'] is None or 
            not results['radial_profiles']['gamma'] or 
            len(results['radial_profiles']['gamma']) == 0):
            print("No radial profile data available for detailed Lorentz factor plot")
            return
        
        fig = plt.figure(figsize=(12, 8))
        
        r = results['radial_profiles']['r']
        times = results['radial_profiles']['times']
        
        # Create colormap for time evolution
        colormap = plt.cm.viridis
        norm = plt.Normalize(min(times), max(times))
        
        # Show more profiles for detailed view
        for i in range(0, len(results['radial_profiles']['gamma']), 2):
            gamma_prof = results['radial_profiles']['gamma'][i]
            time = times[i]
            color = colormap(norm(time))
            alpha = 0.6 if i < len(results['radial_profiles']['gamma']) - 5 else 0.9  # Highlight latest profiles
            linewidth = 2 if i < len(results['radial_profiles']['gamma']) - 5 else 3
            
            plt.loglog(r, gamma_prof, color=color, alpha=alpha, linewidth=linewidth)
        
        plt.xlabel('Radius (r/rg)', fontsize=14)
        plt.ylabel('Lorentz Factor γ', fontsize=14)
        plt.title('Detailed Radial Acceleration: γ(r) Evolution', fontsize=16, pad=20)
        plt.grid(True, alpha=0.3)
        plt.tick_params(labelsize=12)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
        cbar.set_label('Time', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Add annotations for key physics
        if results['sigma_initial'] and results['lorentz_factors']:
            sqrt_sigma0 = np.sqrt(results['sigma_initial'])
            final_gamma = results['lorentz_factors'][-1]
            
            # Add horizontal lines for reference
            plt.axhline(y=sqrt_sigma0, color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.axhline(y=final_gamma, color='blue', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add text annotations
            plt.text(0.02, 0.95, f'Theoretical Max: γ = √σ0 = {sqrt_sigma0:.1f}', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.9))
            
            plt.text(0.02, 0.85, f'Achieved: γ_final = {final_gamma:.2f}', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
            
            efficiency = final_gamma / sqrt_sigma0 * 100
            plt.text(0.02, 0.75, f'Efficiency: {efficiency:.0f}% of theoretical max', 
                    transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
        
        # Save detailed plot
        filename = os.path.join(self.output_dir, "monopole_1d_lorentz_detailed.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved detailed Lorentz factor plot: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()


def get_dump_files(dump_folder="dumps", pattern="dump[0-9][0-9][0-9]"):
    """Get sorted list of dump files"""
    dump_files = sorted(glob.glob(os.path.join(dump_folder, pattern)))
    dump_files = [os.path.basename(f) for f in dump_files]  # Remove path
    return dump_files


def main():
    parser = argparse.ArgumentParser(description="Analyze Magnetized Black Hole Problems")
    parser.add_argument("--problem", type=str, default="monopole_1d", 
                       choices=["monopole_1d", "monopole_2d", "bz_monopole"],
                       help="Type of magnetized problem to analyze")
    parser.add_argument("--output", type=str, default="./magnetized_plots", 
                       help="Output directory for plots and movies")
    parser.add_argument("--sample", type=int, default=1, 
                       help="Sample every N dump files")
    parser.add_argument("--animate", action="store_true", 
                       help="Create density evolution animation (2D only)")
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MagnetizedAnalysis(output_dir=args.output)
    
    # Get dump files
    dump_files = get_dump_files()
    if not dump_files:
        print("No dump files found! Make sure simulation has run.")
        return
    
    print(f"Found {len(dump_files)} dump files")
    print(f"Analyzing {args.problem} problem...")
    
    if args.problem == "monopole_1d":
        # Analyze 1D monopole problem
        results = analyzer.analyze_1d_monopole(dump_files, sample_every=args.sample)
        analyzer.plot_1d_monopole_results(results)
        
        print("\n=== KEY RESULTS ===")
        if results['sigma_initial']:
            print(f"Initial magnetization σ₀: {results['sigma_initial']:.2e}")
            print(f"√σ₀: {np.sqrt(results['sigma_initial']):.2f}")
        if results['lorentz_factors']:
            print(f"Final Lorentz factor: {results['lorentz_factors'][-1]:.2f}")
        if results['omega_ratios']:
            avg_omega = np.mean(results['omega_ratios'])
            print(f"Average ΩF/ΩH: {avg_omega:.3f} (theory: 0.5)")
    
    elif args.problem in ["monopole_2d", "bz_monopole"]:
        # Analyze 2D monopole problems
        results = analyzer.analyze_2d_monopole(dump_files, sample_every=args.sample)
        analyzer.plot_2d_monopole_results(results)
        
        # Create animation if requested
        if args.animate:
            analyzer.create_2d_density_animation(dump_files)
            analyzer.create_frame_dragging_animation(results)
            analyzer.create_power_extraction_animation(results)
        
        print("\n=== 2D KEY RESULTS ===")
        if results['omega_theta_profiles']:
            latest_omega = results['omega_theta_profiles'][-1]
            omega_ratio = latest_omega['omega_ratio']
            if hasattr(omega_ratio, '__len__'):
                avg_omega = np.mean(omega_ratio)
                print(f"Average ΩF/ΩH: {avg_omega:.3f} (theory: 0.5)")
                print(f"ΩF/ΩH range: {np.min(omega_ratio):.3f} to {np.max(omega_ratio):.3f}")
            else:
                print(f"ΩF/ΩH: {omega_ratio:.3f} (theory: 0.5)")
        
        if results['fast_surface_data']:
            print(f"Fast surface analysis completed for {len(results['fast_surface_data'])} time steps")
        
        if results['power_extraction']:
            print(f"Power extraction analysis completed for {len(results['power_extraction'])} time steps")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
FIXED Magnetized Black Hole Analysis Script
For HARM simulations of monopole and other magnetized problems
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
    
    def plot_1d_monopole_results(self, results, show=True):
        """Plot results from 1D monopole analysis - clean and focused"""
        
        # Create figure with more space and larger subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Use GridSpec with better spacing - more room between suptitle and subplots
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4, 
                     left=0.06, right=0.96, top=0.82, bottom=0.12)
        
        # 1. Lorentz factor evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if results['times'] and results['lorentz_factors']:
            ax1.plot(results['times'], results['lorentz_factors'], 'b-', linewidth=3)
            ax1.set_xlabel('Time', fontsize=13)
            ax1.set_ylabel('Lorentz Factor γ', fontsize=13)
            ax1.set_title('Plasma Acceleration', fontsize=14, pad=20)
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
            ax2.set_title('Frame Dragging Efficiency', fontsize=14, pad=20)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=12)
            
            # Clean annotation
            final_omega = np.mean(results['omega_ratios'][-100:])
            ax2.text(0.05, 0.95, f'Final = {final_omega:.3f}\nTheory = 0.500', 
                    transform=ax2.transAxes, fontsize=13, fontweight='bold', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.9))
        
        # 3. Key values comparison (with more space)
        ax3 = fig.add_subplot(gs[0, 2])
        if results['sigma_initial'] and results['lorentz_factors']:
            categories = ['σ0', '√σ0', 'γ_final']
            values = [results['sigma_initial'], np.sqrt(results['sigma_initial']), results['lorentz_factors'][-1]]
            colors = ['green', 'orange', 'blue']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Value', fontsize=13)
            ax3.set_title('Magnetization vs Acceleration', fontsize=14, pad=20)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(labelsize=12)
            
            # Value labels on bars with better positioning
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            # Adjust y-axis to give more space for labels
            ax3.set_ylim(0, max(values) * 1.15)
        
        # 4. Radial acceleration profiles with colorbar
        ax4 = fig.add_subplot(gs[1, 0])
        if results['radial_profiles']['r'] is not None and results['radial_profiles']['gamma']:
            r = results['radial_profiles']['r']
            times = results['radial_profiles']['times']
            
            # Create colormap for time evolution
            colormap = plt.cm.plasma
            norm = plt.Normalize(min(times), max(times))
            
            # Show every 4th profile to avoid clutter
            for i in range(0, len(results['radial_profiles']['gamma']), 4):
                gamma_prof = results['radial_profiles']['gamma'][i]
                time = times[i]
                color = colormap(norm(time))
                ax4.loglog(r, gamma_prof, color=color, alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Radius (r/rg)', fontsize=13)
            ax4.set_ylabel('Lorentz Factor γ', fontsize=13)
            ax4.set_title('Radial Acceleration Profiles', fontsize=14, pad=20)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=12)
            
            # Add clean colorbar
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax4, shrink=0.8, pad=0.02)
            cbar.set_label('Time', fontsize=12)
            cbar.ax.tick_params(labelsize=11)
        
        # 5. Numerical results summary (spans two columns with more space)
        ax5 = fig.add_subplot(gs[1, 1:])  # Span two columns
        ax5.axis('off')
        
        if results['sigma_initial'] and results['lorentz_factors'] and results['omega_ratios']:
            sigma0 = results['sigma_initial']
            gamma_final = results['lorentz_factors'][-1]
            omega_ratio = np.mean(results['omega_ratios'][-100:])
            sqrt_sigma0 = np.sqrt(sigma0)
            efficiency = gamma_final / sqrt_sigma0
            final_velocity = np.sqrt(1 - 1/gamma_final**2)
            
            # Create clean results table
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
            
            ax5.text(0.15, 0.98, results_text, transform=ax5.transAxes, 
                    fontsize=13, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="lightgray", alpha=0.9, edgecolor='black'))
        
        # Add clean main title with optimal spacing
        fig.suptitle('1D Monopole Magnetosphere Analysis', 
                    fontsize=18, fontweight='bold', y=0.90)
        
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


if __name__ == "__main__":
    main()
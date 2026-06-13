#!/usr/bin/env python3
"""
Complete Torus Problem Analysis Script for HARM Simulations
Based on SOMA2017 exercises - analyzing MRI, accretion, and magnetized dynamics
"""

import harm_script as hs
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['savefig.facecolor'] = 'white'   # inline backend sets transparent figs; force opaque saves
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import argparse
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import glob
import re


class TorusAnalysis:
    """Class for analyzing magnetized torus problems"""
    
    def __init__(self, output_dir="./torus_analysis"):
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
        return hs.t
    
    def get_torus_mask(self, rho_threshold=0.1, r_min=6, r_max=50):
        """
        Get mask for torus region
        
        Parameters:
        -----------
        rho_threshold : float
            Minimum density (default 0.1, assuming rho_max normalized to 1)
        r_min : float
            Minimum radius (default 6 rg, inner edge of torus)
        r_max : float
            Maximum radius (default 50 rg, approximate outer edge)
        
        Returns:
        --------
        mask : numpy array
            Boolean mask, True in torus region
        """
        rho = hs.rho.squeeze() if hasattr(hs, 'rho') else None
        r = hs.r.squeeze() if hasattr(hs, 'r') else None
        
        if rho is None or r is None:
            raise ValueError("Cannot create mask: rho or r not available")
        
        mask = (rho > rho_threshold) & (r > r_min) & (r < r_max)
        
        return mask

    def get_disk_mask(self, rho_threshold=0.1, r_min=6, r_max=50, theta_width=0.3):
        """
        Get mask for disk midplane region
        
        Parameters:
        -----------
        rho_threshold : float
            Minimum density
        r_min, r_max : float
            Radial range
        theta_width : float
            Angular width around equator (radians)
        
        Returns:
        --------
        mask : numpy array
            Boolean mask, True in disk midplane
        """
        rho = hs.rho.squeeze() if hasattr(hs, 'rho') else None
        r = hs.r.squeeze() if hasattr(hs, 'r') else None
        th = hs.h.squeeze() if hasattr(hs, 'h') else None
        
        if rho is None or r is None or th is None:
            raise ValueError("Cannot create mask: data not available")
        
        mask = (rho > rho_threshold) & \
            (r > r_min) & (r < r_max) & \
            (np.abs(th - np.pi/2) < theta_width)
        
        return mask

    def analyze_mri_resolution(self, dump_file="dump000"):
        """
        Analyze MRI wavelength resolution in initial conditions
        Good resolution: ≥15 cells per wavelength, acceptable: 5-10 cells
        
        FIXED: Only calculates in torus region, not entire domain
        """
        print("=== MRI RESOLUTION ANALYSIS ===")
        
        self.load_data("gdump", dump_file)
        
        # Calculate MRI wavelength resolution using harm_script function
        if hasattr(hs, 'Qmri'):
            # Check resolution in theta direction (dir=2)
            Q_theta = hs.Qmri(dir=2)
            
            # CRITICAL FIX: Only average over torus region
            rho = hs.rho.squeeze() if hasattr(hs, 'rho') else None
            
            if rho is not None and Q_theta is not None:
                # Squeeze to remove singleton dimensions
                if Q_theta.ndim == 3:
                    Q_theta = Q_theta.squeeze()
                
                # Define torus mask - only where material exists
                mask = rho > 0.1
                
                if mask.sum() > 0:
                    # Calculate statistics ONLY in torus
                    avg_resolution = Q_theta[mask].mean()
                    min_resolution = Q_theta[mask].min()
                    max_resolution = Q_theta[mask].max()
                    
                    print(f"MRI resolution in θ-direction (torus only): {avg_resolution:.1f} cells/wavelength")
                    print(f"Minimum resolution: {min_resolution:.1f} cells/wavelength")
                    print(f"Maximum resolution: {max_resolution:.1f} cells/wavelength")
                    print(f"Analyzed {mask.sum()} cells in torus (out of {mask.size} total)")
                else:
                    print("ERROR: No torus material found (ρ > 0.1)")
                    avg_resolution = 0
            else:
                print("ERROR: Could not get density or Qmri data")
                avg_resolution = 0
            
            # Check resolution in radial direction (dir=1) if available
            try:
                Q_radial = hs.Qmri(dir=1)
                if Q_radial is not None and rho is not None:
                    if Q_radial.ndim == 3:
                        Q_radial = Q_radial.squeeze()
                    Q_r_avg = Q_radial[mask].mean()
                    print(f"MRI resolution in r-direction (torus only): {Q_r_avg:.1f} cells/wavelength")
            except Exception as e:
                print("Radial MRI resolution not available")
            
            # Assessment
            if avg_resolution >= 15:
                assessment = "EXCELLENT"
            elif avg_resolution >= 10:
                assessment = "GOOD"
            elif avg_resolution >= 5:
                assessment = "ACCEPTABLE"
            else:
                assessment = "POOR - MRI may not develop properly"
            
            print(f"Resolution Assessment: {assessment}")

            # Plot Q_theta spatial map + histogram (so we see WHERE MRI is resolved)
            self._plot_mri_resolution(Q_theta, rho, mask, dump_file)

            return {
                'Q_theta': Q_theta,
                'avg_resolution': avg_resolution,
                'assessment': assessment
            }
        else:
            print("ERROR: Qmri function not available in harm_script")
            return None
    def _plot_mri_resolution(self, Q_theta, rho, mask, dump_file):
        """Save 2D map + histogram of Q_theta inside torus mask."""
        r = hs.r.squeeze()
        z = hs.r.squeeze() * np.cos(hs.h.squeeze())
        R = hs.r.squeeze() * np.sin(hs.h.squeeze())

        # Mask out non-torus cells so colormap focuses on the disk
        Q_masked = np.where(mask, Q_theta, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: 2D spatial map of Q_theta in poloidal plane
        im = axes[0].pcolormesh(R, z, Q_masked, cmap="viridis",
                                vmin=0, vmax=20, shading="gouraud")
        axes[0].set_xlabel("R [r_g]")
        axes[0].set_ylabel("z [r_g]")
        axes[0].set_title(f"Q_theta map (torus cells only), {dump_file}")
        axes[0].set_xlim(0, 80)
        axes[0].set_ylim(-40, 40)
        axes[0].set_aspect("equal")
        cbar = fig.colorbar(im, ax=axes[0])
        cbar.set_label("Q_theta [cells per MRI wavelength]")
        # Reference lines for resolution thresholds
        for val, lbl in [(6, "Q=6"), (10, "Q=10")]:
            cbar.ax.axhline(val, color="red", lw=0.8, ls="--")

        # Right: histogram of Q_theta inside torus
        Q_inside = Q_theta[mask]
        Q_inside = Q_inside[np.isfinite(Q_inside)]
        axes[1].hist(Q_inside, bins=60, range=(0, 30), color="steelblue", edgecolor="k")
        axes[1].axvline(6,  color="orange", ls="--", lw=1.5, label="Q=6 (marginal)")
        axes[1].axvline(10, color="green",  ls="--", lw=1.5, label="Q=10 (good)")
        axes[1].axvline(np.median(Q_inside), color="red", ls="-", lw=2,
                        label=f"median = {np.median(Q_inside):.1f}")
        axes[1].set_xlabel("Q_theta")
        axes[1].set_ylabel("number of cells")
        axes[1].set_title("Distribution of MRI resolution inside torus")
        axes[1].legend()

        plt.tight_layout()
        out = f"{self.output_dir}/mri_resolution_{dump_file}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    def analyze_time_evolution(self, dump_files, sample_every=5):
        """Analyze time evolution to identify quasi-stationary regime"""
        print("=== TIME EVOLUTION ANALYSIS ===")
        
        results = {
            'times': [],
            'mdot': [],               # flux-integrated mdot at 2*r_horizon (positive=inflow)
            'r_measure': None,        # radius where mdot measured (set once below)
            'spin': None,             # black hole spin a, for reproducibility
            'E_mag': [],              # volume-weighted magnetic energy in torus (rho>0.1)
            'u0_mean': [],            # mean u^0 / Lorentz factor proxy (NOT kinetic energy)
            'density_center': [],     # mean rho over inner third in radius
            'quasi_stationary_start': None
        }
        
        for dump_file in dump_files[::sample_every]:
            try:
                current_time = self.load_data("gdump", dump_file)
                
                # Get 2D data
                r_2d = hs.r.squeeze()
                rho_2d = hs.rho.squeeze()
                
                # Flux-integrated mdot at 2*r_horizon: -∮ rho u^r sqrt(-g) dx2 dx3.
                # Positive = inflow. r_horizon from hs.rhor, so this tracks spin automatically.
                r_measure = 2.0 * hs.rhor
                i_mdot = hs.iofr(r_measure)
                mdot = -np.sum(hs.rho[i_mdot,:,:] * hs.uu[1][i_mdot,:,:]
                               * hs.gdet[i_mdot,:,:]) * hs._dx2 * hs._dx3
                
                # Volume-weighted magnetic energy in torus: ∫ (b^2/2) sqrt(-g) dx1 dx2 dx3
                # over rho>0.1 cells. Real energy, not an unweighted mean.
                if hasattr(hs, 'bsq'):
                    dV = hs.gdet.squeeze() * hs._dx1 * hs._dx2 * hs._dx3
                    bsq_sq = hs.bsq.squeeze()
                    tor = rho_2d > 0.1
                    E_mag = np.sum(0.5 * bsq_sq[tor] * dV[tor])
                else:
                    E_mag = 0
                
                # mean u^0 (Lorentz factor proxy), not kinetic energy
                if hasattr(hs, 'uu'):
                    u0_mean = np.mean(hs.uu[0].squeeze())
                else:
                    u0_mean = 0
                
                # Central density (as indicator of torus evolution)
                center_idx = len(r_2d) // 3  # Inner third of domain
                density_center = np.mean(rho_2d[:center_idx, :])
                
                results['times'].append(current_time)
                results['mdot'].append(mdot)
                results['r_measure'] = float(r_measure)
                results['spin'] = float(hs.a)
                results['E_mag'].append(E_mag)
                results['u0_mean'].append(u0_mean)
                results['density_center'].append(density_center)
                
                print(f"t={current_time:.1f}: ρ_center={density_center:.3e}, E_mag={E_mag:.3e}") # before: B²={bsq_mean:.3e}
                
            except Exception as e:
                print(f"Error processing {dump_file}: {e}")
                continue
        
        # Identify quasi-stationary regime
        if len(results['times']) > 10:
            # Look for when magnetic energy stabilizes
            mag_proxy = np.array(results['E_mag'])
            times = np.array(results['times'])
            
            # Simple criterion: when magnetic energy stops growing rapidly
            if len(mag_proxy) > 20:
                growth_rate = np.gradient(mag_proxy, times)
                # Find when growth rate becomes small
                stable_indices = np.where(np.abs(growth_rate) < 0.1 * np.max(np.abs(growth_rate)))[0]
                if len(stable_indices) > 0:
                    quasi_start_idx = stable_indices[len(stable_indices)//3]  # Take later stable period
                    results['quasi_stationary_start'] = times[quasi_start_idx]
                    print(f"Quasi-stationary regime estimated to start at t ≈ {results['quasi_stationary_start']:.1f}")
        
        self._plot_time_evolution(results)
        
        return results
    
    def _plot_time_evolution(self, results):
        """Save canonical time-evolution figure: E_mag(t), mdot(t), rho_center(t).

        Plateau statistics are computed over t > quasi_stationary_start when
        available, else over the last half of the run.
        """
        times = np.array(results['times'])
        if times.size == 0:
            print("No time-evolution data to plot; skipping figure.")
            return
        E_mag = np.array(results['E_mag'])
        mdot = np.array(results['mdot'])
        rho_c = np.array(results['density_center'])

        t_qs = results['quasi_stationary_start']
        t_plateau = t_qs if t_qs is not None else 0.5 * times.max()
        plateau = times > t_plateau

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: volume-weighted magnetic energy (torus, rho>0.1), log scale
        axes[0].plot(times, E_mag, color="navy", lw=1.5)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("t [r_g/c]")
        axes[0].set_ylabel("E_mag = ∫ (b²/2) √(-g) dV  (torus, ρ>0.1)")
        axes[0].set_title("Magnetic energy: MRI growth and saturation")
        if plateau.sum() > 1:
            m, s = E_mag[plateau].mean(), E_mag[plateau].std()
            axes[0].axhspan(m - s, m + s, color="navy", alpha=0.15)
            axes[0].axhline(m, color="navy", ls="--", lw=1,
                            label=f"plateau mean = {m:.1f} ± {s:.2f}")
            axes[0].legend()

        # Middle: flux-integrated mdot at r_measure (positive = inflow)
        axes[1].plot(times, mdot, color="darkgreen", lw=1.2)
        axes[1].axhline(0, color="gray", lw=0.8)
        axes[1].set_xlabel("t [r_g/c]")
        axes[1].set_ylabel(r"$\dot{M} = -\oint \rho\, u^{r}\, \sqrt{-g}\, dx^{2}\, dx^{3}$") # originally literal "mdot = -∮ ρ uʳ √(-g) dx² dx³"
        axes[1].set_title(f"Accretion rate at r = {results['r_measure']:.2f} r_g "
                          f"(2 r_horizon, a = {results['spin']:.2f})")
        if plateau.sum() > 1:
            m, s = mdot[plateau].mean(), mdot[plateau].std()
            axes[1].axhspan(m - s, m + s, color="darkgreen", alpha=0.15)
            axes[1].axhline(m, color="darkgreen", ls="--", lw=1,
                            label=f"plateau mean = {m:.3f} ± {s:.3f}")
            axes[1].legend()

        # Right: mean density over inner third in radius (inward mass transport)
        axes[2].plot(times, rho_c, color="maroon", lw=1.5)
        axes[2].set_xlabel("t [r_g/c]")
        axes[2].set_ylabel("⟨ρ⟩, inner third in radius")
        axes[2].set_title("Inner-region density: inward mass transport")

        # Mark quasi-stationary start on all panels
        if t_qs is not None:
            for ax in axes:
                ax.axvline(t_qs, color="black", ls=":", lw=1.2)
            axes[0].text(t_qs, axes[0].get_ylim()[1], f" t_qs ≈ {t_qs:.0f}",
                         va="top", fontsize=10)

        plt.tight_layout()
        out = f"{self.output_dir}/time_evolution.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")
        
    def calculate_alpha_parameter(self, dump_files, quasi_start_time=None, sample_every=3):
        """
        Alpha parameter from orthonormal-frame stress: α = T^{r̂φ̂} / p_gas,
        T^{r̂φ̂} ≈ sqrt(gcov11/gcov33) * Tud[1,3] (code coords; neglects t-φ
        frame-dragging mixing — OK for disk-body stats at r>6).

        Maxwell part: -b^{r̂} b^{φ̂} ≈ -sqrt(gcov11/gcov33) * bu[1] * bd[3];
        the remainder w*u^r*u_φ is HYDRODYNAMIC stress (mean-flow advection +
        turbulence), NOT pure turbulent Reynolds stress.

        Sign convention: positive α = outward angular-momentum transport.
        MRI Maxwell stress should come out POSITIVE; the hydro remainder is
        typically NEGATIVE in the inflow (inward advection of ang. momentum).
        """
        print("=== ALPHA PARAMETER ANALYSIS ===")

        if quasi_start_time is None:
            analysis_files = dump_files
        else:
            analysis_files = []
            for dump_file in dump_files:
                try:
                    self.load_data("gdump", dump_file)
                    if hs.t >= quasi_start_time:
                        analysis_files.append(dump_file)
                except:
                    continue

        alpha_total_sum = None
        alpha_mag_sum = None
        alpha_rey_sum = None
        beta_sum = None
        mask_count = None          # per-cell: how many snapshots cell was in torus
        count = 0

        for dump_file in analysis_files[::sample_every]:
            try:
                self.load_data("gdump", dump_file)
                hs.aux()

                if not (hasattr(hs, 'Tud') and hasattr(hs, 'bu') and hasattr(hs, 'bd')):
                    print(f"Warning: stress/field not available for {dump_file}")
                    continue

                T13 = hs.Tud[1, 3].squeeze()
                pg = (hs.gam - 1) * hs.ug.squeeze()
                rho = hs.rho.squeeze()
                r = hs.r.squeeze()

                # Orthonormal projection factor sqrt(g_11/g_33), code coordinates
                g11 = hs.gcov[1, 1].squeeze()
                g33 = hs.gcov[3, 3].squeeze()
                with np.errstate(divide='ignore', invalid='ignore'):
                    hat = np.sqrt(g11 / g33)

                mask = (rho > 0.1) & (r > 6) & (r < 50) & np.isfinite(hat)
                if mask.sum() == 0:
                    continue

                alpha_total = np.zeros_like(T13)
                alpha_mag = np.zeros_like(T13)
                alpha_total[mask] = (hat * T13)[mask] / (pg[mask] + 1e-20)
                # Maxwell stress -b^r b_phi (harmpi names: bu/bd, NOT bcon/bcov)
                T13_mag = -hs.bu[1].squeeze() * hs.bd[3].squeeze()
                alpha_mag[mask] = (hat * T13_mag)[mask] / (pg[mask] + 1e-20)
                alpha_rey = alpha_total - alpha_mag   # hydrodynamic remainder

                # plasma beta — always computed (was trapped in a dead branch)
                bsq_local = hs.bsq.squeeze()
                beta_local = np.zeros_like(pg)
                beta_local[mask] = 2 * pg[mask] / (bsq_local[mask] + 1e-20)

                if alpha_total_sum is None:
                    alpha_total_sum = alpha_total.copy()
                    alpha_mag_sum = alpha_mag.copy()
                    alpha_rey_sum = alpha_rey.copy()
                    beta_sum = beta_local.copy()
                    mask_count = mask.astype(int)
                else:
                    alpha_total_sum += alpha_total
                    alpha_mag_sum += alpha_mag
                    alpha_rey_sum += alpha_rey
                    beta_sum += beta_local
                    mask_count += mask

                count += 1

            except Exception as e:
                print(f"Error processing {dump_file}: {e}")
                continue

        if count > 0:
            # Per-cell average over the snapshots in which the cell was in-torus
            n = np.maximum(mask_count, 1)
            alpha_total_avg = alpha_total_sum / n
            alpha_mag_avg = alpha_mag_sum / n
            alpha_rey_avg = alpha_rey_sum / n
            beta_avg = beta_sum / n

            # Robust mask: in torus for at least half the snapshots
            mask = mask_count >= max(1, count // 2)

            # SIGNED means (abs() hid the sign-convention bug)
            alpha_tot_mean = alpha_total_avg[mask].mean()
            alpha_mag_mean = alpha_mag_avg[mask].mean()
            alpha_rey_mean = alpha_rey_avg[mask].mean()

            denom = np.abs(alpha_mag_mean) + np.abs(alpha_rey_mean)
            mag_fraction = np.abs(alpha_mag_mean) / (denom + 1e-20)
            separation_valid = True

            print(f"\nTime-averaged alpha parameters (signed; from {count} snapshots):")
            print(f"  Total α          = {alpha_tot_mean:+.3f}")
            print(f"  Maxwell α        = {alpha_mag_mean:+.3f}  (expect positive)")
            print(f"  Hydro remainder  = {alpha_rey_mean:+.3f}  (advection+turbulence, NOT pure Reynolds)")
            print(f"  Maxwell fraction of |stress|: {mag_fraction*100:.1f}%")

            beta_median = np.median(beta_avg[mask])
            beta_lo = np.percentile(beta_avg[mask], 10)
            beta_hi = np.percentile(beta_avg[mask], 90)
            print(f"\n  Plasma β in torus: median {beta_median:.1f}, 10-90% {beta_lo:.1f}-{beta_hi:.1f}")
            if beta_median > 10:
                print("  → gas-pressure-dominated disk body (expected for SANE/beta=100 start)")
            elif beta_median > 1:
                print("  → moderately magnetized")
            else:
                print("  → magnetically dominated — check mask")

            results = {
                'alpha_total': alpha_total_avg,
                'alpha_magnetic': alpha_mag_avg,
                'alpha_reynolds': alpha_rey_avg,   # key kept; content = hydro remainder
                'separation_valid': separation_valid,
                'beta': beta_avg,
                'count': count,
                'mask': mask,
                'mask_count': mask_count,
                'r_grid': r,
                'theta_grid': hs.h.squeeze()
            }
            return results
        else:
            print("ERROR: No valid data for alpha calculation")
            return None

    def analyze_angular_velocity(self, dump_files, quasi_start_time=None, sample_every=3):
        """
        Calculate angular velocity Ω = u^φ / u^t
        Compare to Keplerian Ω_K = 1/(r^(3/2) + a)
        
        FIXED: Only calculates in disk midplane, not poles
        """
        print("=== ANGULAR VELOCITY ANALYSIS ===")
        
        if quasi_start_time is None:
            analysis_files = dump_files
        else:
            analysis_files = []
            for dump_file in dump_files:
                try:
                    self.load_data("gdump", dump_file)
                    if hs.t >= quasi_start_time:
                        analysis_files.append(dump_file)
                except:
                    continue
        
        omega_sum = None
        omega_K_sum = None
        count = 0
        
        for dump_file in analysis_files[::sample_every]:
            try:
                self.load_data("gdump", dump_file)
                
                if not hasattr(hs, 'uu'):
                    continue
                
                # Angular velocity Ω = u^φ / u^t
                omega = hs.uu[3].squeeze() / (hs.uu[0].squeeze() + 1e-20)
                # Ω = u^φ/u^t: dxdxp[3,3]=1.0 verified on this grid, so code x3 IS
                # physical φ — no transform factor needed (unlike Qmri's general case).
                
                # Get coordinates
                r = hs.r.squeeze()
                th = hs.h.squeeze()
                rho = hs.rho.squeeze()
                
                # Keplerian angular velocity
                omega_K = 1 / (r**(3/2) + hs.a)
                
                # CRITICAL FIX: Mask for disk midplane only
                # Include: torus material + near equator + reasonable radii
                mask = (rho > 0.1) & (r > 6) & (r < 50) & (np.abs(th - np.pi/2) < 0.3)
                
                if mask.sum() == 0:
                    continue
                
                # Time averaging
                if omega_sum is None:
                    omega_sum = omega.copy()
                    omega_K_sum = omega_K.copy()
                    mask_count = mask.astype(int)
                else:
                    omega_sum += omega
                    omega_K_sum += omega_K
                    mask_count += mask
                
                count += 1
                
            except Exception as e:
                print(f"Error in angular velocity calculation: {e}")
                continue
        
        if count > 0:
            omega_avg = omega_sum / count
            omega_K_avg = omega_K_sum / count
            
            # Calculate ratio ONLY in disk
            mask = mask_count >= max(1, count // 2)   # in disk midplane ≥ half the snapshots
            
            omega_ratio = np.zeros_like(omega_avg)
            omega_ratio[mask] = omega_avg[mask] / (omega_K_avg[mask] + 1e-20)
            
            ratio_mean = omega_ratio[mask].mean()
            
            print(f"\nAngular velocity analysis (from {count} snapshots):")
            print(f"  Ω/Ω_Keplerian = {ratio_mean:.2f}")
            print(f"  Analyzed {mask.sum()} cells in disk midplane")
            
            if ratio_mean < 0.8:
                print("  → Sub-Keplerian rotation (expected for accretion disk)")
            elif ratio_mean > 1.2:
                print("  → Super-Keplerian rotation (unusual - check mask)")
            else:
                print("  → Near-Keplerian rotation")
            
            results = {
                'omega': omega_avg,
                'omega_keplerian': omega_K_avg,
                'omega_K': omega_K_avg,
                'omega_ratio': omega_ratio,
                'count': count,
                'mask': mask,
                'mask_count': mask_count,
                'r_grid': r,
                'theta_grid': th
            }
            
            return results
        else:
            return None
    
    def find_sonic_surfaces(self, dump_file):
        """Find sonic and magnetosonic surfaces in the torus"""
        print("=== SONIC SURFACE ANALYSIS ===")
        
        self.load_data("gdump", dump_file)
        
        surfaces = {}
        
        # Sound speed
        if hasattr(hs, 'pg') or hasattr(hs, 'ug'):
            if hasattr(hs, 'pg'):
                cs2 = hs.gam * hs.pg / hs.rho  # pg already available
            else:
                pg = (hs.gam - 1) * hs.ug
                cs2 = hs.gam * pg / hs.rho
            
            cs = np.sqrt(cs2.squeeze())
            surfaces['sound_speed'] = cs
        
        # Alfven speed
        if hasattr(hs, 'bsq') and hasattr(hs, 'rho'):
            va2 = hs.bsq / hs.rho
            va = np.sqrt(va2.squeeze())
            surfaces['alfven_speed'] = va
            
            # Fast magnetosonic speed (approximate)
            if 'sound_speed' in surfaces:
                cf2 = cs2 + va2  # Simplified fast speed
                cf = np.sqrt(cf2.squeeze())
                surfaces['fast_magnetosonic_speed'] = cf
        
        # Velocity magnitude
        if hasattr(hs, 'uu'):
            v2 = hs.uu[1]**2 + hs.uu[2]**2 + hs.uu[3]**2  # Spatial components
            v = np.sqrt(v2.squeeze())
            surfaces['velocity'] = v
        
        print("Calculated characteristic speeds for sonic surface analysis")
        return surfaces
    
    def create_density_movie(self, dump_files, output_file="torus_density_evolution.mp4", fps=10):
        """Create movie of density evolution with magnetic field lines"""
        print(f"Creating torus density movie with {len(dump_files)} frames...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample files for reasonable movie length
        sampled_files = dump_files[::max(1, len(dump_files)//200)]  # Max 200 frames
        print(f"Using {len(sampled_files)} frames for movie")
        
        # Find global density range
        global_rho_min, global_rho_max = float('inf'), float('-inf')
        
        print("Computing global density range...")
        for dump_file in sampled_files[::10]:  # Sample for range
            try:
                self.load_data("gdump", dump_file)
                rho = hs.rho.squeeze()
                positive_rho = rho[rho > 0]
                if len(positive_rho) > 0:
                    global_rho_min = min(global_rho_min, positive_rho.min())
                    global_rho_max = max(global_rho_max, positive_rho.max())
            except:
                continue
        
        print(f"Density range: {global_rho_min:.2e} to {global_rho_max:.2e}")
        
        def update(frame):
            ax.clear()
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            
            # Get data
            rho = hs.rho.squeeze()
            r = hs.r.squeeze()
            h = hs.h.squeeze()
            
            # Convert to Cartesian for visualization
            x = r * np.sin(h)
            z = r * np.cos(h)
            
            # Plot density
            im = ax.pcolormesh(x, z, rho, cmap='viridis', 
                              norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max),
                              shading='auto')
            
            # Overplot magnetic field lines if available
            if hasattr(hs, 'B'):
                # Calculate vector potential for field lines
                try:
                    aphi = hs.psicalc()  # Magnetic flux function
                    # Plot field lines
                    ax.contour(x, z, aphi, levels=20, colors='white', alpha=0.7, linewidths=0.8)
                except:
                    pass  # Skip if field line calculation fails
            
            # Add black hole
            rhor = 1 + (1 - hs.a**2)**0.5
            circle = plt.Circle((0, 0), rhor, color='black', alpha=1.0)
            ax.add_patch(circle)
            
            ax.set_xlabel('X (r_g)')
            ax.set_ylabel('Z (r_g)')
            ax.set_title(f'Torus Evolution: Density + B-field (t = {hs.t:.1f})')
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.set_aspect('equal')
            
            return im,
        
        # Create colorbar
        self.load_data("gdump", sampled_files[0])
        rho = hs.rho.squeeze()
        r = hs.r.squeeze()
        h = hs.h.squeeze()
        x = r * np.sin(h)
        z = r * np.cos(h)
        
        im = ax.pcolormesh(x, z, rho, cmap='viridis',
                          norm=LogNorm(vmin=global_rho_min, vmax=global_rho_max))
        cbar = fig.colorbar(im, ax=ax, label='Density (log scale)')
        
        ani = animation.FuncAnimation(fig, update, frames=len(sampled_files),
                                     blit=False, interval=100, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved torus density movie: {output_path}")
        except Exception as e:
            print(f"Error saving movie: {e}")
        
        plt.close(fig)
    
    def plot_comprehensive_analysis(self, evolution_results, alpha_results, omega_results, 
                                   mri_results, show=True):
        """Create comprehensive analysis plots"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)
        
        fig.suptitle('Magnetized Torus Analysis: MRI, Accretion & Turbulence', 
                    fontsize=18, fontweight='bold')
        
        # 1. Time evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if evolution_results and evolution_results['times']:
            times = evolution_results['times']
            ax1.semilogy(times, evolution_results['E_mag'], 'r-', linewidth=2, label=r'$E_{\rm mag}$ (torus, code units)')
            ax1.semilogy(times, evolution_results['u0_mean'], 'b-', linewidth=2, label=r'$\langle u^0 \rangle$ (Lorentz factor proxy)')
            
            if evolution_results['quasi_stationary_start']:
                ax1.axvline(evolution_results['quasi_stationary_start'], 
                           color='green', linestyle='--', alpha=0.7, label='Quasi-stationary')
            
            ax1.set_xlabel('Time [M]')
            ax1.set_ylabel('proxy value (code units)')
            ax1.set_title('Field & flow proxies vs time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. MRI Resolution
        ax2 = fig.add_subplot(gs[0, 1])
        if mri_results:
            Q_theta = mri_results['Q_theta']
            r_2d = hs.r.squeeze() if hasattr(hs, 'r') else None
            
            if r_2d is not None and Q_theta is not None:
                # Plot resolution vs radius (averaged over theta)
                if Q_theta.ndim > 1:
                    # Mask to torus cells (rho>0.1) and exclude pathological Q spikes
                    # (omega->0 cells produce Q up to ~1e5). Average over valid cells per radius.
                    rho_2d = hs.rho.squeeze()
                    valid = (rho_2d > 0.1) & (Q_theta < 100) & np.isfinite(Q_theta)
                    Q_masked = np.where(valid, Q_theta, np.nan)
                    with np.errstate(invalid='ignore'):
                        valid_count = np.sum(np.isfinite(Q_masked), axis=1)
                        Q_avg = np.full(Q_masked.shape[0], np.nan)
                        good = valid_count > 0
                        Q_avg[good] = np.nanmean(Q_masked[good, :], axis=1)
                    r_1d = r_2d[:, 0] if r_2d.ndim > 1 else r_2d
                else:
                    Q_avg = Q_theta
                    r_1d = np.arange(len(Q_avg))
                
                ax2.plot(r_1d, Q_avg, 'g-', linewidth=3)
                ax2.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Excellent (≥15)')
                ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Good (≥10)')
                ax2.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Minimum (≥5)')
                
                ax2.set_xlabel('Radius')
                ax2.set_ylabel('Cells per MRI wavelength')
                ax2.set_title(f'MRI resolution vs radius (assessment: {mri_results["assessment"]})')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Alpha parameter map
        ax3 = fig.add_subplot(gs[0, 2])
        if alpha_results:
            alpha_total = alpha_results['alpha_total']
            r_grid = alpha_results['r_grid']
            theta_grid = alpha_results['theta_grid']
            
            # Convert to Cartesian for plotting
            x = r_grid * np.sin(theta_grid)
            z = r_grid * np.cos(theta_grid)
            
            # Plot alpha parameter
            # Mask to torus so vacuum NaNs don't smear the map; tighten scale to the data range
            alpha_plot = np.where(alpha_results['mask'], alpha_total, np.nan)
            cmap = plt.cm.RdBu_r.copy()
            cmap.set_bad(alpha=0.0)          # NaN (vacuum) renders transparent, not grey
            im = ax3.pcolormesh(x, z, alpha_plot, cmap=cmap,
                                vmin=-0.05, vmax=0.05, shading='gouraud')
            cbar3 = plt.colorbar(im, ax=ax3, label='α_total')
            
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title('Alpha Parameter Map')
            ax3.set_aspect('equal')
        
        # 4. Mass accretion rate evolution
        ax4 = fig.add_subplot(gs[1, 0])
        if evolution_results and evolution_results['times']:
            ax4.plot(evolution_results['times'], evolution_results['mdot'], 'purple', linewidth=2)
            ax4.set_xlabel('Time [M]')
            ax4.set_ylabel(r'$\dot{M}$ at $2r_+$ (code units)')
            ax4.set_title('Accretion rate vs time')
            ax4.grid(True, alpha=0.3)
        
        # 5. Alpha components comparison
        ax5 = fig.add_subplot(gs[1, 1])
        if alpha_results:
            alpha_total = alpha_results['alpha_total']
            alpha_magnetic = alpha_results['alpha_magnetic']
            alpha_reynolds = alpha_results['alpha_reynolds']
            
            # Radial profiles (theta-averaged)
            if alpha_total.ndim > 1:
                r_1d = alpha_results['r_grid'][:, 0] if alpha_results['r_grid'].ndim > 1 else alpha_results['r_grid']
                pm = alpha_results['mask']
                alpha_total_masked = np.where(pm, alpha_total,    np.nan)
                alpha_mag_masked   = np.where(pm, alpha_magnetic, np.nan)
                alpha_rey_masked   = np.where(pm, alpha_reynolds, np.nan)
                with np.errstate(invalid='ignore'):
                    nrows = alpha_total_masked.shape[0]
                    good = np.sum(np.isfinite(alpha_total_masked), axis=1) > 0
                    alpha_tot_avg = np.full(nrows, np.nan)
                    alpha_mag_avg = np.full(nrows, np.nan)
                    alpha_rey_avg = np.full(nrows, np.nan)
                    alpha_tot_avg[good] = np.nanmean(alpha_total_masked[good, :], axis=1)
                    alpha_mag_avg[good] = np.nanmean(alpha_mag_masked[good, :],   axis=1)
                    alpha_rey_avg[good] = np.nanmean(alpha_rey_masked[good, :],   axis=1)
            else:
                r_1d = np.arange(len(alpha_total))
                alpha_tot_avg = alpha_total
                alpha_mag_avg = alpha_magnetic
                alpha_rey_avg = alpha_reynolds
            
            ax5.plot(r_1d, alpha_tot_avg, 'k-', linewidth=3, label='Total')
            ax5.plot(r_1d, alpha_mag_avg, 'r-', linewidth=2, label='Magnetic')
            ax5.plot(r_1d, alpha_rey_avg, 'b-', linewidth=2, label='Hydro (advection+turb.)')
            
            ax5.set_xlim(5, 55)
            ax5.axhline(0, color='gray', lw=0.8)
            ax5.set_xlabel('Radius')
            ax5.set_ylabel('Alpha Parameter')
            ax5.set_title('Alpha Components vs Radius')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Angular velocity comparison
        ax6 = fig.add_subplot(gs[1, 2])
        if omega_results:
            omega = omega_results['omega']
            omega_kep = omega_results['omega_keplerian']
            r_grid = omega_results['r_grid']
            
            # Theta-averaged profiles
            if omega.ndim > 1:
                r_1d = r_grid[:, 0] if r_grid.ndim > 1 else r_grid
                om = omega_results['mask']
                omega_masked     = np.where(om, omega,     np.nan)
                omega_kep_masked = np.where(om, omega_kep, np.nan)
                with np.errstate(invalid='ignore'):
                    nrows = omega_masked.shape[0]
                    good = np.sum(np.isfinite(omega_masked), axis=1) > 0
                    omega_avg     = np.full(nrows, np.nan)
                    omega_kep_avg = np.full(nrows, np.nan)
                    omega_avg[good]     = np.nanmean(omega_masked[good, :],     axis=1)
                    omega_kep_avg[good] = np.nanmean(omega_kep_masked[good, :], axis=1)
            else:
                r_1d = np.arange(len(omega))
                omega_avg = omega
                omega_kep_avg = omega_kep
            
            ax6.loglog(r_1d, np.abs(omega_avg), 'b-', linewidth=3, label='Simulation Ω')
            ax6.loglog(r_1d, omega_kep_avg, 'r--', linewidth=2, label='Keplerian Ω_K')
            
            ax6.set_xlabel('Radius')
            ax6.set_ylabel('Angular Velocity')
            ax6.set_title('Ω vs Ω_Keplerian')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Density snapshot (equatorial slice)
        ax7 = fig.add_subplot(gs[2, 0])
        if hasattr(hs, 'rho') and hasattr(hs, 'r'):
            rho = hs.rho.squeeze()
            r_grid = hs.r.squeeze()
            
            if rho.ndim > 1:
                # Take equatorial slice
                eq_idx = rho.shape[1] // 2
                rho_eq = rho[:, eq_idx]
                r_eq = r_grid[:, 0] if r_grid.ndim > 1 else r_grid
            else:
                rho_eq = rho
                r_eq = r_grid
            
            ax7.loglog(r_eq, rho_eq, 'g-', linewidth=3)
            ax7.set_xlabel('Radius')
            ax7.set_ylabel('Density (equatorial)')
            ax7.set_title('Radial Density Profile')
            ax7.grid(True, alpha=0.3)
        
        # 8. Summary statistics
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        summary_text = "TORUS ANALYSIS SUMMARY\n\n"
        
        if mri_results:
            summary_text += f"MRI Resolution: {mri_results['avg_resolution']:.1f} cells/wavelength ({mri_results['assessment']})\n"
        
        if alpha_results:
            # FIXED: Use masked values, not entire array
            if 'mask' in alpha_results:
                mask = alpha_results['mask']
                alpha_mean = alpha_results['alpha_total'][mask].mean()
                alpha_mag = alpha_results['alpha_magnetic'][mask].mean()
                alpha_rey = alpha_results['alpha_reynolds'][mask].mean()
            else:
                alpha_mean = np.nanmean(alpha_results['alpha_total'])
                alpha_mag = np.nanmean(alpha_results['alpha_magnetic'])
                alpha_rey = np.nanmean(alpha_results['alpha_reynolds'])
            summary_text += f"Alpha Parameter: α_total = {alpha_mean:.3f}\n"
            summary_text += f"  α_magnetic = {alpha_mag:.3f}\n"
            summary_text += f"  α_reynolds = {alpha_rey:.3f}\n"
        
        if evolution_results and evolution_results['quasi_stationary_start']:
            summary_text += f"Quasi-stationary regime starts: t ≈ {evolution_results['quasi_stationary_start']:.1f}\n"
        
        if omega_results:
        # Use masked values
            if 'mask' in omega_results:
                mask = omega_results['mask']
                omega_ratio_mean = omega_results['omega_ratio'][mask].mean()
            else:
                omega_ratio_mean = np.nanmean(omega_results['omega_ratio'])
            summary_text += f"Ω/Ω_K ratio: {omega_ratio_mean:.2f}\n"
        
        summary_text += f"\nNotes:\n"
        summary_text += f"• E_mag: volume-weighted ∫(b²/2)√g dV over torus (ρ>0.1)\n"
        summary_text += r"• $\dot{M}$: flux-integrated $-\oint \rho\, u^{r}\sqrt{-g}\, dx^{2}dx^{3}$ at $2r_+$" + "\n"
        summary_text += f"• α: orthonormal-frame T^(r̂φ̂)/p_g, signed; 'Reynolds'=hydro incl. mean-flow advection\n"
        summary_text += f"• quasi-stationary start is heuristic"
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        # Save figure
        filename = os.path.join(self.output_dir, "torus_comprehensive_analysis.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved comprehensive analysis: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def analyze_vertical_alpha(self, dump_files, quasi_start_time=None, sample_every=3):
        """
        Calculate vertical alpha parameter α_z = T^θ_φ/P
        Compare with radial α_r for stress anisotropy
        
        FIXED: Only calculates in torus region
        """
        print("=== VERTICAL ALPHA PARAMETER ANALYSIS ===")
        
        if quasi_start_time is None:
            analysis_files = dump_files
        else:
            analysis_files = []
            for dump_file in dump_files:
                try:
                    self.load_data("gdump", dump_file)
                    if hs.t >= quasi_start_time:
                        analysis_files.append(dump_file)
                except:
                    continue
        
        alpha_z_sum = None
        alpha_r_sum = None
        count = 0
        
        for dump_file in analysis_files[::sample_every]:
            try:
                self.load_data("gdump", dump_file)
                hs.aux()  # Calculate stress tensor
                
                if not hasattr(hs, 'Tud'):
                    continue
                
                # Stress tensor components
                T_z_phi = hs.Tud[2, 3].squeeze()  # T^θ_φ (vertical stress)
                T_r_phi = hs.Tud[1, 3].squeeze()  # T^r_φ (radial stress)
                
                # Pressure
                pg = (hs.gam - 1) * hs.ug.squeeze()
                rho = hs.rho.squeeze()
                r = hs.r.squeeze()
                
                # CRITICAL FIX: Torus mask
                mask = (rho > 0.1) & (r > 6) & (r < 50)
                
                if mask.sum() == 0:
                    continue
                
                # Initialize alpha arrays
                alpha_z = np.zeros_like(T_z_phi)
                alpha_r = np.zeros_like(T_r_phi)
                
                # Calculate ONLY in torus
                alpha_z[mask] = -T_z_phi[mask] / (pg[mask] + 1e-20)
                alpha_r[mask] = -T_r_phi[mask] / (pg[mask] + 1e-20)
                
                # Time averaging
                if alpha_z_sum is None:
                    alpha_z_sum = alpha_z.copy()
                    alpha_r_sum = alpha_r.copy()
                else:
                    alpha_z_sum += alpha_z
                    alpha_r_sum += alpha_r
                
                count += 1
                
            except Exception as e:
                print(f"Error in vertical alpha calculation: {e}")
                continue
        
        if count > 0:
            alpha_z_avg = alpha_z_sum / count
            alpha_r_avg = alpha_r_sum / count
            
            # Get mask for final statistics
            mask = (rho > 0.1) & (r > 6) & (r < 50)
            
            # Calculate averages ONLY in torus
            alpha_z_mean = alpha_z_avg[mask].mean()
            alpha_r_mean = alpha_r_avg[mask].mean()
            anisotropy = alpha_z_mean / (alpha_r_mean + 1e-20)
            
            print(f"\nVertical alpha calculated from {count} snapshots:")
            print(f"  α_z (vertical) = {alpha_z_mean:.3f}")
            print(f"  α_r (radial) = {alpha_r_mean:.3f}")
            print(f"  Anisotropy α_z/α_r = {anisotropy:.2f}")
            
            results = {
                'alpha_z': alpha_z_avg,
                'alpha_r': alpha_r_avg,
                'alpha_ratio': alpha_z_avg / (alpha_r_avg + 1e-20),
                'count': count,
                'mask': mask
            }
            
            return results
        else:
            return None
    
    def create_alpha_evolution_movie(self, dump_files, quasi_start_time=None, 
                                    output_file="torus_alpha_evolution.mp4", fps=5):
        """Create movie showing alpha parameter evolution"""
        print("Creating alpha parameter evolution movie...")
        
        if quasi_start_time is None:
            movie_files = dump_files
        else:
            movie_files = []
            for dump_file in dump_files:
                try:
                    self.load_data("gdump", dump_file)
                    if hs.t >= quasi_start_time:
                        movie_files.append(dump_file)
                except:
                    continue
        
        # Sample files for movie
        sampled_files = movie_files[::max(1, len(movie_files)//100)]
        print(f"Using {len(sampled_files)} frames for alpha movie")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            hs.aux()
            
            if not hasattr(hs, 'Tud'):
                return
            
            # Calculate alpha
            T_r_phi = hs.Tud[1, 3].squeeze()
            pg = (hs.gam - 1) * hs.ug.squeeze()
            alpha = T_r_phi / (pg + 1e-20)
            
            # Get coordinates
            r = hs.r.squeeze()
            h = hs.h.squeeze()
            x = r * np.sin(h)
            z = r * np.cos(h)
            
            # Plot 1: 2D alpha map
            im1 = ax1.pcolormesh(x, z, alpha, cmap='RdBu_r', 
                                vmin=-0.2, vmax=0.2, shading='auto')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Z')
            ax1.set_title(f'Alpha Parameter (t = {hs.t:.1f})')
            ax1.set_aspect('equal')
            ax1.set_xlim(-50, 50)
            ax1.set_ylim(-25, 25)
            
            # Plot 2: Radial profile
            if alpha.ndim > 1:
                r_1d = r[:, 0] if r.ndim > 1 else r
                alpha_avg = alpha.mean(axis=1)
            else:
                r_1d = r
                alpha_avg = alpha
            
            ax2.semilogx(r_1d, alpha_avg, 'b-', linewidth=3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Radius')
            ax2.set_ylabel('Alpha Parameter')
            ax2.set_title('Radial Alpha Profile')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.2, 0.2)
            
            return im1,
        
        ani = animation.FuncAnimation(fig, update, frames=len(sampled_files),
                                     blit=False, interval=200, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"Saved alpha evolution movie: {output_path}")
        except Exception as e:
            print(f"Error saving alpha movie: {e}")
        
        plt.close(fig)


def get_dump_files(dump_folder="dumps", pattern="dump[0-9][0-9][0-9]*"):
    """Get sorted list of dump files, ordered by numeric index."""
    dump_files = glob.glob(os.path.join(dump_folder, pattern))
    dump_files = [os.path.basename(f) for f in dump_files]
    dump_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
    return dump_files


def main():
    """Main analysis function for torus problem"""
    parser = argparse.ArgumentParser(description="Analyze Magnetized Torus Problem")
    parser.add_argument("--output", type=str, default="./torus_analysis", 
                       help="Output directory for plots and movies")
    parser.add_argument("--sample", type=int, default=5, 
                       help="Sample every N dump files for analysis")
    parser.add_argument("--movie", action="store_true", 
                       help="Create density evolution movie")
    parser.add_argument("--alpha-movie", action="store_true",
                       help="Create alpha parameter evolution movie")
    parser.add_argument("--mri-only", action="store_true",
                       help="Only analyze MRI resolution")
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TorusAnalysis(output_dir=args.output)
    
    # Get dump files
    dump_files = get_dump_files()
    if not dump_files:
        print("No dump files found! Make sure torus simulation has run.")
        return
    
    print(f"Found {len(dump_files)} dump files")
    print("Analyzing magnetized torus problem...")
    
    # 1. MRI Resolution Analysis (from initial conditions)
    print("\n" + "="*50)
    print("STEP 1: MRI RESOLUTION ANALYSIS")
    print("="*50)
    mri_results = analyzer.analyze_mri_resolution()
    
    if args.mri_only:
        print("MRI resolution analysis complete.")
        return
    
    # 2. Time Evolution Analysis
    print("\n" + "="*50)
    print("STEP 2: TIME EVOLUTION & QUASI-STATIONARY IDENTIFICATION")
    print("="*50)
    evolution_results = analyzer.analyze_time_evolution(dump_files, sample_every=args.sample)
    
    quasi_time = evolution_results.get('quasi_stationary_start', None)
    if quasi_time:
        print(f"Using quasi-stationary period starting at t = {quasi_time:.1f}")
    else:
        print("No clear quasi-stationary period identified, using all data")
    
    # 3. Alpha Parameter Analysis
    print("\n" + "="*50)
    print("STEP 3: ALPHA PARAMETER ANALYSIS")
    print("="*50)
    alpha_results = analyzer.calculate_alpha_parameter(dump_files, quasi_time, sample_every=args.sample)
    
    # 4. Angular Velocity Analysis
    print("\n" + "="*50)
    print("STEP 4: ANGULAR VELOCITY ANALYSIS")
    print("="*50)
    omega_results = analyzer.analyze_angular_velocity(dump_files, quasi_time, sample_every=args.sample)
    
    # 5. Vertical Alpha Analysis
    print("\n" + "="*50)
    print("STEP 5: VERTICAL ALPHA PARAMETER")
    print("="*50)
    vertical_alpha_results = analyzer.analyze_vertical_alpha(dump_files, quasi_time, sample_every=args.sample)
    
    # 6. Sonic Surface Analysis (final snapshot)
    print("\n" + "="*50)
    print("STEP 6: SONIC SURFACE ANALYSIS")
    print("="*50)
    sonic_results = analyzer.find_sonic_surfaces(dump_files[-1])
    
    # 7. Create comprehensive plots
    print("\n" + "="*50)
    print("STEP 7: CREATING COMPREHENSIVE ANALYSIS PLOTS")
    print("="*50)
    analyzer.plot_comprehensive_analysis(evolution_results, alpha_results, omega_results, mri_results)
    
    # 8. Create movies if requested
    if args.movie:
        print("\n" + "="*50)
        print("STEP 8: CREATING DENSITY EVOLUTION MOVIE")
        print("="*50)
        analyzer.create_density_movie(dump_files)
    
    if args.alpha_movie:
        print("\n" + "="*50)
        print("STEP 9: CREATING ALPHA PARAMETER MOVIE")
        print("="*50)
        analyzer.create_alpha_evolution_movie(dump_files, quasi_time)
    
    # 9. Print final summary
    print("\n" + "="*60)
    print("TORUS ANALYSIS COMPLETE - FINAL SUMMARY")
    print("="*60)
    
    if mri_results:
        print(f"MRI Resolution: {mri_results['avg_resolution']:.1f} cells/wavelength")
        print(f"Assessment: {mri_results['assessment']}")
    
    if alpha_results:
        # FIXED: Use masked values, not entire array
        if 'mask' in alpha_results:
            mask = alpha_results['mask']
            alpha_mean = alpha_results['alpha_total'][mask].mean()
            alpha_mag = alpha_results['alpha_magnetic'][mask].mean()
            alpha_rey = alpha_results['alpha_reynolds'][mask].mean()
        else:
            alpha_mean = np.nanmean(alpha_results['alpha_total'])
            alpha_mag = np.nanmean(alpha_results['alpha_magnetic'])
            alpha_rey = np.nanmean(alpha_results['alpha_reynolds'])
        print(f"\nAlpha Parameters:")
        print(f"  Total α = {alpha_mean:.3f}")
        print(f"  Magnetic α = {alpha_mag:.3f}")
        print(f"  Reynolds α = {alpha_rey:.3f}")
        print(f"  Magnetic dominance: {abs(alpha_mag)/(abs(alpha_mag)+abs(alpha_rey))*100:.1f}%")
    
    if vertical_alpha_results:
        # FIXED: Use masked values
        if 'mask' in vertical_alpha_results:
            mask = vertical_alpha_results['mask']
            alpha_z_mean = vertical_alpha_results['alpha_z'][mask].mean()
            alpha_r_mean = vertical_alpha_results['alpha_r'][mask].mean()
        else:
            alpha_z_mean = np.nanmean(vertical_alpha_results['alpha_z'])
            alpha_r_mean = np.nanmean(vertical_alpha_results['alpha_r'])
        print(f"\nVertical vs Radial Stress:")
        print(f"  α_z (vertical) = {alpha_z_mean:.3f}")
        print(f"  α_r (radial) = {alpha_r_mean:.3f}")
        print(f"  Anisotropy α_z/α_r = {alpha_z_mean/alpha_r_mean:.2f}")
    
    if omega_results:
        # FIXED: Use masked values
        if 'mask' in omega_results:
            mask = omega_results['mask']
            omega_ratio = omega_results['omega_ratio'][mask].mean()
        else:
            omega_ratio = np.nanmean(omega_results['omega_ratio'])
        print(f"\nAngular Velocity:")
        print(f"  Ω/Ω_Keplerian = {omega_ratio:.2f}")
        if omega_ratio < 0.8:
            print("  → Sub-Keplerian rotation (expected for accretion)")
        elif omega_ratio > 1.2:
            print("  → Super-Keplerian rotation (unusual)")
        else:
            print("  → Near-Keplerian rotation")
    
    if evolution_results:
        total_time = evolution_results['times'][-1] - evolution_results['times'][0]
        print(f"\nTime Evolution:")
        print(f"  Total simulation time: {total_time:.1f}")
        if quasi_time:
            quasi_fraction = (evolution_results['times'][-1] - quasi_time) / total_time
            print(f"  Quasi-stationary period: {quasi_fraction*100:.1f}% of simulation")
    
    print(f"\nPhysical Interpretation:")
    print(f"• MRI drives magnetorotational turbulence")
    print(f"• Alpha parameter quantifies effective viscosity")
    print(f"• Magnetic stresses transport angular momentum")
    print(f"• Turbulence enables efficient accretion")
    
    if mri_results and mri_results['avg_resolution'] < 10:
        print(f"\nWARNING: MRI may be under-resolved!")
        print(f"Consider increasing resolution for better MRI development")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

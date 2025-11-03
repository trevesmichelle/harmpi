import harm_script as hs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import argparse
import glob
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.special import legendre 


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
    
    def detect_field_type(self, dump_file):
        """
        Detect field type using Legendre decomposition (proper method).
        
        This replaces the old heuristic method with proper multipole analysis.
        """
        try:
            self.load_data("gdump", dump_file)
            
            if not hasattr(hs, 'B'):
                return "Unknown"
            
            B_r = hs.B[1].squeeze()
            
            if B_r.ndim < 2:
                return "1D"
            
            # Use Legendre decomposition for proper classification
            try:
                legendre_results = self.analyze_field_multipoles_legendre()
                field_type = legendre_results['field_type']
                
                # Map Legendre field types to simpler names for plot titles
                type_map = {
                    "Regular Monopole": "Monopole",
                    "Pure Dipole": "Dipole",
                    "Split Monopole": "Split Monopole",
                    "Quadrupole": "Quadrupole",
                    "Mixed Multipole": "Mixed/Evolving"
                }
                
                # Also handle "Higher Multipole (l=X)" case
                if "Higher Multipole" in field_type:
                    return "Mixed/Evolving"
                
                return type_map.get(field_type, "Mixed/Evolving")
                
            except Exception as e:
                print(f"Legendre classification failed, using fallback: {e}")
                # Fallback to simple heuristic if Legendre fails
                return self._detect_field_type_fallback(B_r)
                
        except Exception as e:
            print(f"Could not detect field type: {e}")
            return "Unknown"


    def _detect_field_type_fallback(self, B_r):
        """
        Fallback heuristic if Legendre decomposition fails.
        More lenient thresholds than before.
        """
        n_theta = B_r.shape[1]
        north_quarter = n_theta // 4
        south_quarter = 3 * n_theta // 4
        equator_idx = n_theta // 2
        
        # Average field in northern and southern hemispheres
        B_r_north = B_r[:10, :north_quarter].mean()
        B_r_south = B_r[:10, south_quarter:].mean()
        
        # Check for field reversal (dipole signature)
        field_reversal = (B_r_north * B_r_south < 0)
        
        # Check pole/equator ratio
        B_theta = hs.B[2].squeeze()
        pole_field = np.sqrt(B_r[:10, 0]**2 + B_theta[:10, 0]**2).mean()
        equator_field = np.sqrt(B_r[:10, equator_idx]**2 + B_theta[:10, equator_idx]**2).mean()
        topology_ratio = pole_field / equator_field if equator_field > 0 else 1
        
        # More lenient classification logic
        if field_reversal:
            return "Dipole"
        elif topology_ratio > 1.2:  # Reduced from 1.5
            return "Monopole"
        else:
            return "Mixed/Evolving"

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
    
    
    def extract_omega_at_horizon(self):
        """
        Extract ΩF/ΩH precisely at the black hole horizon.
        
        Physical Principle:
        ------------------
        Frame dragging causes magnetic field lines near the horizon to rotate
        with angular velocity ΩF. The BZ mechanism predicts:
        
            ΩF/ΩH = 0.5 (for monopole field)
        
        where ΩH = a/(2r_h) is the horizon's angular velocity.
        
        CRITICAL: Must measure at r = r_horizon, NOT domain-averaged!
        Domain averaging dilutes the signal and gives incorrect values.
        
        Method:
        ------
        1. Calculate exact horizon radius: r_h = 1 + √(1 - a²)
        2. Find grid cell closest to r_h
        3. Extract ΩF(θ) at that radius
        4. Compute ΩF/ΩH ratio with error bars
        
        Parameters:
        ----------
        i_horizon_approx : int
            Approximate grid index for horizon (for validation only)
        
        Returns:
        -------
        dict : {
            'omega_ratio': ndarray,  # ΩF/ΩH vs θ
            'theta': ndarray,        # θ coordinates
            'horizon_radius': float, # r_h in geometric units
            'avg_omega_ratio': float,
            'std_omega_ratio': float,
            'deviation_percent': float  # |measured - 0.5| / 0.5 × 100%
        }
        
        Notes:
        -----
        - Typical deviation should be < 5% for well-resolved simulations
        - Deviation > 50% indicates a problem (check grid resolution)
        - Angular variation should be small for monopole (< 10%)
        """
        # Compute auxiliary quantities (includes omegaf2)
        try:
            hs.aux()
        except Exception as e:
            raise RuntimeError(f"hs.aux() failed to compute omega_f: {e}")
        
        if not hasattr(hs, 'omegaf2'):
            raise RuntimeError("omegaf2 not computed. aux() may have failed.")
        
        # Black hole parameters
        a = hs.a
        rhor = 1 + (1 - a**2)**0.5  # Horizon radius
        omega_h = a / (2 * rhor)    # Black hole angular velocity
        
        # Get omega_f and radial coordinate
        omega_f = hs.omegaf2.squeeze()
        r = hs.r.squeeze()
        
        # Determine if 1D or 2D
        is_1d = (r.ndim == 1)
        
        if is_1d:
            # 1D case: omega_f is 1D array vs radius
            horizon_idx = np.abs(r - rhor).argmin()
            
            # Validate we found a reasonable horizon
            if np.abs(r[horizon_idx] - rhor) > 0.5:
                raise ValueError(f"Horizon not well-resolved: closest r = {r[horizon_idx]:.3f}, r_h = {rhor:.3f}")
            
            omega_f_horizon = float(omega_f[horizon_idx])
            omega_ratio_horizon = omega_f_horizon / omega_h if omega_h != 0 else 0.0
            
            return {
                'omega_f_horizon': omega_f_horizon,
                'omega_ratio_horizon': omega_ratio_horizon,
                'omega_h': omega_h,
                'horizon_radius': rhor,
                'horizon_idx': horizon_idx,
                'theta': None,
                'is_1d': True,
                'mean_omega_ratio': omega_ratio_horizon,
                'std_omega_ratio': 0.0
            }
        
        else:
            # 2D case: omega_f is 2D array (r, theta)
            r_1d = r[:, 0]  # Extract radial coordinate
            horizon_idx = np.abs(r_1d - rhor).argmin()
            
            # Validate horizon
            if np.abs(r_1d[horizon_idx] - rhor) > 0.5:
                raise ValueError(f"Horizon not well-resolved: closest r = {r_1d[horizon_idx]:.3f}, r_h = {rhor:.3f}")
            
            # Extract omega_f at horizon as function of theta
            omega_f_horizon = omega_f[horizon_idx, :]
            
            # Get theta coordinate at horizon
            h = hs.h.squeeze()
            theta_horizon = h[horizon_idx, :] if h.ndim > 1 else h
            
            # Calculate ratio
            omega_ratio_horizon = omega_f_horizon / omega_h if omega_h != 0 else omega_f_horizon * 0
            
            # Statistics
            mean_ratio = float(np.mean(omega_ratio_horizon))
            std_ratio = float(np.std(omega_ratio_horizon))
            
            return {
                'omega_f_horizon': omega_f_horizon,
                'omega_ratio_horizon': omega_ratio_horizon,
                'omega_h': omega_h,
                'horizon_radius': rhor,
                'horizon_idx': horizon_idx,
                'theta': theta_horizon,
                'is_1d': False,
                'mean_omega_ratio': mean_ratio,
                'std_omega_ratio': std_ratio
            }

    def calculate_hemisphere_flux(self, i_horizon_approx=5):
        """
        Calculate magnetic flux through northern and southern hemispheres.
        
        Physical Principle:
        ------------------
        For a surface at radius r with area element dA = r² sin(θ) dθ dφ:
        
            Φ = ∫∫ B_r · dA = 2π r² ∫ B_r(θ) sin(θ) dθ
        
        Split into hemispheres:
            Φ_north = 2π r_h² ∫₀^(π/2) B_r(θ) sin(θ) dθ
            Φ_south = 2π r_h² ∫_(π/2)^π B_r(θ) sin(θ) dθ
        
        Field Classification:
        --------------------
        - Monopole: Φ_N and Φ_S have SAME sign → Φ_total ≠ 0
        - Dipole:   Φ_N and Φ_S have OPPOSITE signs → Φ_total ≈ 0
        - Split monopole: Φ_N ≠ 0, Φ_S ≈ 0 (or vice versa)
        
        This is independent validation of the zero-crossing method!
        
        Parameters:
        ----------
        i_horizon_approx : int
            Approximate grid index near horizon (for validation)
        
        Returns:
        -------
        dict : {
            'flux_north': float,         # Flux through northern hemisphere
            'flux_south': float,         # Flux through southern hemisphere  
            'flux_total': float,         # Total flux (should ≈ 0 for dipole)
            'same_sign': bool,           # True = monopole-like, False = dipole-like
            'field_type': str,           # 'Monopole-like' or 'Dipole-like'
            'theta': ndarray,            # Angular coordinates
            'B_r_horizon': ndarray,      # B_r(θ) at horizon
            'horizon_radius': float,     # r_h in geometric units
            'integrand_north': ndarray,  # For debugging
            'integrand_south': ndarray   # For debugging
        }
        
        Notes:
        -----
        - Uses trapezoidal integration (numpy.trapezoid)
        - Finds horizon radius exactly: r_h = 1 + √(1 - a²)
        - Extracts B_r at the horizon (not domain-averaged)
        - Sign convention: B_r > 0 means field points outward
        
        Example:
        -------
        >>> analyzer = MagnetizedAnalysis()
        >>> analyzer.load_data("gdump", "dump999")
        >>> flux_data = analyzer.calculate_hemisphere_flux()
        >>> print(f"Field type: {flux_data['field_type']}")
        >>> print(f"Φ_north = {flux_data['flux_north']:.3f}")
        >>> print(f"Φ_south = {flux_data['flux_south']:.3f}")
        """
        
        # Check if data is loaded
        if not hasattr(hs, 'B'):
            raise ValueError("No magnetic field data loaded. Call load_data() first.")
        
        # Get magnetic field components
        B_r = hs.B[1].squeeze()  # Radial component
        
        # Get coordinates
        if hasattr(hs, 'r') and hasattr(hs, 'h'):
            r_2d = hs.r.squeeze()
            theta_2d = hs.h.squeeze()  # h is theta in harm_script
        else:
            raise ValueError("Coordinate arrays not available")
        
        # Calculate exact horizon radius
        a = hs.a  # Black hole spin
        rhor = 1 + np.sqrt(1 - a**2)
        
        # Find horizon in grid
        if r_2d.ndim == 2:
            # 2D case
            r_1d = r_2d[:, 0]  # Radial coordinate (first column)
            horizon_idx = np.abs(r_1d - rhor).argmin()
            
            # Extract at horizon
            B_r_horizon = B_r[horizon_idx, :]
            theta_horizon = theta_2d[horizon_idx, :]
        else:
            # 1D case
            horizon_idx = np.abs(r_2d - rhor).argmin()
            B_r_horizon = B_r[horizon_idx]
            
            # For 1D, assume theta from 0 to π
            if hasattr(hs, 'h'):
                theta_horizon = hs.h.squeeze()
            else:
                # Create uniform theta grid if not available
                n_theta = len(B_r_horizon) if hasattr(B_r_horizon, '__len__') else 1
                theta_horizon = np.linspace(0, np.pi, n_theta)
        
        # Ensure theta is 1D array
        if theta_horizon.ndim > 1:
            theta_horizon = theta_horizon.flatten()
        if B_r_horizon.ndim > 1:
            B_r_horizon = B_r_horizon.flatten()
        
        # Split into hemispheres
        n_theta = len(theta_horizon)
        
        # Find equator (θ = π/2)
        equator_idx = np.abs(theta_horizon - np.pi/2).argmin()
        
        # Northern hemisphere: θ ∈ [0, π/2]
        theta_north = theta_horizon[:equator_idx+1]
        B_r_north = B_r_horizon[:equator_idx+1]
        
        # Southern hemisphere: θ ∈ [π/2, π]
        theta_south = theta_horizon[equator_idx:]
        B_r_south = B_r_horizon[equator_idx:]
        
        # Calculate integrand: B_r(θ) × sin(θ) × r_h²
        integrand_north = B_r_north * np.sin(theta_north) * rhor**2
        integrand_south = B_r_south * np.sin(theta_south) * rhor**2
        
        # Integrate over each hemisphere
        # Φ = 2π × ∫ integrand dθ
        flux_north = 2 * np.pi * np.trapezoid(integrand_north, theta_north)
        flux_south = 2 * np.pi * np.trapezoid(integrand_south, theta_south)
        
        # Total flux
        flux_total = flux_north + flux_south
        
        # Determine field type
        # If same sign: monopole-like (flux through both hemispheres in same direction)
        # If opposite signs: dipole-like (flux in, then out)
        same_sign = (flux_north * flux_south > 0)
        
        if same_sign:
            field_type = "Monopole-like"
        else:
            field_type = "Dipole-like"
        
        # Package results
        results = {
            'flux_north': flux_north,
            'flux_south': flux_south,
            'flux_total': flux_total,
            'same_sign': same_sign,
            'field_type': field_type,
            'theta': theta_horizon,
            'B_r_horizon': B_r_horizon,
            'horizon_radius': rhor,
            'horizon_idx': horizon_idx,
            'integrand_north': integrand_north,
            'integrand_south': integrand_south,
            'equator_idx': equator_idx
        }
        
        return results

    def calculate_energy_flux_from_primitives(self):
        """
        Calculate energy flux -T^r_t from primitive variables.
        
        This is DIFFERENT from magnetic flux! Energy flux tells you where
        Blandford-Znajek power extraction occurs.
        
        Physical Formula:
        ----------------
        Energy flux = -T^r_t = -√g (ρh + b²) u^r u_t
        
        Where:
        - ρ = rest-mass density (hs.rho)
        - h = specific enthalpy = 1 + Γp/ρ
        - Γ = adiabatic index (4/3 for radiation-dominated)
        - p = pressure = (Γ-1) × u  [calculated from internal energy]
        - u = internal energy density (hs.ug)
        - b² = magnetic field energy density (hs.bsq)
        - u^r = radial 4-velocity component (hs.uu[1])
        - u_t = timelike covariant 4-velocity (hs.ud[0])
        - √g = metric determinant (hs.gdet)
        
        Interpretation:
        --------------
        - Positive: Energy flowing outward (extraction from black hole)
        - Negative: Energy flowing inward (accretion)
        - Used in stagnation surface visualization (Panel 2)
        
        Returns:
        -------
        ndarray : Energy flux as 2D array matching grid shape
        
        Raises:
        ------
        ValueError : If required primitive variables not loaded
        
        Example:
        -------
        >>> analyzer = MagnetizedAnalysis()
        >>> analyzer.load_data("gdump", "dump999")
        >>> energy_flux = analyzer.calculate_energy_flux_from_primitives()
        >>> print(f"Max extraction: {energy_flux.max():.3e}")
        """
        # Check if data is loaded - pg (gas pressure) is already computed
        required_fields = ['rho', 'pg', 'bsq', 'uu', 'ud', 'gdet']
        missing_fields = []
        for field in required_fields:
            if not hasattr(hs, field):
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Required fields not available: {', '.join(missing_fields)}. Call load_data() first.")
        
        # Get primitive variables
        rho = hs.rho.squeeze()
        p = hs.pg.squeeze()  # Gas pressure (already computed by harm_script)
        bsq = hs.bsq.squeeze()
        u_r = hs.uu[1].squeeze()
        u_t = hs.ud[0].squeeze()
        gdet = hs.gdet.squeeze()
        
        # NOTE: We use pg (gas pressure) directly rather than calculating from internal energy
        # This is more accurate as it's what the simulation actually uses
        gam = 4./3.  # Adiabatic index
        
        # Calculate specific enthalpy
        # h = 1 + Γp/ρ where Γ = 4/3
        # Avoid division by zero
        h = np.where(rho > 0, 1 + gam * p / rho, 1.0)
        
        # Calculate energy flux: -T^r_t
        # This represents energy flow in the radial direction
        energy_flux = -gdet * (rho * h + bsq) * u_r * u_t
        
        return energy_flux

    def print_hemisphere_flux_summary(self, flux_data):
        """
        Print a human-readable summary of hemisphere flux results.
        
        Parameters:
        ----------
        flux_data : dict
            Output from calculate_hemisphere_flux()
        """
        print("\n" + "="*60)
        print("HEMISPHERE FLUX ANALYSIS")
        print("="*60)
        
        print(f"\nHorizon radius: r_h = {flux_data['horizon_radius']:.3f} r_g")
        print(f"Black hole spin: a = {hs.a:.3f}")
        
        print(f"\nNorthern hemisphere flux: Φ_N = {flux_data['flux_north']:+.4e}")
        print(f"Southern hemisphere flux: Φ_S = {flux_data['flux_south']:+.4e}")
        print(f"Total flux:               Φ_T = {flux_data['flux_total']:+.4e}")
        
        print(f"\nFlux sign comparison:")
        if flux_data['same_sign']:
            print("  ✓ Φ_N and Φ_S have SAME sign → Monopole-like field")
            print("  → Net flux through horizon ≠ 0")
        else:
            print("  ✓ Φ_N and Φ_S have OPPOSITE signs → Dipole-like field")
            print("  → Net flux through horizon ≈ 0")
        
        print(f"\nField classification: {flux_data['field_type']}")
        
        # Symmetry check
        asymmetry = abs(abs(flux_data['flux_north']) - abs(flux_data['flux_south']))
        avg_flux = (abs(flux_data['flux_north']) + abs(flux_data['flux_south'])) / 2
        if avg_flux > 0:
            asymmetry_percent = asymmetry / avg_flux * 100
            print(f"Hemisphere asymmetry: {asymmetry_percent:.1f}%")
            
            if asymmetry_percent < 5:
                print("  → Highly symmetric field")
            elif asymmetry_percent < 20:
                print("  → Moderately symmetric field")
            else:
                print("  → Asymmetric field (may be evolving or split monopole)")
        
        print("="*60)

    def analyze_field_multipoles_legendre(self, B_r_horizon=None, theta=None, max_order=10):
        """
        Analyze magnetic field topology using Legendre polynomial decomposition.
        
        This is the PROPER way to identify multipoles, as explained:
        "The way to identify multipoles in the magnetic field is by decomposing into 
        harmonic functions. When there is cylindrical symmetry, the harmonic functions 
        become Legendre polynomials."
        
        Physical Basis:
        --------------
        For axisymmetric fields, the poloidal magnetic field can be expanded as:
        
            B_r(θ) = Σ_l a_l P_l(cos(θ))
        
        where P_l are Legendre polynomials and a_l are expansion coefficients.
        
        Key Insights:
        -------------------------
        - **Regular Monopole**: Only a_0 ≠ 0 (constant field)
          → l=0 dominates, all other coefficients ≈ 0
        
        - **Pure Dipole**: Only a_1 ≠ 0  
          → l=1 dominates, field ∝ cos(θ)
        
        - **Split Monopole**: 
          → Even Legendre polynomials (l=0,2,4,...) contribute one way
          → Odd Legendre polynomials (l=1,3,5,...) contribute differently
          → This distinguishes it from dipole (which has only l=1)
        
        Why This Method is Superior to Zero-Crossing:
        --------------------------------------------
        Zero-crossing detection CANNOT distinguish split monopole from dipole 
        because both vanish at the same points (equator). Legendre decomposition 
        uses the FULL angular distribution, capturing the complete field structure.
        
        Parameters:
        ----------
        B_r_horizon : ndarray, optional
            Radial field component at horizon vs θ
        theta : ndarray, optional  
            Angular coordinates (0 to π)
        max_order : int
            Maximum Legendre polynomial order to compute (default: 10)
        
        Returns:
        -------
        dict : {
            'coefficients': ndarray,           # a_l for l=0,1,2,...,max_order
            'normalized_coefficients': ndarray,# Normalized to l=0 (for monopole) or max coeff
            'dominant_order': int,             # l with largest |a_l|
            'field_type': str,                 # Classification
            'monopole_strength': float,        # |a_0|
            'dipole_strength': float,          # |a_1|  
            'even_power': float,               # Σ a_l² for even l
            'odd_power': float,                # Σ a_l² for odd l
            'even_odd_ratio': float,           # even_power / odd_power
            'confidence': str,
            'B_r_reconstructed': ndarray,      # Reconstructed field from expansion
            'theta': ndarray
        }
        
        Classification Logic:
        --------------------
        - **Regular Monopole**: a_0 >> all other coefficients (>10x)
        - **Pure Dipole**: a_1 >> all others, even_power << odd_power
        - **Split Monopole**: Significant even AND odd contributions, even_power ~ odd_power
        - **Quadrupole**: a_2 dominates
        - **Mixed**: No clear dominant order
        
        Example:
        -------
        >>> analyzer = MagnetizedAnalysis()
        >>> analyzer.load_data("gdump", "dump999")  
        >>> multipole = analyzer.analyze_field_multipoles_legendre()
        >>> print(f"Field type: {multipole['field_type']}")
        >>> print(f"Dominant order: l={multipole['dominant_order']}")
        >>> print(f"Even/Odd ratio: {multipole['even_odd_ratio']:.3f}")
        """
        
        # Get B_r at horizon if not provided
        if B_r_horizon is None or theta is None:
            flux_data = self.calculate_hemisphere_flux()
            B_r_horizon = flux_data['B_r_horizon']
            theta = flux_data['theta']
        
        # Ensure 1D arrays
        B_r_horizon = np.atleast_1d(B_r_horizon).flatten()
        theta = np.atleast_1d(theta).flatten()
        
        # Convert theta to x = cos(theta) for Legendre polynomials
        # θ ∈ [0,π] → x ∈ [-1,1]
        x = np.cos(theta)
        
        # Compute Legendre coefficients by projection
        # a_l = (2l+1)/2 * ∫ B_r(x) P_l(x) dx
        #
        # Numerically: use trapezoidal integration over x
        coefficients = np.zeros(max_order + 1)
        
        for l in range(max_order + 1):
            # Get Legendre polynomial P_l(x)
            P_l = legendre(l)
            P_l_values = P_l(x)
            
            # Integrate B_r(x) * P_l(x) over x ∈ [-1,1]
            # Note: dx/dθ = -sin(θ), so we need to account for Jacobian
            # But since we're working in x = cos(θ), we integrate directly in x
            
            # Sort by x for proper integration (x should be decreasing as θ increases)
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            B_r_sorted = B_r_horizon[sort_idx]
            P_l_sorted = P_l_values[sort_idx]
            
            # Trapezoidal integration
            integrand = B_r_sorted * P_l_sorted
            integral = np.trapezoid(integrand, x_sorted)
            
            # Normalization factor for Legendre polynomials
            coefficients[l] = (2*l + 1) / 2.0 * integral
        
        # Calculate power in even and odd orders
        even_indices = np.arange(0, max_order + 1, 2)
        odd_indices = np.arange(1, max_order + 1, 2)
        
        even_power = np.sum(coefficients[even_indices]**2)
        odd_power = np.sum(coefficients[odd_indices]**2)
        
        # Avoid division by zero
        if odd_power > 0:
            even_odd_ratio = even_power / odd_power
        else:
            even_odd_ratio = np.inf if even_power > 0 else 1.0
        
        # Find dominant order
        abs_coefficients = np.abs(coefficients)
        dominant_order = np.argmax(abs_coefficients)
        
        # Normalize coefficients (relative to dominant or monopole)
        if abs_coefficients[0] > 0:
            # Normalize to monopole component if present
            normalized_coefficients = coefficients / abs_coefficients[0]
        elif abs_coefficients[dominant_order] > 0:
            # Otherwise normalize to dominant component
            normalized_coefficients = coefficients / abs_coefficients[dominant_order]
        else:
            normalized_coefficients = coefficients
        
        # ============================================================
        # IMPROVED CLASSIFICATION LOGIC (INTEGRATED VERSION)
        # ============================================================

        abs_coefficients = np.abs(coefficients)
        dominant_order = np.argmax(abs_coefficients)

        a0 = abs_coefficients[0]
        a1 = abs_coefficients[1] if len(abs_coefficients) > 1 else 0
        a2 = abs_coefficients[2] if len(abs_coefficients) > 2 else 0

        # Compute dominance measures
        sorted_coeffs = np.sort(abs_coefficients)[::-1]
        second_largest = sorted_coeffs[1] if len(sorted_coeffs) > 1 else 0
        third_largest = sorted_coeffs[2] if len(sorted_coeffs) > 2 else 0

        strong_dominance = 3.0
        moderate_dominance = 1.5

        # Total power
        total_power = np.sum(abs_coefficients**2)
        power_fraction_threshold = 0.6
        dominant_power_fraction = abs_coefficients[dominant_order]**2 / total_power if total_power > 0 else 0

        # ============================================================
        # CLASSIFICATION
        # ============================================================

        if dominant_order == 0 and dominant_power_fraction > power_fraction_threshold:
            field_type = "Regular Monopole"
            confidence = "High"
            description = f"l=0 (monopole) dominates: a₀={a0:.3e}, {dominant_power_fraction*100:.1f}% of power"

        elif dominant_order == 0 and a0 > strong_dominance * second_largest and even_odd_ratio > 3.0:
            field_type = "Regular Monopole"
            confidence = "High"
            description = f"l=0 (monopole) dominates: a₀={a0:.3e} >> others, even/odd={even_odd_ratio:.2f}"

        elif dominant_order == 1 and dominant_power_fraction > power_fraction_threshold:
            field_type = "Pure Dipole"
            confidence = "High"
            description = f"l=1 (dipole) dominates: a₁={a1:.3e}, {dominant_power_fraction*100:.1f}% of power"

        elif dominant_order == 1 and a1 > strong_dominance * a0 and a1 > strong_dominance * a2:
            field_type = "Pure Dipole"
            confidence = "High"
            description = f"l=1 (dipole) dominates: a₁={a1:.3e} >> a₀={a0:.3e}"

        elif dominant_order == 1 and a1 > moderate_dominance * max(a0, a2) and even_odd_ratio < 0.5:
            field_type = "Pure Dipole"
            confidence = "Medium"
            description = f"l=1 (dipole) dominates: a₁={a1:.3e}, even/odd={even_odd_ratio:.2f}"

        elif (dominant_order in [0, 1] and 
            0.3 <= even_odd_ratio <= 3.0 and
            a0 > 0.2 * a1 and a1 > 0.2 * a0):
            field_type = "Split Monopole"
            confidence = "High" if 0.5 <= even_odd_ratio <= 2.0 else "Medium"
            description = f"Both even and odd significant: a₀={a0:.3e}, a₁={a1:.3e}, ratio={even_odd_ratio:.2f}"

        elif dominant_order == 2 and a2 > strong_dominance * max(a0, a1):
            field_type = "Quadrupole"
            confidence = "High"
            description = f"l=2 (quadrupole) dominates: a₂={a2:.3e}"

        elif dominant_order > 2 and dominant_power_fraction > 0.4:
            field_type = f"Higher Multipole (l={dominant_order})"
            confidence = "Medium"
            description = f"l={dominant_order} is dominant: {dominant_power_fraction*100:.1f}% of power"

        else:
            field_type = "Mixed Multipole"
            confidence = "Low"
            top_3 = f"a₀={a0:.2e}, a₁={a1:.2e}, a₂={a2:.2e}"
            description = f"No clear dominant: {top_3}, even/odd={even_odd_ratio:.2f}"

        
        # Reconstruct B_r from Legendre expansion (for validation)
        B_r_reconstructed = np.zeros_like(B_r_horizon)
        for l in range(max_order + 1):
            P_l = legendre(l)
            P_l_values = P_l(x)
            B_r_reconstructed += coefficients[l] * P_l_values
        
        # Calculate reconstruction error
        reconstruction_error = np.sqrt(np.mean((B_r_horizon - B_r_reconstructed)**2))
        relative_error = reconstruction_error / (np.max(np.abs(B_r_horizon)) + 1e-10)
        
        # Package results
        results = {
            'coefficients': coefficients,
            'normalized_coefficients': normalized_coefficients,
            'dominant_order': int(dominant_order),
            'field_type': field_type,
            'description': description,
            'confidence': confidence,
            'monopole_strength': float(a0),
            'dipole_strength': float(a1),
            'quadrupole_strength': float(a2),
            'even_power': float(even_power),
            'odd_power': float(odd_power),
            'even_odd_ratio': float(even_odd_ratio),
            'B_r_reconstructed': B_r_reconstructed,
            'B_r_horizon': B_r_horizon,
            'theta': theta,
            'reconstruction_error': float(reconstruction_error),
            'relative_error': float(relative_error)
        }
        
        return results

    def plot_Br_theta_evolution(self, dump_files, times=None, indices=None,
                                show=True, save=True):
        """
        Plot B_r(θ) at horizon for multiple times to show field evolution.
        
        EFFICIENT VERSION: Only loads dumps that will be plotted.
        
        Parameters:
        ----------
        dump_files : list
            Available dump files
        times : list of float, optional
            Approximate times to visualize. If provided, will find closest dumps.
            Requires loading all dumps to build time map (slower).
        indices : list of int, optional
            Direct dump file indices to plot (faster).
            E.g., [0, 50, 99] plots first, middle, last.
        show : bool
            Whether to display plot
        save : bool
            Whether to save plot to file
        
        Returns:
        -------
        fig, axes : matplotlib figure and axes
        
        Notes:
        -----
        For efficiency, prefer using `indices` over `times` when possible.
        
        Examples:
        --------
        # Fast: plot specific dumps
        >>> analyzer.plot_Br_theta_evolution(dumps, indices=[0, 500, 999])
        
        # Slower: search by time (loads all dumps to find closest)
        >>> analyzer.plot_Br_theta_evolution(dumps, times=[0, 50, 100])
        """
        
        # Determine which dumps to plot
        if indices is not None:
            # Fast path: use specified indices directly
            selected_dumps = [dump_files[i] for i in indices]
            actual_times = []
            
            # Load only these dumps to get times
            for dump_file in selected_dumps:
                self.load_data("gdump", dump_file)
                actual_times.append(float(hs.t))
                
        elif times is not None:
            # Slow path: must load all dumps to find closest times
            print(f"  Note: Loading all {len(dump_files)} dumps to find closest times...")
            print(f"  Tip: Use indices=[...] instead for faster plotting")
            
            time_to_dump = {}
            for dump_file in dump_files:
                try:
                    self.load_data("gdump", dump_file)
                    current_time = float(hs.t)
                    time_to_dump[current_time] = (dump_file, current_time)
                except:
                    continue
            
            if not time_to_dump:
                print("Could not find any valid dump files")
                return None, None
            
            # Find closest times
            selected_dumps = []
            actual_times = []
            available_times = sorted(time_to_dump.keys())
            
            for target_time in times:
                closest_time = min(available_times, key=lambda t: abs(t - target_time))
                dump_file, actual_time = time_to_dump[closest_time]
                selected_dumps.append(dump_file)
                actual_times.append(actual_time)
        else:
            # Default: plot first, middle, last
            indices = [0, len(dump_files)//2, len(dump_files)-1]
            selected_dumps = [dump_files[i] for i in indices]
            
            actual_times = []
            for dump_file in selected_dumps:
                self.load_data("gdump", dump_file)
                actual_times.append(float(hs.t))
        
        if not selected_dumps:
            print("No dump files selected")
            return None, None
        
        # Create figure
        n_panels = len(selected_dumps)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 5))
        
        if n_panels == 1:
            axes = [axes]
        
        # Plot each time snapshot
        for i, (dump_file, time) in enumerate(zip(selected_dumps, actual_times)):
            ax = axes[i]
            
            # Load data
            self.load_data("gdump", dump_file)
            
            # Get topology data
            flux_data = self.calculate_hemisphere_flux()
            topo_data = self.analyze_field_multipoles_legendre(
                B_r_horizon=flux_data['B_r_horizon'],
                theta=flux_data['theta']
            )
            
            theta = flux_data['theta']
            B_r = flux_data['B_r_horizon']
            
            # Plot B_r(θ) - clean and simple
            ax.plot(theta, B_r, 'k-', linewidth=2.5, zorder=5, label='Simulation data')
            # Plot reconstructed field from Legendre decomposition (for validation)
            if 'B_r_reconstructed' in topo_data:
                ax.plot(topo_data['theta'], topo_data['B_r_reconstructed'], 
                        'r--', linewidth=2, alpha=0.7, zorder=4, label='Legendre fit')
            # Zero line and equator
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=np.pi/2, color='gray', linestyle=':', alpha=0.3, linewidth=1.5)
            
            # Color-code plot based on field classification
            field_type = topo_data.get('field_type', 'Unknown')
            confidence = topo_data.get('confidence', 'Low')
            
            # Map field type to color
            field_colors = {
                'Regular Monopole': '#2E7D32',  # Green
                'Pure Dipole': '#1565C0',       # Blue
                'Split Monopole': '#C62828',    # Red
                'Quadrupole': '#F57C00',        # Orange
                'Mixed Multipole': '#6A1B9A'    # Purple
            }
            field_color = field_colors.get(field_type, '#424242')
            
            # Clean formatting
            ax.set_xlabel('θ (rad)', fontsize=11)
            if i == 0:
                ax.set_ylabel('B_r  [simulation units]', fontsize=11)
            
            # Title: just time and field type
            ax.set_title(f't = {time:.1f} M\n{topo_data["field_type"]}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.25)
            ax.set_xlim(0, np.pi)
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
            
            # Minimal stats box - just key numbers
            stats_text = f'Field Type: {field_type}\n'
            stats_text += f'Φ_N: {flux_data["flux_north"]:+.2e}\n'
            stats_text += f'Φ_S: {flux_data["flux_south"]:+.2e}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor='gray', alpha=0.9, linewidth=1))
        
        # Simple overall title
        fig.suptitle('Magnetic Field Evolution: B_r(θ) at Horizon', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        if save:
            filename = os.path.join(self.output_dir, "Br_theta_evolution.png")
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Saved B_r(θ) evolution plot: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, axes

    def validate_omega_ratio(self, omega_data, tolerance=0.15):
        """
        Validate that ΩF/ΩH is close to theoretical value of 0.5.
        
        Args:
            omega_data: dict returned by extract_omega_at_horizon()
            tolerance: acceptable fractional deviation (default 15%)
        
        Returns:
            dict with validation results
        """
        mean_ratio = omega_data['mean_omega_ratio']
        std_ratio = omega_data['std_omega_ratio']
        theory = 0.5
        
        deviation = abs(mean_ratio - theory) / theory
        is_valid = deviation <= tolerance
        
        result = {
            'is_valid': is_valid,
            'mean_omega_ratio': mean_ratio,
            'std_omega_ratio': std_ratio,
            'theory': theory,
            'fractional_deviation': deviation,
            'percent_deviation': deviation * 100
        }
        
        # Generate message
        if omega_data['is_1d']:
            msg = f"ΩF/ΩH at horizon = {mean_ratio:.4f} (theory: {theory:.3f}, deviation: {deviation*100:.1f}%)"
        else:
            msg = f"<ΩF/ΩH> at horizon = {mean_ratio:.4f} ± {std_ratio:.4f} (theory: {theory:.3f}, deviation: {deviation*100:.1f}%)"
        
        if is_valid:
            result['message'] = "✓ " + msg
            result['status'] = "VALID"
        elif deviation <= 0.30:  # Within 30%
            result['message'] = "⚠ " + msg + " (outside tolerance but reasonable)"
            result['status'] = "WARNING"
        else:
            result['message'] = "✗ " + msg + " (SIGNIFICANT DEVIATION - CHECK SIMULATION)"
            result['status'] = "ERROR"
        
        return result

    
    def calculate_bz_power_prediction(self, flux_data=None, omega_ratio_avg=None):
        """
        Calculate BZ power prediction
        
        Args:
            flux_data: Magnetic flux data (optional - will compute if None)
            omega_ratio_avg: DEPRECATED - will be computed from extract_omega_at_horizon()
        
        Returns:
            dict with BZ power prediction
        """
        # If no flux_data provided, compute it
        if flux_data is None:
            flux_data = self.calculate_hemisphere_flux()
            if flux_data is None:
                print("Cannot calculate BZ power - no flux data")
                return None
        
        # Get omega ratio from proper horizon extraction
        if omega_ratio_avg is None:
            omega_data = self.extract_omega_at_horizon()
            if omega_data is None:
                print("Cannot calculate BZ power - no omega data")
                return None
            omega_ratio_avg = omega_data['mean_omega_ratio']   # ← CORRECT KEY
        
        # BZ parameters
        a = hs.a
        rhor = flux_data.get('horizon_radius', 1 + np.sqrt(1 - a**2))
        total_flux = flux_data['flux_total']
        
        # BZ power formula: P_BZ = (a²Φ²)/(4πr_H²) × (ΩF/ΩH)²
        bz_power = (a**2 * total_flux**2) / (4 * np.pi * rhor**2) * (omega_ratio_avg)**2
        
        # Concise output (no verbose printing unless debugging)
        if False:  # Set to True for debugging
            print(f"\n=== BZ POWER CALCULATION ===")
            print(f"Black hole spin a = {a:.3f}")
            print(f"Horizon radius r_h = {rhor:.3f}")
            print(f"Magnetic flux Φ = {total_flux:.3e}")
            print(f"Frame dragging ΩF/ΩH = {omega_ratio_avg:.3f}")
            print(f"BZ Power Prediction: P_BZ = {bz_power:.3e}")
        
        return {
            'theoretical_power': bz_power,     # For backward compatibility
            'bz_power_prediction': bz_power,   # Also provide this
            'flux': total_flux,
            'omega_ratio': omega_ratio_avg,
            'spin': a,
            'horizon_radius': rhor
        }
        
    def calculate_total_power_from_simulation(self, results):
        """
        Integrate energy flux to get total power and compare with BZ prediction.
        
        Physical principle: Power = ∫ (Energy Flux) · dA over spherical surface
        This comes from Poynting's theorem: ∂U/∂t + ∇·S = -J·E
        where S is the Poynting vector (energy flux density)
        """
        if not results['power_extraction']:
            print("No power extraction data available")
            return None
            
        print("\n=== TOTAL POWER INTEGRATION ===")
        
        # Use the latest timestep for integration
        latest_power = results['power_extraction'][-1]
        r_coord = latest_power['r_coord']
        energy_flux = latest_power['energy_flux']  # This is already averaged over theta
        time = latest_power['time']
        
        print(f"Integrating power at t = {time:.1f}")
        print(f"Energy flux shape: {np.array(energy_flux).shape}")
        print(f"Radial coordinate shape: {np.array(r_coord).shape}")
        
        # Energy flux is already Er (energy per unit area per unit time)
        # To get total power, integrate over spherical surface: P = ∫ Er * dA
        # For spherical surface: dA = r² dr dΩ, but we want ∫ over surface at fixed r
        # So we need: P(r) = Er(r) * 4πr² (assuming spherical symmetry after theta averaging)
        
        total_powers = np.abs(energy_flux) * 4 * np.pi * r_coord**2
        
        # Find power at different characteristic radii
        horizon_radius = 1.35  # Approximate
        
        # Find indices for different radii of interest
        idx_2rg = np.abs(r_coord - 2.0).argmin()
        idx_5rg = np.abs(r_coord - 5.0).argmin() 
        idx_10rg = np.abs(r_coord - 10.0).argmin()
        idx_horizon = np.abs(r_coord - horizon_radius).argmin()
        
        print(f"\nPower extraction at different radii:")
        print(f"At horizon (r â‰ˆ {r_coord[idx_horizon]:.2f}): P = {total_powers[idx_horizon]:.3e}")
        print(f"At r = {r_coord[idx_2rg]:.1f} r_g: P = {total_powers[idx_2rg]:.3e}")
        print(f"At r = {r_coord[idx_5rg]:.1f} r_g: P = {total_powers[idx_5rg]:.3e}")
        print(f"At r = {r_coord[idx_10rg]:.1f} r_g: P = {total_powers[idx_10rg]:.3e}")
        
        # The power should be approximately constant with radius in the force-free zone
        # Use power measured at ~5 gravitational radii as representative
        measured_power = total_powers[idx_5rg]
        
        print(f"\nRepresentative measured power (at 5 r_g): {measured_power:.3e}")
        
        return {
            'measured_power': measured_power,
            'power_profile': total_powers,
            'radii': r_coord,
            'measurement_radius': r_coord[idx_5rg],
            'time': time
        }
    
    def compare_theory_vs_simulation(self, flux_data, power_data, omega_ratio_avg=None, bz_prediction=None):
        """
        Compare BZ theoretical prediction with simulation measurement
        """
        if flux_data is None or power_data is None:
            print("Cannot compare - missing flux or power data")
            return
            
        print("\n" + "="*50)
        print("BZ THEORY vs SIMULATION COMPARISON") 
        print("="*50)
        
        # Get omega_ratio if not provided
        if omega_ratio_avg is None and bz_prediction is None:
            omega_data = self.extract_omega_at_horizon()
            if omega_data:
                omega_ratio_avg = omega_data['mean_omega_ratio']
        
        # Use existing BZ prediction if provided, otherwise calculate it
        if bz_prediction is not None:
            theoretical_power = bz_prediction.get('bz_power_prediction', 
                                                bz_prediction.get('theoretical_power', 0))
        else:
            bz_pred = self.calculate_bz_power_prediction(flux_data, omega_ratio_avg)
            if bz_pred is None:
                print("Cannot calculate BZ prediction")
                return
            theoretical_power = bz_pred['theoretical_power']
            
        measured_power = power_data['measured_power']
        
        # Calculate agreement
        ratio = measured_power / theoretical_power if theoretical_power != 0 else 0
        percent_difference = abs(ratio - 1) * 100
        
        # CLEANED UP: Just the facts
        print(f"\nResults:")
        print(f"  Theoretical BZ Power: {theoretical_power:.3e}")
        print(f"  Measured Power:       {measured_power:.3e}")
        print(f"  Ratio (Sim/Theory):   {ratio:.2f}")
        print(f"  Deviation:            {percent_difference:.1f}%")
        
        # Simple assessment
        if 0.5 <= ratio <= 2.0:
            agreement = "GOOD"
        elif 0.2 <= ratio <= 5.0:
            agreement = "REASONABLE" 
        else:
            agreement = "ENHANCED" if ratio > 5.0 else "POOR"
        
        print(f"  Assessment:           {agreement}")
        
        return {
            'theoretical_power': theoretical_power,
            'measured_power': measured_power,
            'ratio': ratio,
            'agreement': agreement,
            'percent_difference': percent_difference
        }
    
    def analyze_1d_monopole(self, dump_files, sample_every=1):
        """Complete analysis of 1D monopole problem - FIXED ΩF/ΩH calculation"""
        results = {
            'times': [],
            'sigma_horizon': [],
            'sigma_initial': None,
            'lorentz_factors': [],
            'omega_ratios': [],  # FIXED: Now stores horizon values only
            'omega_ratio_evolution': [],  # FIXED: Full evolution data
            'radial_profiles': {
                'r': None,
                'gamma': [],
                'sigma': [],
                'times': []
            }
        }
        
        print("=== 1D MONOPOLE ANALYSIS ===")
        
        sampled_files = dump_files[::sample_every]
        
        for i, dump_file in enumerate(sampled_files):
            try:
                self.load_data("gdump", dump_file)
                current_time = float(hs.t)
                
                # Calculate key quantities
                sigma = self.calculate_magnetization()
                gamma = self.calculate_lorentz_factor()
                
                # FIXED: Extract ΩF/ΩH at horizon (not domain-averaged)
                try:
                    omega_data = self.extract_omega_at_horizon()
                    omega_ratio_horizon = omega_data['omega_ratio_horizon']
                    
                    # Store for time series
                    results['omega_ratios'].append(omega_ratio_horizon)
                    results['omega_ratio_evolution'].append({
                        'time': current_time,
                        'omega_ratio': omega_ratio_horizon,
                        'horizon_radius': omega_data['horizon_radius'],
                        'horizon_idx': omega_data['horizon_idx']
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not extract ΩF/ΩH for {dump_file}: {e}")
                    omega_ratio_horizon = None
                
                if sigma is not None and gamma is not None:
                    # Extract 1D profiles
                    r_profile = hs.r.squeeze()
                    sigma_profile = sigma.squeeze()
                    gamma_profile = gamma.squeeze()
                    
                    # Store initial magnetization
                    if results['sigma_initial'] is None:
                        horizon_idx = 5  # First few cells near horizon
                        results['sigma_initial'] = np.mean(sigma_profile[:horizon_idx])
                        results['radial_profiles']['r'] = r_profile
                    
                    # Store current state
                    results['times'].append(current_time)
                    results['sigma_horizon'].append(sigma_profile[0])
                    results['lorentz_factors'].append(gamma_profile[-1])
                    
                    # Store profiles for evolution
                    if i % 5 == 0:
                        results['radial_profiles']['gamma'].append(gamma_profile)
                        results['radial_profiles']['sigma'].append(sigma_profile)
                        results['radial_profiles']['times'].append(current_time)
                
                print(f"Processed {dump_file}: t={current_time:.3f}, ΩF/ΩH(horizon)={omega_ratio_horizon:.4f if omega_ratio_horizon else 'N/A'}")
                
            except Exception as e:
                print(f"Error processing {dump_file}: {e}")
                continue
        
        # FIXED: Validate ΩF/ΩH at end
        if results['omega_ratios']:
            final_omega = results['omega_ratios'][-1]
            print(f"\n=== VALIDATION ===")
            print(f"Final ΩF/ΩH at horizon: {final_omega:.4f}")
            print(f"Theoretical prediction: 0.500")
            print(f"Deviation: {abs(final_omega - 0.5)/0.5 * 100:.1f}%")
        
        return results


    def analyze_2d_monopole(self, dump_files, sample_every=5):
        """
        Comprehensive analysis for 2D BZ monopole problems.
        
        Physical Context:
        ---------------
        The BZ mechanism (Blandford-Znajek mechanism) extracts rotational energy 
        from spinning black holes via magnetic fields. Key observables:
        
        1. Frame Dragging (ΩF/ΩH):
        - ΩF: Angular velocity of magnetic field lines at horizon
        - ΩH: Angular velocity of horizon itself
        - Theory predicts ΩF/ΩH = 0.5 for monopole field
        - Must be measured AT THE HORIZON, not domain-averaged
        
        2. Stagnation Surface (u^r = 0):
        - Location where radial velocity vanishes
        - Inside: outflow (u^r > 0), energy extraction occurs here
        - Outside: may be infall (u^r < 0) depending on boundary conditions
        - NOT the fast magnetosonic surface! (common misconception)
        
        3. Power Extraction:
        - Energy flux: dEr = -√g × T^r_t
        - Total power: P = ∫ dEr over sphere
        - Compare with BZ prediction: P_BZ = (a²Φ²/4πr_h²) × (ΩF/ΩH)²
        
        Parameters:
        ----------
        dump_files : list
            List of dump file names to analyze
        sample_every : int
            Analyze every N-th file (default: 5)
        
        Returns:
        -------
        dict : Analysis results containing:
            - times: List of simulation times
            - power_extraction: Energy flux profiles vs radius
            - stagnation_surface_data: u^r = 0 surface location and properties
            - omega_theta_profiles: ΩF(θ)/ΩH at horizon for each time
            - horizon_data: Properties measured at r = r_horizon
        """
        print("=== 2D BZ MONOPOLE ANALYSIS ===")
        
        results = {
            'times': [],
            'power_extraction': [],
            'stagnation_surface_data': [],  # Renamed from stagnation_surface_data
            'omega_theta_profiles': [],  # FIXED: Now stores full θ-profile at horizon
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
                h_2d = hs.h.squeeze()
                rho_2d = hs.rho.squeeze()
                
                # Power extraction analysis
                if hasattr(hs, 'Tud'):
                    dEr = -hs.gdet * hs.Tud[1,0] * hs._dx2 * hs._dx3
                    Er = dEr.sum(axis=-1) if dEr.ndim > 2 else dEr.sum(axis=1)
                    
                    if Er.ndim > 1:
                        Er_avg = Er.mean(axis=1)
                    else:
                        Er_avg = Er
                    
                    results['power_extraction'].append({
                        'time': current_time,
                        'energy_flux': Er_avg,
                        'r_coord': r_2d[:,0] if r_2d.ndim > 1 else r_2d
                    })
                
                # Stagnation surface analysis
                if hasattr(hs, 'uu'):
                    ur = hs.uu[1].squeeze()
                    results['stagnation_surface_data'].append({
                        'time': current_time,
                        'ur': ur,
                        'r': r_2d,
                        'h': h_2d,
                        'rho': rho_2d
                    })
                
                # FIXED: ΩF/ΩH analysis at horizon (full θ-profile)
                try:
                    omega_data = self.extract_omega_at_horizon()
                    
                    if not omega_data['is_1d']:
                        # 2D case: we have full θ-profile
                        results['omega_theta_profiles'].append({
                            'time': current_time,
                            'theta': omega_data['theta'],
                            'omega_f': omega_data['omega_f_horizon'],
                            'omega_ratio': omega_data['omega_ratio_horizon'],
                            'mean_omega_ratio': omega_data['mean_omega_ratio'],
                            'std_omega_ratio': omega_data['std_omega_ratio'],
                            'horizon_radius': omega_data['horizon_radius'],
                            'horizon_idx': omega_data['horizon_idx']
                        })
                        
                        print(f"  <ΩF/ΩH> = {omega_data['mean_omega_ratio']:.4f} ± {omega_data['std_omega_ratio']:.4f}")
                    
                except Exception as e:
                    print(f"  Warning: Could not extract ΩF/ΩH: {e}")
                
                results['times'].append(current_time)
                
            except Exception as e:
                print(f"Error in 2D analysis for {dump_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # FIXED: Final validation
        if results['omega_theta_profiles']:
            final_omega = results['omega_theta_profiles'][-1]
            print(f"\n=== VALIDATION ===")
            print(f"Final <ΩF/ΩH> at horizon: {final_omega['mean_omega_ratio']:.4f} ± {final_omega['std_omega_ratio']:.4f}")
            print(f"Theoretical prediction: 0.500")
            print(f"Deviation: {abs(final_omega['mean_omega_ratio'] - 0.5)/0.5 * 100:.1f}%")
        
        return results
    
    def plot_1d_monopole_results(self, results, show=True):
        """Plot results from 1D monopole analysis - FIXED for horizon ΩF/ΩH"""
        
        fig = plt.figure(figsize=(16, 10))
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
            
            final_gamma = results['lorentz_factors'][-1]
            ax1.text(0.05, 0.95, f'Final γ = {final_gamma:.2f}', 
                    transform=ax1.transAxes, fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9))
        
        # 2. Frame dragging ratio - FIXED
        ax2 = fig.add_subplot(gs[0, 1])
        if results['times'] and results['omega_ratios']:
            ax2.plot(results['times'], results['omega_ratios'], 'r-', linewidth=3)
            ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.8, linewidth=2, label='BZ Theory')
            ax2.set_xlabel('Time', fontsize=13)
            ax2.set_ylabel('ΩF/ΩH at Horizon', fontsize=13)  # FIXED: Specify "at Horizon"
            ax2.set_title('Frame Dragging Efficiency', fontsize=14, pad=15)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=12)
            ax2.legend(fontsize=11, loc='best')
            
            # FIXED: Report precise values
            final_omega = results['omega_ratios'][-1]
            theory = 0.500
            deviation = abs(final_omega - theory) / theory * 100
            
            text = f'ΩF/ΩH = {final_omega:.4f}\nTheory = {theory:.3f}\nDev = {deviation:.1f}%'
            ax2.text(0.05, 0.95, text, 
                    transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.9))
        
        # 3. Key values comparison
        ax3 = fig.add_subplot(gs[0, 2])
        if results['sigma_initial'] and results['lorentz_factors']:
            categories = ['σ₀', '√σ₀', 'γ_final']
            values = [
                results['sigma_initial'], 
                np.sqrt(results['sigma_initial']), 
                results['lorentz_factors'][-1]
            ]
            colors = ['green', 'orange', 'blue']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.7, 
                        edgecolor='black', linewidth=1.5)
            ax3.set_ylabel('Value', fontsize=13)
            ax3.set_title('Magnetization vs Acceleration', fontsize=14, pad=15)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(labelsize=12)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{value:.1f}', ha='center', va='bottom', 
                        fontsize=13, fontweight='bold')
            
            ax3.set_ylim(0, max(values) * 1.15)
        
        # 4. Radial acceleration profiles with time progression
        ax4 = fig.add_subplot(gs[1, 0])
        if results['radial_profiles']['r'] is not None and results['radial_profiles']['gamma']:
            r = results['radial_profiles']['r']
            times = results['radial_profiles']['times']
            
            colormap = plt.cm.plasma
            norm = plt.Normalize(min(times), max(times))
            
            for i in range(0, len(results['radial_profiles']['gamma']), 5):
                gamma_prof = results['radial_profiles']['gamma'][i]
                time = times[i]
                color = colormap(norm(time))
                ax4.loglog(r, gamma_prof, color=color, alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Radius (r/rg)', fontsize=13)
            ax4.set_ylabel('Lorentz Factor Î³', fontsize=13)
            ax4.set_title('Radial Acceleration Evolution', fontsize=14, pad=15)
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(labelsize=12)
            
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax4, shrink=0.8, pad=0.02)
            cbar.set_label('Time', fontsize=12)
            cbar.ax.tick_params(labelsize=11)
        
        # 5. Numerical results summary - FIXED
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')
        
        if results['sigma_initial'] and results['lorentz_factors'] and results['omega_ratios']:
            sigma0 = results['sigma_initial']
            gamma_final = results['lorentz_factors'][-1]
            omega_final = results['omega_ratios'][-1]
            sqrt_sigma0 = np.sqrt(sigma0)
            efficiency = gamma_final / sqrt_sigma0
            final_velocity = np.sqrt(1 - 1/gamma_final**2)
            
            # FIXED: Precise reporting with exact numbers
            theory_omega = 0.500
            omega_deviation = abs(omega_final - theory_omega) / theory_omega * 100
            
            results_text = f"""
    KEY SIMULATION RESULTS (1D Monopole)

    Initial Magnetization:        σ₀ = {sigma0:.1e}
    Theoretical Max γ:            √σ₀ = {sqrt_sigma0:.2f}

    Final Lorentz Factor:         γ_final = {gamma_final:.2f}
    Theoretical Limit:            √σ₀ = {sqrt_sigma0:.2f}
    Acceleration Efficiency:      γ/√σ₀ = {efficiency:.3f} ({efficiency*100:.1f}%)
    Final Velocity:               v/c = {final_velocity:.4f}

    Frame Dragging at Horizon:    ΩF/ΩH = {omega_final:.4f}
    Theoretical Prediction:       ΩF/ΩH = {theory_omega:.3f}
    Fractional Deviation:         {omega_deviation:.1f}%
            """
            
            ax5.text(0.15, 0.95, results_text, transform=ax5.transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", 
                            alpha=0.9, edgecolor='black'))
        
        fig.suptitle('1D Monopole Magnetosphere Analysis', 
                    fontsize=18, fontweight='bold', y=0.92)
        
        filename = os.path.join(self.output_dir, "monopole_1d_results.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved results plot: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_2d_monopole_results(self, results, field_type="Auto", problem_name="", show=True):
        """
        Plot 2D BZ analysis results with precise ΩF/ΩH reporting at horizon.
        """
        if not results['times']:
            print("No results to plot!")
            return
            
        # Auto-detect field type if not specified
        if field_type == "Auto":
            dump_files = get_dump_files()
            if dump_files:
                field_type = self.detect_field_type(dump_files[-1])
        
        print(f"Plotting results from {len(results['times'])} time steps")
        print(f"Detected field type: {field_type}")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # 1. Power extraction evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if results['power_extraction']:
            times = [data['time'] for data in results['power_extraction']]
            colormap = plt.cm.viridis
            norm = plt.Normalize(min(times), max(times))
            
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
            ax1.set_title('Power Extraction Evolution', fontsize=13)
            ax1.grid(True, alpha=0.3)
            
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8, pad=0.02)
            cbar1.set_label('Time', fontsize=11)
            cbar1.ax.tick_params(labelsize=10)
        
        # 2. Stagnation surface visualization
        ax2 = fig.add_subplot(gs[0, 1])
        if results['stagnation_surface_data']:  # RENAMED
            latest_data = results['stagnation_surface_data'][-1]  # RENAMED
            r_2d = latest_data['r']
            h_2d = latest_data['h']
            ur_2d = latest_data['ur']
            rho_2d = latest_data['rho']
            
            # Convert to Cartesian
            x = r_2d * np.sin(h_2d)
            z = r_2d * np.cos(h_2d)
            
            # Mirror for full circle
            x_full = np.concatenate([x, -x], axis=1)
            z_full = np.concatenate([z, z], axis=1)
            rho_full = np.concatenate([rho_2d, rho_2d], axis=1)
            ur_full = np.concatenate([ur_2d, ur_2d], axis=1)
            
            # Better density normalization
            rho_positive = rho_full[rho_full > 0]
            if len(rho_positive) > 0:
                vmin = np.percentile(rho_positive, 1)
                vmax = np.percentile(rho_positive, 99)
            else:
                vmin, vmax = rho_full.min(), rho_full.max()
            
            im = ax2.pcolormesh(x_full, z_full, rho_full, cmap='viridis', 
                            norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.8)
            
            # Overplot stagnation surface contour
            try:
                ax2.contour(x_full, z_full, ur_full, levels=[0], 
                        colors='red', linewidths=2, alpha=0.9)
            except:
                pass
            
            # Add black hole
            if results['omega_theta_profiles']:
                rh = results['omega_theta_profiles'][-1]['horizon_radius']
                circle = plt.Circle((0, 0), rh, color='black', alpha=1.0, zorder=10)
                ax2.add_patch(circle)
            
            ax2.set_xlabel('X (r_g)', fontsize=12)
            ax2.set_ylabel('Z (r_g)', fontsize=12)
            ax2.set_title('Stagnation Surface (u^r=0)', fontsize=13)  # UPDATED
            ax2.set_xlim(-50, 50)
            ax2.set_ylim(-50, 50)
            ax2.set_aspect('equal')
            
            cbar2 = plt.colorbar(im, ax=ax2, label='density', shrink=0.8)
        
        # 3. ΩF(θ)/ΩH profile at horizon
        ax3 = fig.add_subplot(gs[1, 0])
        if results['omega_theta_profiles']:
            latest_omega = results['omega_theta_profiles'][-1]
            theta = latest_omega['theta']
            omega_ratio = latest_omega['omega_ratio']
            final_time = latest_omega['time']
            
            if hasattr(theta, '__len__') and len(theta) > 1:
                # Plot with error band
                ax3.plot(theta, omega_ratio, 'b-', linewidth=3, label='Simulation')
                ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.8, linewidth=2, 
                        label='BZ Theory = 0.500')
                ax3.fill_between(theta, 0.5, omega_ratio, alpha=0.2, color='lightblue')
                
                # Calculate precise statistics
                avg_omega = np.mean(omega_ratio)
                std_omega = np.std(omega_ratio)
                min_omega = np.min(omega_ratio)
                max_omega = np.max(omega_ratio)
                deviation_percent = abs(avg_omega - 0.5) / 0.5 * 100
                
                ax3.set_xlabel('θ (radians)', fontsize=12)
                ax3.set_ylabel('ΩF/ΩH', fontsize=12)
                ax3.set_title(f'Frame Dragging at Horizon (t={final_time:.0f})', fontsize=13)
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(0, np.pi)
                ax3.set_ylim(0.4, 0.8)
                
                # Theta labels
                ax3.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
                ax3.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
                
                # PRECISE numerical annotation (NO VAGUE LANGUAGE)
                stats_text = f'⟨ΩF/ΩH⟩ = {avg_omega:.4f} ± {std_omega:.4f}\n'
                stats_text += f'Theory = 0.5000\n'
                stats_text += f'Deviation = {deviation_percent:.1f}%\n'
                stats_text += f'Range: [{min_omega:.4f}, {max_omega:.4f}]'
                
                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
                
                ax3.legend(loc='best', fontsize=10)
            else:
                # 1D case
                deviation_percent = abs(omega_ratio - 0.5) / 0.5 * 100
                text = f'ΩF/ΩH = {omega_ratio:.4f}\n'
                text += f'Theory = 0.5000\n'
                text += f'Deviation = {deviation_percent:.1f}%'
                
                ax3.text(0.5, 0.5, text, transform=ax3.transAxes, 
                        ha='center', va='center', fontsize=14, fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
                ax3.set_title('Frame Dragging Ratio at Horizon', fontsize=13)
        
        # 4. Numerical summary with PRECISE values
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        if results['omega_theta_profiles']:
            latest_omega = results['omega_theta_profiles'][-1]
            omega_ratio = latest_omega['omega_ratio']
            
            if hasattr(omega_ratio, '__len__'):
                avg_omega = np.mean(omega_ratio)
                std_omega = np.std(omega_ratio)
                min_omega = np.min(omega_ratio)
                max_omega = np.max(omega_ratio)
                deviation_percent = abs(avg_omega - 0.5) / 0.5 * 100
                asymmetry = (max_omega - min_omega) / avg_omega * 100
                
                summary_text = f"""
    2D BZ PROBLEM - QUANTITATIVE RESULTS

    Frame Dragging at Horizon:
    ⟨ΩF/ΩH⟩ = {avg_omega:.4f} ± {std_omega:.4f}
    Min ΩF/ΩH = {min_omega:.4f}
    Max ΩF/ΩH = {max_omega:.4f}
    BZ Theory = 0.5000
    Deviation = {deviation_percent:.2f}%
    Angular variation = {asymmetry:.1f}%

    Black Hole Properties:
    Horizon radius = {latest_omega['horizon_radius']:.3f} rg
    Spin parameter a â‰ˆ {hs.a:.3f}

    Analysis Coverage:
    Time span: {results['times'][0]:.1f} - {results['times'][-1]:.1f} M
    Snapshots analyzed: {len(results['times'])}
                """
            else:
                deviation_percent = abs(omega_ratio - 0.5) / 0.5 * 100
                
                summary_text = f"""
    2D BZ PROBLEM - QUANTITATIVE RESULTS

    Frame Dragging at Horizon:
    ΩF/ΩH = {omega_ratio:.4f}
    BZ Theory = 0.5000
    Deviation = {deviation_percent:.2f}%

    Black Hole Properties:
    Horizon radius = {latest_omega['horizon_radius']:.3f} rg
    Spin parameter a ≈ {hs.a:.3f}

    Analysis Coverage:
    Time span: {results['times'][0]:.1f} - {results['times'][-1]:.1f} M
    Snapshots analyzed: {len(results['times'])}
                """
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
        
        # Dynamic title
        title_map = {
            "Monopole": "2D BZ Monopole Magnetosphere: Quantitative Analysis",
            "Dipole": "2D BZ Dipole Magnetosphere: Quantitative Analysis",
            "Split Monopole": "2D BZ Split Monopole Magnetosphere: Quantitative Analysis",  # NEW!
            "Quadrupole": "2D BZ Quadrupole Magnetosphere: Quantitative Analysis",  # NEW!
            "Mixed/Evolving": "2D BZ Evolving Magnetosphere: Quantitative Analysis", 
            "Unknown": "2D BZ Magnetosphere: Quantitative Analysis"
        }
        
        main_title = title_map.get(field_type, f"2D BZ {field_type} Magnetosphere: Quantitative Analysis")
        if problem_name:
            main_title = f"{main_title} - Problem: {problem_name}"
        fig.suptitle(main_title, fontsize=18, fontweight='bold')
        
        # Save
        safe_field_type = field_type.lower().replace('/', '_').replace(' ', '_')
        filename = os.path.join(self.output_dir, f"bz_magnetosphere_{safe_field_type}_results.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved {field_type} magnetosphere analysis: {filename}")
        
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
            
            # Get radial velocity for stagnation surface
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
            
            # Add stagnation surface if available
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
            ax.set_title(f'2D BZ Monopole: Density + Stagnation Surface (t = {hs.t:.1f})', fontsize=16)
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
            print(f"Saved 2D density animation: {output_path}")
            print(f"Animation duration: {len(sampled_files)/fps:.1f} seconds")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Make sure ffmpeg is installed")
        
        plt.close(fig)
    
    def create_frame_dragging_animation(self, dump_files, output_file="frame_dragging.mp4", fps=6, sample_every=5, problem_name=""):
        """
        Create animation showing ΩF/ΩH(θ) evolution over time at the horizon.
        
        This animation visualizes frame dragging as predicted by the BZ mechanism,
        showing how the magnetic field lines are dragged by the rotating black hole.
        
        Args:
            dump_files: List of dump files to sample from
            output_file: Output filename for animation
            fps: Frames per second for animation
            sample_every: Sample every N-th dump file
            problem_name: Problem name for title (e.g., "bz_dipole")
        """
        print(f"\nCreating frame dragging animation from {len(dump_files)} dumps...")
        print(f"Sampling every {sample_every} files...")
        
        # Sample dumps for animation
        sampled_files = dump_files[::sample_every]
        
        if len(sampled_files) < 5:
            print(f"Warning: Only {len(sampled_files)} frames available. Animation may be short.")
            sampled_files = dump_files  # Use all files if too few
        
        print(f"Using {len(sampled_files)} frames for animation")
        
        # Collect omega data from sampled dumps
        omega_profiles = []
        for i, dump_file in enumerate(sampled_files):
            try:
                self.load_data("gdump", dump_file)
                omega_data = self.extract_omega_at_horizon()
                
                # Only use 2D cases with theta profiles
                if not omega_data['is_1d'] and omega_data['theta'] is not None:
                    omega_profiles.append({
                        'time': hs.t,
                        'theta': omega_data['theta'],
                        'omega_ratio': omega_data['omega_ratio_horizon'],
                        'horizon_radius': omega_data['horizon_radius']
                    })
                    
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(sampled_files)} frames...")
                    
            except Exception as e:
                print(f"Warning: Could not process {dump_file}: {e}")
                continue
        
        if len(omega_profiles) < 5:
            print(f"Error: Insufficient omega profiles ({len(omega_profiles)}). Need at least 5 for animation.")
            return
        
        print(f"Successfully collected {len(omega_profiles)} omega profiles")
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add problem name to title if provided
        title_base = "Frame Dragging Evolution at Horizon"
        if problem_name:
            title_base = f"{title_base} - {problem_name.replace('_', ' ').title()}"
        
        def update(frame):
            ax.clear()
            
            omega_data = omega_profiles[frame]
            theta = omega_data['theta']
            omega_ratio = omega_data['omega_ratio']
            time = omega_data['time']
            
            # Plot current profile
            ax.plot(theta, omega_ratio, 'b-', linewidth=4, alpha=0.9, label='Simulation')
            
            # Add theory reference
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.8, linewidth=2, 
                      label='BZ Theory = 0.5')
            
            # Fill area between theory and simulation
            ax.fill_between(theta, 0.5, omega_ratio, alpha=0.3, color='lightblue')
            
            # Styling
            ax.set_xlabel('θ (radians)', fontsize=14)
            ax.set_ylabel('ΩF/ΩH', fontsize=14)
            ax.set_title(f'{title_base} (t = {time:.1f} M)', fontsize=15)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, np.pi)
            ax.set_ylim(0.35, 0.85)
            
            # Add theta labels
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
            
            # Calculate and display statistics
            avg_omega = np.mean(omega_ratio)
            std_omega = np.std(omega_ratio)
            min_omega = np.min(omega_ratio)
            max_omega = np.max(omega_ratio)
            deviation_pct = abs(avg_omega - 0.5) / 0.5 * 100
            
            stats_text = f'⟨ΩF/ΩH⟩ = {avg_omega:.4f} ± {std_omega:.4f}\n'
            stats_text += f'Range: [{min_omega:.4f}, {max_omega:.4f}]\n'
            stats_text += f'Deviation: {deviation_pct:.1f}%\n'
            stats_text += f'Frame: {frame + 1}/{len(omega_profiles)}'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
            
            ax.legend(loc='best', fontsize=11)
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(omega_profiles),
                                     blit=False, interval=1000/fps, repeat=True)
        
        # Save animation
        output_path = os.path.join(self.output_dir, output_file)
        try:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            duration = len(omega_profiles) / fps
            print(f"\n✓ Saved frame dragging animation: {output_path}")
            print(f"  Frames: {len(omega_profiles)}")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  FPS: {fps}")
        except Exception as e:
            print(f"\n✗ Error saving frame dragging animation: {e}")
            print("  Make sure ffmpeg is installed: sudo apt-get install ffmpeg")
        
        plt.close(fig)
    
    def create_power_extraction_animation(self, results, output_file="bz_monopole_power_extraction.mp4", fps=6):
        """Create animation showing power extraction evolution over time - FIXED
        TODO: delete unused method"""
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

    def plot_stagnation_surface_physics(self, dump_file, show=True, save=False):
        """
        Improved 2-panel plot showing stagnation surface physics.
        
        Panel 1: Radial velocity (kinematics)
        Panel 2: Energy flux (energetics)
        
        Returns fig, axes for further customization
        """
        print("Generating improved stagnation surface physics plot...")
        
        # Load data
        self.load_data("gdump", dump_file)
        current_time = float(hs.t)
        
        # Get data
        r_2d = hs.r.squeeze()
        h_2d = hs.h.squeeze()
        ur_2d = hs.uu[1].squeeze()
        
        a = hs.a
        rhor = 1 + (1 - a**2)**0.5
        
        # Convert to Cartesian for full-circle visualization
        x = r_2d * np.sin(h_2d)
        z = r_2d * np.cos(h_2d)
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        ur_full = np.concatenate([ur_2d[:, ::-1], ur_2d], axis=1)
        
        # Create figure with 1x2 layout
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax1, ax2 = axes
        
        # Calculate statistics for summary
        outflow_frac = (ur_full > 0).sum() / ur_full.size * 100
        infall_frac = (ur_full < 0).sum() / ur_full.size * 100
        
        # Suptitle with physics summary
        suptitle_text = (f'Stagnation Surface Physics | t = {current_time:.1f} M | '
                        f'BH spin: a = {a:.3f}\n'
                        f'Stagnation surface (green): u^r = 0  |  '
                        f'Inside: Outflow ({outflow_frac:.1f}%)  |  '
                        f'Outside: Infall ({infall_frac:.1f}%)')
        fig.suptitle(suptitle_text, fontsize=14, fontweight='bold', y=0.96)
        
        # ========== PANEL 1: RADIAL VELOCITY ==========
        vmax = np.percentile(np.abs(ur_full), 95)
        im1 = ax1.pcolormesh(x_full, z_full, ur_full,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                            shading='auto', rasterized=True)
        
        # Stagnation surface contour
        try:
            ax1.contour(x_full, z_full, ur_full, levels=[0],
                    colors='lime', linewidths=3, linestyles='-')
        except:
            pass
        
        # Black hole
        circle1 = plt.Circle((0, 0), rhor, facecolor='black',
                            edgecolor='white', linewidth=2, zorder=10)
        ax1.add_patch(circle1)
        
        ax1.set_title('Radial Velocity Field (u^r)', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('x (r_g)', fontsize=12)
        ax1.set_ylabel('z (r_g)', fontsize=12)
        ax1.set_aspect('equal')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)
        
        # Colorbar for Panel 1
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Radial Velocity', fontsize=11)
        
        # ========== PANEL 2: ENERGY FLUX ==========
        try:
            # Calculate energy flux
            energy_flux_2d = self.calculate_energy_flux_from_primitives()
            energy_flux_full = np.concatenate([energy_flux_2d[:, ::-1],
                                            energy_flux_2d], axis=1)
            
            # Use symmetric log scale with DIFFERENT colormap
            vmax_abs = np.abs(energy_flux_full).max()
            linthresh = 1e-6
            
            im2 = ax2.pcolormesh(x_full, z_full, energy_flux_full,
                                cmap='PRGn',  # DIFFERENT from Panel 1!
                                norm=SymLogNorm(linthresh=linthresh, 
                                            vmin=-vmax_abs, vmax=vmax_abs),
                                shading='auto', rasterized=True)
            
            # Stagnation surface contour
            try:
                ax2.contour(x_full, z_full, ur_full, levels=[0],
                        colors='black', linewidths=3, linestyles='-', alpha=0.8)
            except:
                pass
            
            # Black hole
            circle2 = plt.Circle((0, 0), rhor, facecolor='black',
                            edgecolor='white', linewidth=2, zorder=10)
            ax2.add_patch(circle2)
            
            # Energy statistics
            extraction_frac = (energy_flux_full > 0).sum() / energy_flux_full.size * 100
            
        except Exception as e:
            # Fallback: show error
            ax2.text(0.5, 0.5, f"Energy flux calculation failed:\n{str(e)[:50]}",
                    transform=ax2.transAxes, ha='center', va='center',
                    fontsize=11, color='red', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8))
            im2 = None
        
        ax2.set_title('Energy Flux $-T^r_t$', fontsize=13, fontweight='bold', pad=10)  # (BZ Power)
        ax2.set_xlabel('x (r_g)', fontsize=12)
        ax2.set_ylabel('z (r_g)', fontsize=12)
        ax2.set_aspect('equal')
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(-50, 50)
        
        # Colorbar for Panel 2
        if im2 is not None:
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('Energy Flux', fontsize=11)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for suptitle
        
        # Save if requested
        if save:
            safe_time = f"{current_time:.0f}".replace(".", "_")
            filename = os.path.join(self.output_dir, 
                                f"stagnation_surface_physics_t{safe_time}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def create_velocity_and_stagnation_animation(self, dump_files, 
                                                output_file="velocity_stagnation.mp4",
                                                fps=10, sample_every=3, early_resolution_boost=True):
        """
        Two-panel animation: Velocity magnitude + Flow direction
        
        Panel 1: Velocity magnitude (scalar) with ur=0 contour (red line)
        Panel 2: Flow direction with color-coded arrows (black=inward, gray=outward)
                Same velocity background for consistency
        
        Features:
        - Full-circle visualization (proper mirroring)
        - Calibrated colorbars from real data
        - Color-coded flow arrows
        - Clear stagnation surface marking
        - Variable sampling: detailed early, overview late (if enabled)
        
        Parameters:
        -----------
        dump_files : list
            List of dump file names
        output_file : str
            Output filename
        fps : int
            Frames per second
        sample_every : int
            Use every N-th dump file (only if early_resolution_boost=False)
        early_resolution_boost : bool
            If True, uses detailed sampling for first 20% then skips more
        """
        
        print(f"\n=== Creating Velocity & Stagnation Surface Animation ===")
        
        # Variable sampling
        if early_resolution_boost:
            n_early = len(dump_files) // 5
            early_dumps = dump_files[:n_early:1]
            late_dumps = dump_files[n_early::6]
            sampled_files = list(early_dumps) + list(late_dumps)
            print(f"Variable sampling: {len(early_dumps)} early + {len(late_dumps)} late")
        else:
            sampled_files = dump_files[::sample_every]
            print(f"Uniform sampling: {len(sampled_files)} frames")
        
        # Data calibration
        print("Calibrating colorbars...")
        all_vmag = []
        for dump_file in sampled_files[::5]:
            try:
                self.load_data("gdump", dump_file)
                if hasattr(hs, 'uu') and len(hs.uu) > 2:
                    vx = hs.uu[1].squeeze()
                    vz = hs.uu[2].squeeze()
                    vmag = np.sqrt(vx**2 + vz**2)
                    all_vmag.extend(vmag.flatten())
            except:
                continue
        
        if all_vmag:
            global_vmag_min = np.percentile(all_vmag, 1)
            global_vmag_max = np.percentile(all_vmag, 99)
        else:
            global_vmag_min, global_vmag_max = 1e-10, 1.0
        
        print(f"Velocity range: {global_vmag_min:.3e} to {global_vmag_max:.3e}")
        
        # Create figure with better layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # IMPORTANT: Adjust layout to prevent title cutoff
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, wspace=0.25)
        
        # Initialize with first frame
        self.load_data("gdump", sampled_files[0])
        r_2d = hs.r.squeeze()
        h_2d = hs.h.squeeze()
        
        vx = hs.uu[1].squeeze() if hasattr(hs, 'uu') else np.zeros_like(r_2d)
        vz = hs.uu[2].squeeze() if hasattr(hs, 'uu') else np.zeros_like(r_2d)
        vmag = np.sqrt(vx**2 + vz**2)
        
        # Full circle coordinates
        x = r_2d * np.sin(h_2d)
        z = r_2d * np.cos(h_2d)
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        vmag_full = np.concatenate([vmag[:, ::-1], vmag], axis=1)
        
        # Panel 1
        im1 = ax1.pcolormesh(x_full, z_full, vmag_full, cmap='viridis',
                            norm=LogNorm(vmin=max(global_vmag_min, 1e-10), vmax=global_vmag_max),
                            shading='auto', rasterized=True)
        cbar1 = plt.colorbar(im1, ax=ax1, label='|v| (code units)', fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)  # Smaller tick labels
        
        # Panel 2 (lighter background)
        im2 = ax2.pcolormesh(x_full, z_full, vmag_full, cmap='viridis',
                            norm=LogNorm(vmin=max(global_vmag_min, 1e-10), vmax=global_vmag_max),
                            shading='auto', alpha=0.4, rasterized=True)
        
        # Set axis properties
        for ax in [ax1, ax2]:
            ax.set_xlabel('X (rg)', fontsize=11)
            ax.set_ylabel('Z (rg)', fontsize=11)
            ax.set_aspect('equal')
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.tick_params(labelsize=9)  # Smaller tick labels
        
        def animate(frame):
            # Clear plot contents
            for artist in ax1.collections + ax1.patches + ax1.lines + ax1.texts:
                artist.remove()
            for artist in ax2.collections + ax2.patches + ax2.lines + ax2.texts:
                artist.remove()
            
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            current_time = float(hs.t)
            
            r_2d = hs.r.squeeze()
            h_2d = hs.h.squeeze()
            
            vx = hs.uu[1].squeeze() if hasattr(hs, 'uu') else np.zeros_like(r_2d)
            vz = hs.uu[2].squeeze() if hasattr(hs, 'uu') else np.zeros_like(r_2d)
            vmag = np.sqrt(vx**2 + vz**2)
            ur = hs.uu[1].squeeze() if hasattr(hs, 'uu') else None
            
            a = hs.a
            rhor = 1 + (1 - a**2)**0.5
            
            # Full circle
            x = r_2d * np.sin(h_2d)
            z = r_2d * np.cos(h_2d)
            x_full = np.concatenate([-x[:, ::-1], x], axis=1)
            z_full = np.concatenate([z[:, ::-1], z], axis=1)
            vmag_full = np.concatenate([vmag[:, ::-1], vmag], axis=1)
            
            # PANEL 1: Velocity Magnitude
            im1_new = ax1.pcolormesh(x_full, z_full, vmag_full, cmap='viridis',
                                    norm=LogNorm(vmin=max(global_vmag_min, 1e-10), vmax=global_vmag_max),
                                    shading='auto', rasterized=True)
            
            # Stagnation surface (RED contour)
            if ur is not None:
                try:
                    ur_full = np.concatenate([ur[:, ::-1], ur], axis=1)
                    ax1.contour(x_full, z_full, ur_full, levels=[0],
                            colors='red', linewidths=3, alpha=0.95)
                except:
                    pass
            
            # Black hole
            circle1 = plt.Circle((0, 0), rhor, facecolor='black', edgecolor='white',
                                linewidth=2, alpha=1.0, zorder=10)
            ax1.add_patch(circle1)
            
            # Legend (minimal)
            ax1.text(0.02, 0.98, 'Red: ur=0',
                    transform=ax1.transAxes, fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax1.set_title(f'Velocity Magnitude | t={current_time:.1f}M',
                        fontsize=12, fontweight='bold', pad=8)
            ax1.grid(True, alpha=0.2)
            
            # PANEL 2: Flow Direction
            im2_new = ax2.pcolormesh(x_full, z_full, vmag_full, cmap='viridis',
                                    norm=LogNorm(vmin=max(global_vmag_min, 1e-10), vmax=global_vmag_max),
                                    shading='auto', alpha=0.4, rasterized=True)
            
            # FIX: Mirror arrows to full circle!
            skip = 6  # Reduced for more arrows
            
            # Right half
            x_vec_right = x[::skip, ::skip]
            z_vec_right = z[::skip, ::skip]
            vx_vec_right = vx[::skip, ::skip]
            vz_vec_right = vz[::skip, ::skip]
            ur_vec_right = ur[::skip, ::skip] if ur is not None else None
            
            # Left half (mirrored)
            x_vec_left = -x[::skip, ::-skip]  # Mirror x, reverse indexing
            z_vec_left = z[::skip, ::-skip]
            vx_vec_left = -vx[::skip, ::-skip]  # Mirror vx
            vz_vec_left = vz[::skip, ::-skip]
            ur_vec_left = ur[::skip, ::-skip] if ur is not None else None
            
            # Combine both halves
            x_vec_combined = np.concatenate([x_vec_left, x_vec_right], axis=1)
            z_vec_combined = np.concatenate([z_vec_left, z_vec_right], axis=1)
            vx_vec_combined = np.concatenate([vx_vec_left, vx_vec_right], axis=1)
            vz_vec_combined = np.concatenate([vz_vec_left, vz_vec_right], axis=1)
            if ur_vec_right is not None:
                ur_vec_combined = np.concatenate([ur_vec_left, ur_vec_right], axis=1)
            else:
                ur_vec_combined = None
            
            # Plot color-coded arrows
            if ur_vec_combined is not None:
                x_flat = x_vec_combined.flatten()
                z_flat = z_vec_combined.flatten()
                vx_flat = vx_vec_combined.flatten()
                vz_flat = vz_vec_combined.flatten()
                ur_flat = ur_vec_combined.flatten()
                
                inward_mask = ur_flat < 0
                outward_mask = ur_flat >= 0
                
                # Inward (BLACK) - BIGGER arrows
                if np.any(inward_mask):
                    ax2.quiver(x_flat[inward_mask], z_flat[inward_mask],
                            vx_flat[inward_mask], vz_flat[inward_mask],
                            scale=0.3, scale_units='xy',  # Bigger: 0.3 instead of 0.5
                            color='black', alpha=0.95, width=0.008,  # Thicker
                            headwidth=6, headlength=7, zorder=5)  # Bigger heads
                
                # Outward (LIGHT GRAY) - BIGGER arrows
                if np.any(outward_mask):
                    ax2.quiver(x_flat[outward_mask], z_flat[outward_mask],
                            vx_flat[outward_mask], vz_flat[outward_mask],
                            scale=0.3, scale_units='xy',
                            color='lightgray', edgecolors='gray', linewidths=0.5,
                            alpha=0.95, width=0.008,
                            headwidth=6, headlength=7, zorder=5)
            
            
            # Black hole
            circle2 = plt.Circle((0, 0), rhor, facecolor='black', edgecolor='white',
                                linewidth=2, alpha=1.0, zorder=10)
            ax2.add_patch(circle2)
            
            # Simple legend
            legend_elements = [
                Line2D([0], [0], marker='>', color='w', markerfacecolor='black',
                    markersize=10, label='Inward (ur<0)'),
                Line2D([0], [0], marker='>', color='w', markerfacecolor='lightgray',
                    markeredgecolor='gray', markersize=10, label='Outward (ur≥0)')
            ]
            ax2.legend(handles=legend_elements, loc='best', fontsize=9, framealpha=0.9)
            
            ax2.set_title(f'Flow Direction | t={current_time:.1f}M',
                        fontsize=12, fontweight='bold', pad=8)
            ax2.grid(True, alpha=0.2)
            
            return [im1_new, im2_new]
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(sampled_files),
                                    blit=False, interval=1000/fps, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            print(f"Saving animation...")
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"✓ Saved: {output_path}")
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"  Size: {file_size:.2f} MB, Duration: {len(sampled_files)/fps:.1f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        plt.close(fig)


    def create_energy_zones_animation(self, dump_files,
                                    output_file="energy_zones.mp4",
                                    fps=10, sample_every=3, early_resolution_boost=True):
        """
        Two-panel animation: Energy flux zones + Integrated Power Evolution
        
        Panel 1: -T^t_r (energy flux) showing extraction (blue) vs dissipation (red)
        Panel 2: INTEGRATED POWER over time (replaces spatial coverage)
        
        NEW APPROACH: Uses physics from create_power_extraction_animation
        - Calculates P(r) = |E_r| × 4πr² at each timestep
        - Tracks extraction power, dissipation power, net power
        - More physically meaningful than cell counting
        
        Parameters:
        -----------
        dump_files : list
            List of dump file names
        output_file : str
            Output filename
        fps : int
            Frames per second
        sample_every : int
            Use every N-th dump file (only if early_resolution_boost=False)
        early_resolution_boost : bool
            If True, uses detailed sampling for first 20% then skips more
        """
        
        print(f"\n=== Creating Energy Extraction Zones Animation ===")
        
        # Variable sampling
        if early_resolution_boost:
            n_early = len(dump_files) // 5
            early_dumps = dump_files[:n_early:2]
            late_dumps = dump_files[n_early::8]
            sampled_files = list(early_dumps) + list(late_dumps)
            print(f"Variable sampling: {len(early_dumps)} early + {len(late_dumps)} late")
        else:
            sampled_files = dump_files[::sample_every]
            print(f"Uniform sampling: {len(sampled_files)} frames")
        
        # Data calibration
        print("Calibrating energy flux...")
        all_Ttr = []
        
        for dump_file in sampled_files[::5]:
            try:
                self.load_data("gdump", dump_file)
                if hasattr(hs, 'T') and len(hs.T) > 2:
                    Ttr = -hs.T[0,1].squeeze()
                else:
                    rho = hs.rho.squeeze()
                    ug = hs.ug.squeeze()
                    ur = hs.uu[1].squeeze() if hasattr(hs, 'uu') else np.zeros_like(rho)
                    Ttr = -(rho + ug) * ur
                
                all_Ttr.extend(Ttr.flatten())
            except:
                continue
        
        if all_Ttr:
            abs_max = np.percentile(np.abs(all_Ttr), 95)
            global_Ttr_min = -abs_max
            global_Ttr_max = abs_max
        else:
            global_Ttr_min, global_Ttr_max = -0.1, 0.1
        
        print(f"Energy flux range: {global_Ttr_min:.3e} to {global_Ttr_max:.3e}")
        
        # Create figure
        fig = plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        
        plt.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.95, wspace=0.25)
        
        # Storage for integrated power (NEW!)
        times_list = []
        extraction_power_list = []  # Total power from extraction zones
        dissipation_power_list = []  # Total power from dissipation zones
        net_power_list = []  # Net power (extraction - dissipation)
        
        # Initialize first frame
        self.load_data("gdump", sampled_files[0])
        r_2d = hs.r.squeeze()
        h_2d = hs.h.squeeze()
        
        if hasattr(hs, 'T') and len(hs.T) > 2:
            Ttr = -hs.T[0,1].squeeze()
        else:
            rho = hs.rho.squeeze()
            ug = hs.ug.squeeze()
            ur = hs.uu[1].squeeze() if hasattr(hs, 'uu') else np.zeros_like(rho)
            Ttr = -(rho + ug) * ur
        
        # Full circle
        x = r_2d * np.sin(h_2d)
        z = r_2d * np.cos(h_2d)
        x_full = np.concatenate([-x[:, ::-1], x], axis=1)
        z_full = np.concatenate([z[:, ::-1], z], axis=1)
        Ttr_full = np.concatenate([Ttr[:, ::-1], Ttr], axis=1)
        
        # Panel 1
        im1 = ax1.pcolormesh(x_full, z_full, Ttr_full, cmap='RdBu_r',
                            vmin=global_Ttr_min, vmax=global_Ttr_max,
                            shading='auto', rasterized=True)
        cbar1 = plt.colorbar(im1, ax=ax1, label='-T^t_r (code units)', 
                            fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=9)
        cbar1.ax.axhline(y=0, color='black', linewidth=2, linestyle='--')
        
        # Set axis properties
        ax1.set_xlabel('X (rg)', fontsize=11)
        ax1.set_ylabel('Z (rg)', fontsize=11)
        ax1.set_aspect('equal')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)
        ax1.tick_params(labelsize=9)
        
        ax2.set_xlabel('Time (M)', fontsize=11)
        ax2.set_ylabel('Integrated Power (code units)', fontsize=11)
        ax2.tick_params(labelsize=9)
        
        # Suptitle
        fig.suptitle('BZ Energy Extraction: -T^t_r and Integrated Power Evolution', 
                    fontsize=13, fontweight='bold', y=0.96)
        
        def animate(frame):
            # Clear
            for artist in ax1.collections + ax1.patches + ax1.lines + ax1.texts:
                artist.remove()
            ax2.clear()
            
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            current_time = float(hs.t)
            
            r_2d = hs.r.squeeze()
            h_2d = hs.h.squeeze()
            
            if hasattr(hs, 'T') and len(hs.T) > 2:
                Ttr = -hs.T[0,1].squeeze()
            else:
                rho = hs.rho.squeeze()
                ug = hs.ug.squeeze()
                ur = hs.uu[1].squeeze() if hasattr(hs, 'uu') else np.zeros_like(rho)
                Ttr = -(rho + ug) * ur
            
            a = hs.a
            rhor = 1 + (1 - a**2)**0.5
            
            # Full circle
            x = r_2d * np.sin(h_2d)
            z = r_2d * np.cos(h_2d)
            x_full = np.concatenate([-x[:, ::-1], x], axis=1)
            z_full = np.concatenate([z[:, ::-1], z], axis=1)
            Ttr_full = np.concatenate([Ttr[:, ::-1], Ttr], axis=1)
            
            # ====================================================================
            # PANEL 1: Energy Flux Map
            # ====================================================================
            im1_new = ax1.pcolormesh(x_full, z_full, Ttr_full, cmap='RdBu_r',
                                    vmin=global_Ttr_min, vmax=global_Ttr_max,
                                    shading='auto', rasterized=True)
            
            # Zero contour
            try:
                ax1.contour(x_full, z_full, Ttr_full, levels=[0],
                        colors='black', linewidths=2, linestyles='--', alpha=0.8)
            except:
                pass
            
            # Black hole
            circle = plt.Circle((0, 0), rhor, facecolor='black', edgecolor='white',
                            linewidth=2, alpha=1.0, zorder=10)
            ax1.add_patch(circle)
            
            # Labels
            ax1.text(0.98, 0.98, 'Blue: Extraction\n(Energy Out)',
                    transform=ax1.transAxes, fontsize=9, ha='right', va='top',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))
            ax1.text(0.98, 0.02, 'Red: Dissipation\n(Energy In)',
                    transform=ax1.transAxes, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.8))
            
            ax1.set_title(f't = {current_time:.1f}M', fontsize=12, fontweight='bold', pad=8)
            ax1.set_xlabel('X (rg)', fontsize=11)
            ax1.set_ylabel('Z (rg)', fontsize=11)
            ax1.set_xlim(-50, 50)
            ax1.set_ylim(-50, 50)
            ax1.tick_params(labelsize=9)
            ax1.grid(True, alpha=0.2)
            
            # ====================================================================
            # PANEL 2: INTEGRATED POWER EVOLUTION (NEW!)
            # Using logic from create_power_extraction_animation
            # ====================================================================
            
            # Calculate integrated power at this timestep
            # Average energy flux over theta (if 2D)
            if Ttr.ndim > 1:
                energy_flux_avg = Ttr.mean(axis=1)  # Average over theta
                r_coord = r_2d[:, 0]  # Radial coordinate
            else:
                energy_flux_avg = Ttr
                r_coord = r_2d
            
            # Integrated power: P(r) = |E_r| × 4πr²
            # This converts energy flux density to total power through spherical shell
            integrated_power_profile = np.abs(energy_flux_avg) * 4 * np.pi * r_coord**2
            
            # Split into extraction and dissipation zones
            extraction_mask = energy_flux_avg > 0
            dissipation_mask = energy_flux_avg < 0
            
            # Total power in each zone (sum over all radii)
            extraction_power = np.sum(integrated_power_profile[extraction_mask]) if np.any(extraction_mask) else 0
            dissipation_power = np.sum(integrated_power_profile[dissipation_mask]) if np.any(dissipation_mask) else 0
            net_power = extraction_power - dissipation_power
            
            # Store for time series
            times_list.append(current_time)
            extraction_power_list.append(extraction_power)
            dissipation_power_list.append(dissipation_power)
            net_power_list.append(net_power)
            
            # Plot power evolution
            ax2.plot(times_list, extraction_power_list, 'b-', linewidth=2.5,
                    label='Extraction Power', marker='o', markersize=3)
            ax2.plot(times_list, dissipation_power_list, 'r-', linewidth=2.5,
                    label='Dissipation Power', marker='s', markersize=3)
            ax2.plot(times_list, net_power_list, 'k--', linewidth=2.5,
                    label='Net Power', alpha=0.7, marker='^', markersize=3)
            ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
            
            ax2.set_xlabel('Time (M)', fontsize=11)
            ax2.set_ylabel('Integrated Power (code units)', fontsize=11)
            ax2.set_title('Total Integrated Power Evolution', fontsize=12, fontweight='bold', pad=8)
            ax2.legend(loc='best', fontsize=9, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=9)
            
            # Stats box
            stats_text = (f'Time: {current_time:.1f}M\n'
                        f'Extraction: {extraction_power:.2e}\n'
                        f'Dissipation: {dissipation_power:.2e}\n'
                        f'Net: {net_power:.2e}')
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=9, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
            
            return [im1_new]
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(sampled_files),
                                    blit=False, interval=1000/fps, repeat=True)
        
        output_path = os.path.join(self.output_dir, output_file)
        try:
            print(f"Saving animation...")
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
            print(f"✓ Saved: {output_path}")
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"  Size: {file_size:.2f} MB, Duration: {len(sampled_files)/fps:.1f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        plt.close(fig)


    def create_magnetic_topology_animation(self, dump_files,
                                        output_file="magnetic_topology.mp4",
                                        fps=25, sample_every=1, early_resolution_boost=True):
        """
        Two-panel animation: B_r(θ) profile + Hemisphere flux evolution
        
        Panel 1: B_r(θ) at horizon with Legendre polynomial decomposition
        Panel 2: Northern & Southern hemisphere flux time series
        
        Features:
        - Legendre polynomial decomposition (proper multipole identification)
        - Hemisphere flux tracking
        - Clean, informative visualization
        - Variable sampling: detailed early, overview late (if enabled)
        
        Parameters:
        -----------
        dump_files : list
            List of dump file names
        output_file : str
            Output filename
        fps : int
            Frames per second
        sample_every : int
            Use every N-th dump file (only if early_resolution_boost=False)
        early_resolution_boost : bool
            If True, uses detailed sampling for first 20% then skips more
        """
        
        output_path = os.path.join(self.output_dir, output_file)
        print(f"\n=== Creating Magnetic Topology Animation ===")
        
        # Variable sampling
        if early_resolution_boost:
            n_early = len(dump_files) // 5
            early_dumps = dump_files[:n_early:1]
            late_dumps = dump_files[n_early::5]
            sampled_files = list(early_dumps) + list(late_dumps)
            print(f"Variable sampling: {len(early_dumps)} early + {len(late_dumps)} late")
        else:
            sampled_files = dump_files[::sample_every]
            print(f"Uniform sampling: {len(sampled_files)} frames")
        
        # Create figure with better spacing
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                        gridspec_kw={'height_ratios': [3, 1]})
        plt.subplots_adjust(top=0.95, bottom=0.08, hspace=0.25)
        
        # Storage for time series
        time_history = []
        flux_north_history = []
        flux_south_history = []
        Br_max_history = []  # Track B_r magnitude for decay visualization
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            dump_file = sampled_files[frame]
            self.load_data("gdump", dump_file)
            current_time = float(hs.t)
            
            # Get field data
            flux_data = self.calculate_hemisphere_flux()
            topo_data = self.analyze_field_multipoles_legendre(
                B_r_horizon=flux_data['B_r_horizon'],
                theta=flux_data['theta']
            )
            
            theta = flux_data['theta']
            B_r = flux_data['B_r_horizon']
            
            # PANEL 1: B_r(θ) Profile
            ax1.plot(theta, B_r, 'k-', linewidth=3, zorder=5, label='Simulation data')

            # Plot reconstructed field from Legendre decomposition (for validation)
            if 'B_r_reconstructed' in topo_data:
                ax1.plot(topo_data['theta'], topo_data['B_r_reconstructed'], 
                        'r--', linewidth=2, alpha=0.7, zorder=4, label='Legendre fit')
            
            # Zero line and equator
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax1.axvline(x=np.pi/2, color='gray', linestyle=':', alpha=0.3, linewidth=1.5)
            
            # Color-code plot based on field classification
            field_type = topo_data.get('field_type', 'Unknown')
            confidence = topo_data.get('confidence', 'Low')
            
            # Map field type to color
            field_colors = {
                'Regular Monopole': '#2E7D32',  # Green
                'Pure Dipole': '#1565C0',       # Blue
                'Split Monopole': '#C62828',    # Red
                'Quadrupole': '#F57C00',        # Orange
                'Mixed Multipole': '#6A1B9A'    # Purple
            }
            field_color = field_colors.get(field_type, '#424242')
            
            # Formatting
            ax1.set_xlabel('θ', fontsize=12)
            ax1.set_ylabel('B_r  [simulation units]', fontsize=12)  # Clarified "code units"
            ax1.set_xlim(0, np.pi)
            ax1.grid(True, alpha=0.25)
            ax1.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax1.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
            ax1.tick_params(labelsize=10)
            
            # Title with field classification
            ax1.set_title(f'B_r(θ) at Horizon  |  t = {current_time:.1f}M  |  {field_type}',
                        fontsize=13, fontweight='bold', pad=10, color=field_color)
            
            # Stats box with Legendre analysis results
            dominant_l = topo_data.get('dominant_order', 0)
            even_odd_ratio = topo_data.get('even_odd_ratio', 1.0)
            monopole_strength = topo_data.get('monopole_strength', 0)
            dipole_strength = topo_data.get('dipole_strength', 0)
            
            stats_text = f'Field Type: {field_type}\n'
            stats_text += f'Confidence: {confidence}\n'
            stats_text += f'Dominant l: {dominant_l}\n'
            stats_text += f'Even/Odd: {even_odd_ratio:.3f}\n'
            stats_text += f'|a₀|: {monopole_strength:.2e}\n'
            stats_text += f'|a₁|: {dipole_strength:.2e}\n'
            stats_text += f'Î¦_N: {flux_data["flux_north"]:+.2e}\n'
            stats_text += f'Î¦_S: {flux_data["flux_south"]:+.2e}'
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor=field_color, alpha=0.9, linewidth=2))
            
            # Adaptive legend placement
            ax1.legend(loc='best', fontsize=9, framealpha=0.9)
            
            # Dynamic y-axis to show B_r decay
            Br_max_history.append(np.abs(B_r).max())
            current_Br_max = np.abs(B_r).max()
            ax1.set_ylim(-current_Br_max*1.1, current_Br_max*1.1)
            
            # PANEL 2: Hemisphere Flux Evolution 
            time_history.append(current_time)
            flux_north_history.append(flux_data['flux_north'])
            flux_south_history.append(flux_data['flux_south'])
            
            # Plot flux evolution
            ax2.plot(time_history, flux_north_history, 'b-', linewidth=2.5,
                    label='Northern Hemisphere', alpha=0.9, marker='o', markersize=2)
            ax2.plot(time_history, flux_south_history, 'r-', linewidth=2.5,
                    label='Southern Hemisphere', alpha=0.9, marker='s', markersize=2)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark current time
            ax2.axvline(x=current_time, color='black', linestyle=':',
                    alpha=0.5, linewidth=1.5)
            
            # Formatting
            ax2.set_xlabel('Time (M)', fontsize=11)
            ax2.set_ylabel('Magnetic Flux  [simulation units]', fontsize=11)
            ax2.set_title('Hemisphere Flux Evolution over Time',
                        fontsize=12, fontweight='bold', pad=8)
            ax2.grid(True, alpha=0.25)
            ax2.legend(loc='best', fontsize=10, framealpha=0.9)
            ax2.set_xlim(0, max(time_history) if time_history else 1000)
            ax2.tick_params(labelsize=10)
            
            return [ax1, ax2]
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(sampled_files),
                                    blit=False, interval=1000/fps, repeat=True)
        
        try:
            print(f"Saving animation...")
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=100,
                    extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print(f"✓ Saved: {output_path}")
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"  Size: {file_size:.1f} MB, Duration: {len(sampled_files)/fps:.1f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        plt.close(fig)


    def generate_all_animations(self, dump_files, output_dir=None, fps=10, 
                            early_resolution_boost=True, problem_name=""):
        """
        Convenience method to generate all three animations at once
        
        Parameters:
        -----------
        dump_files : list
            List of dump file names
        output_dir : str, optional
            Output directory (uses self.output_dir if None)
        fps : int
            Frames per second
        early_resolution_boost : bool
            If True, uses variable sampling (detailed early, faster late)
        problem_name : str, optional
            Problem name to include in filenames (e.g., "bz_monopole")
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create problem suffix for filenames
        suffix = f"_{problem_name}" if problem_name else ""
        
        print("\n" + "="*70)
        print(f"GENERATING COMPLETE ANIMATION SUITE{' FOR ' + problem_name.upper() if problem_name else ''}")
        print("="*70)
        
        animations = [
            (f"velocity_stagnation{suffix}.mp4",
            lambda df, out: self.create_velocity_and_stagnation_animation(
                df, out, fps=fps, early_resolution_boost=early_resolution_boost)),
            
            (f"energy_zones{suffix}.mp4",
            lambda df, out: self.create_energy_zones_animation(
                df, out, fps=fps, early_resolution_boost=early_resolution_boost)),
            
            (f"magnetic_topology{suffix}.mp4",
            lambda df, out: self.create_magnetic_topology_animation(
                df, out, fps=fps, early_resolution_boost=early_resolution_boost)),
        ]
        
        for filename, method in animations:
            # ✅ Pass ONLY filename, not full path
            print(f"\n>>> Creating: {filename}")
            try:
                method(dump_files, filename)  # ✅ Just filename
                print(f"✓ Success: {filename}")
            except Exception as e:
                print(f"✗ Failed: {filename}")
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*70)
        print("ANIMATION SUITE COMPLETE")
        print("="*70)
        print(f"\nAll animations saved to: {output_dir}")
        print("\nGenerated files:")
        for filename, _ in animations:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024*1024)
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {filename} (not found)")

    def show_all_animations(self, animation_files, problem_label, save=True, show=True):
        """
        Combine all animation outputs (energy zones, magnetic topology, velocity stagnation)
        into a single multi-panel figure for simultaneous viewing.
        
        This creates a unified visualization that shows all three key aspects of the
        magnetized accretion simulation side-by-side, making it easier to identify
        correlations and overall system behavior.
        
        Args:
            animation_files (dict): Dictionary of file paths, e.g.:
                {
                    'energy_zones': 'path/to/energy_zones.mp4',
                    'magnetic_topology': 'path/to/magnetic_topology.mp4',
                    'velocity_stagnation': 'path/to/velocity_stagnation.mp4'
                }
            problem_label (str): Description of the analyzed configuration (e.g., "bz_monopole")
            save (bool): Whether to save the combined figure
            show (bool): Whether to display the result
        
        Returns:
            fig: The created matplotlib figure
        
        Example:
            >>> analyzer = MagnetizedAnalysis()
            >>> anim_files = {
            ...     'energy_zones': './magnetized_plots/energy_zones.mp4',
            ...     'magnetic_topology': './magnetized_plots/magnetic_topology.mp4',
            ...     'velocity_stagnation': './magnetized_plots/velocity_stagnation.mp4'
            ... }
            >>> analyzer.show_all_animations(anim_files, "BZ Monopole")
        """
        
        fig = plt.figure(figsize=(18, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.05)
        
        titles = {
            'energy_zones': 'Energy Zones',
            'magnetic_topology': 'Magnetic Topology',
            'velocity_stagnation': 'Velocity & Stagnation'
        }
        
        for i, key in enumerate(['energy_zones', 'magnetic_topology', 'velocity_stagnation']):
            ax = fig.add_subplot(gs[0, i])
            ax.set_title(titles[key], fontsize=14, fontweight='bold', pad=10)
            ax.axis('off')
            
            if key in animation_files and os.path.exists(animation_files[key]):
                # For video files, we can show the first frame or a representative frame
                # Note: This would require video processing. For now, show placeholder.
                ax.text(0.5, 0.5, 
                       f'{titles[key]}\n\nVideo available at:\n{os.path.basename(animation_files[key])}',
                       ha='center', va='center', fontsize=11, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            else:
                ax.text(0.5, 0.5, f"No data for {key}", 
                       ha='center', va='center', fontsize=12, color='gray')
        
        fig.suptitle(f"Combined BZ Analysis - {problem_label}", 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save:
            safe_label = problem_label.replace(' ', '_').replace('/', '_')
            out_path = os.path.join(self.output_dir, f"combined_animations_{safe_label}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved combined animation panel to: {out_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig

    def analyze_magnetic_hair_loss(self, dump_files, sample_every=5):
        """
        Analyze magnetic field evolution to understand black hole "hair loss".
        
        Physical Context:
        ----------------
        No-Hair Theorem: Classical black holes are characterized only by:
        - Mass M
        - Charge Q  
        - Angular momentum J (spin a)
        
        Any other "hair" (complex field structure) should decay away.
        
        However: Rotating black holes can sustain magnetic structure via
        frame dragging! This is not a violation because:
        - Magnetic field is not "hair" - it's maintained by currents
        - Frame dragging continuously re-orients field lines
        - Result: quasi-stationary magnetosphere
        
        Analysis Goals:
        --------------
        1. Field Persistence:
        - Does B-field strength remain constant or decay?
        - Compare initial vs final field strength
        
        2. Field Topology:
        - Monopole: Constant sign of B_r(θ)
        - Dipole: B_r changes sign at equator
        - Evolution: Dipole → Monopole transition?
        
        3. Frame Dragging:
        - ΩF/ΩH should remain ≈ 0.5 for monopole
        - Validates that field is sustained by rotation
        
        4. Flux Conservation:
        - Total flux Φ = ∫ B_r dA should be conserved
        - Violations indicate numerical diffusion
        
        Parameters:
        ----------
        dump_files : list
            Dump files spanning simulation evolution
        sample_every : int
            Analyze every N-th file
        
        Returns:
        -------
        dict : {
            'field_strength_horizon': List of B-field magnitudes at r_h
            'field_topology_ratio': Pole/equator field ratio vs time #TODO: update documentation
            'frame_dragging_efficiency': ΩF/ΩH vs time
            'magnetic_flux_conservation': Φ(t)/Φ(0)
        }
        
        Interpretation:
        --------------
        - Stable field + constant ΩF/ΩH: BZ mechanism operating
        - Decaying field: No frame dragging (non-rotating BH)
        - Evolving topology: Field restructuring (dipole → monopole)
        """
        print("\n=== ANALYZING BLACK HOLE 'HAIR LOSS' ===")
        print("No-Hair Theorem: Classical BHs characterized only by M, Q, J")
        print("But rotating BHs can sustain magnetic structure via frame dragging!")
        
        results = {
            'times': [],
            'field_strength_horizon': [],
            'field_topology_ratio': [],
            'frame_dragging_efficiency': [],  # Now stores horizon values
            'frame_dragging_evolution': [],  # Full evolution data
            'magnetic_flux_conservation': [],
            'hemisphere_flux_evolution': [],  #  Φ_N, Φ_S over time
            'topology_evolution': []         # Field classification over time

        }
        
        for dump_file in dump_files[::sample_every]:
            try:
                self.load_data("gdump", dump_file)
                current_time = float(hs.t)
                
                if not hasattr(hs, 'B'):
                    print(f"No magnetic field data in {dump_file}")
                    continue
                
                B_r = hs.B[1].squeeze()
                B_theta = hs.B[2].squeeze()
                r_2d = hs.r.squeeze()
                theta_2d = hs.h.squeeze()
                
                # Field strength near horizon
                horizon_field_strength = np.sqrt(B_r[:5, :]**2 + B_theta[:5, :]**2).mean()
                
                # Field topology
                n_theta = B_r.shape[1]
                pole_idx = 0
                equator_idx = n_theta // 2
                
                pole_field = np.sqrt(B_r[:10, pole_idx]**2 + B_theta[:10, pole_idx]**2).mean()
                equator_field = np.sqrt(B_r[:10, equator_idx]**2 + B_theta[:10, equator_idx]**2).mean()
                topology_ratio = pole_field / equator_field if equator_field > 0 else 1
                
                # FIXED: Frame dragging at horizon
                try:
                    omega_data = self.extract_omega_at_horizon()
                    
                    if omega_data['is_1d']:
                        frame_drag_eff = omega_data['omega_ratio_horizon']
                    else:
                        frame_drag_eff = omega_data['mean_omega_ratio']
                    
                    results['frame_dragging_evolution'].append({
                        'time': current_time,
                        'omega_ratio': frame_drag_eff,
                        'std': omega_data['std_omega_ratio'],
                        'horizon_radius': omega_data['horizon_radius']
                    })
                    
                except Exception as e:
                    print(f"Warning: Could not extract ΩF/ΩH: {e}")
                    frame_drag_eff = None
                
                # Flux conservation (equatorial plane for now)
                if r_2d.ndim > 1:
                    flux_equatorial = (B_r[:, equator_idx] * r_2d[:, equator_idx]**2).sum()
                else:
                    flux_equatorial = (B_r * r_2d**2).sum()
                
                # Store results
                results['times'].append(current_time)
                results['field_strength_horizon'].append(horizon_field_strength)
                results['field_topology_ratio'].append(topology_ratio)
                results['frame_dragging_efficiency'].append(frame_drag_eff)
                results['magnetic_flux_conservation'].append(flux_equatorial)
                
                try:
                    flux_data = self.calculate_hemisphere_flux()
                    
                    results['hemisphere_flux_evolution'].append({
                        'time': current_time,
                        'flux_north': flux_data['flux_north'],
                        'flux_south': flux_data['flux_south'],
                        'flux_total': flux_data['flux_total'],
                        'same_sign': flux_data['same_sign'],
                        'field_type': flux_data['field_type']
                    })
                    
                    # Use Legendre polynomial decomposition for proper multipole identification
                    topo_data = self.analyze_field_multipoles_legendre(
                        B_r_horizon=flux_data['B_r_horizon'],
                        theta=flux_data['theta']
                    )
                    
                    results['topology_evolution'].append({
                        'time': current_time,
                        'field_type': topo_data['field_type'],
                        'dominant_order': topo_data['dominant_order'],
                        'even_odd_ratio': topo_data['even_odd_ratio'],
                        'confidence': topo_data['confidence']
                    })
                    
                    # Enhanced print with Legendre classification
                    omega_str = f"{frame_drag_eff:.4f}" if (frame_drag_eff is not None and not np.isnan(frame_drag_eff)) else "N/A"
                    print(f"t={current_time:.1f}: B_horizon={horizon_field_strength:.2e}, "
                        f"topology={topology_ratio:.2f}, ΩF/ΩH={omega_str}, "
                        f"hemisphere={flux_data['field_type']}, "
                        f"Legendre: {topo_data['field_type']} (l={topo_data['dominant_order']}, even/odd={topo_data['even_odd_ratio']:.2f})")
                    
                except Exception as e:
                    print(f"Warning: Could not calculate hemisphere flux: {e}")
                    results['hemisphere_flux_evolution'].append(None)
                    results['topology_evolution'].append(None)
                    
                    # Fallback print if calculation fails
                    omega_str = f"{frame_drag_eff:.4f}" if (frame_drag_eff is not None and not np.isnan(frame_drag_eff)) else "N/A"
                    print(f"t={current_time:.1f}: B_horizon={horizon_field_strength:.2e}, "
                        f"topology={topology_ratio:.2f}, ΩF/ΩH(horizon)={omega_str}")
                
            except Exception as e:
                print(f"Error analyzing {dump_file}: {e}")
                continue
        
        # FIXED: Validation
        if results['frame_dragging_evolution']:
            final_omega = results['frame_dragging_evolution'][-1]
            print(f"\n=== VALIDATION ===")
            print(f"Final ΩF/ΩH at horizon: {final_omega['omega_ratio']:.4f}")
            print(f"Theoretical prediction: 0.500")
            print(f"Deviation: {abs(final_omega['omega_ratio'] - 0.5)/0.5 * 100:.1f}%")
        
        return results

    def plot_hair_loss_analysis(self, results, field_type="Auto", problem_name="", show=True):
        """
        Comprehensive hair loss analysis plot with hemisphere flux evolution and topology.
        
        Creates 2x3 grid:
        - Row 1: Field strength, Hemisphere flux, Field topology
        - Row 2: Frame dragging, Flux conservation, Summary
                
        Args:
            results: Analysis results dictionary
            field_type: Field type classification (Auto-detected if "Auto")
            problem_name: Problem name to display in title (e.g., "bz_monopole")
            show: Whether to display the plot
        """
        if not results['times']:
            return
        
        # Auto-detect field type TODO: replace this method with getting the field configuration from self
        if field_type == "Auto" and results['times']:
            dump_files = get_dump_files()
            if dump_files:
                field_type = self.detect_field_type(dump_files[-1])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Dynamic title with problem name
        title_map = {
            "Monopole": "Magnetic Field Evolution: 2D Monopole",
            "Dipole": "Magnetic Field Evolution: 2D Dipole", 
            "Mixed/Evolving": "Magnetic Field Evolution: Evolving Configuration",
            "1D": "Magnetic Field Evolution: 1D Field",
            "Unknown": "Magnetic Field Evolution"
        }
        
        title = title_map.get(field_type, f"Magnetic Field Evolution: {field_type}")
        
        # Add problem name if provided
        if problem_name:
            title = f"{title} - Problem: {problem_name}"
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        times = results['times']
        
        # ========================================================================
        # Panel 1: Field Strength Persistence
        # ========================================================================
        ax1 = axes[0, 0]
        ax1.semilogy(times, results['field_strength_horizon'], 'b-', linewidth=2)
        ax1.set_xlabel('Time (M)', fontsize=12)
        ax1.set_ylabel('B-field Strength (horizon)', fontsize=12)
        ax1.set_title('Field Persistence', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Stats
        initial_B = results['field_strength_horizon'][0]
        final_B = results['field_strength_horizon'][-1]
        decay_percent = (1 - final_B/initial_B) * 100 if initial_B > 0 else 0
        
        stat_text = f'Initial: {initial_B:.2e}\n'
        stat_text += f'Final: {final_B:.2e}\n'
        stat_text += f'Change: {decay_percent:+.1f}%'
        ax1.text(0.02, 0.98, stat_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        # ========================================================================
        # Panel 2: Hemisphere Flux Evolution
        # ========================================================================
        ax2 = axes[0, 1]
        
        if results.get('hemisphere_flux_evolution'):
            # Extract data
            flux_times = [d['time'] for d in results['hemisphere_flux_evolution'] if d is not None]
            flux_north = [d['flux_north'] for d in results['hemisphere_flux_evolution'] if d is not None]
            flux_south = [d['flux_south'] for d in results['hemisphere_flux_evolution'] if d is not None]
            flux_total = [d['flux_total'] for d in results['hemisphere_flux_evolution'] if d is not None]
            
            if flux_times:
                ax2.plot(flux_times, flux_north, 'b-', linewidth=2, label='Φ_north', marker='o', markersize=4)
                ax2.plot(flux_times, flux_south, 'r-', linewidth=2, label='Φ_south', marker='s', markersize=4)
                ax2.plot(flux_times, flux_total, 'k--', linewidth=2, label='Φ_total', alpha=0.7)
                ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                
                ax2.set_xlabel('Time (M)', fontsize=12)
                ax2.set_ylabel('Magnetic Flux', fontsize=12)
                ax2.set_title('Hemisphere Flux Evolution', fontsize=13, fontweight='bold')
                ax2.legend(loc='best', fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Classification
                final_flux = results['hemisphere_flux_evolution'][-1]
                if final_flux:
                    class_text = f'{final_flux["field_type"]}\n'
                    if final_flux['same_sign']:
                        class_text += 'Φ_N·Φ_S > 0'
                    else:
                        class_text += 'Φ_N·Φ_S < 0'
                    
                    ax2.text(0.02, 0.98, class_text, transform=ax2.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                    edgecolor='gray', alpha=0.9, linewidth=1))
        else:
            ax2.text(0.5, 0.5, 'Hemisphere flux\ndata not available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Hemisphere Flux Evolution', fontsize=13, fontweight='bold')
        
        # ========================================================================
        # Panel 3: Field Topology Evolution (Legendre-Based Multipole Analysis)
        # ========================================================================
        ax3 = axes[0, 2]
        
        if results.get('topology_evolution'):
            # Extract Legendre-based topology data
            topo_times = [d['time'] for d in results['topology_evolution'] if d is not None]
            dominant_orders = [d['dominant_order'] for d in results['topology_evolution'] if d is not None]
            even_odd_ratios = [d['even_odd_ratio'] for d in results['topology_evolution'] if d is not None]
            
            if topo_times:
                # Create twin axis for dual visualization
                ax3_twin = ax3.twinx()
                
                # Plot dominant multipole order
                line1 = ax3.plot(topo_times, dominant_orders, 'b-', linewidth=2, 
                               marker='o', markersize=6, label='Dominant l', alpha=0.8)
                ax3.set_xlabel('Time (M)', fontsize=12)
                ax3.set_ylabel('Dominant Multipole Order l', fontsize=12, color='b')
                ax3.tick_params(axis='y', labelcolor='b')
                ax3.set_ylim(-0.5, max(dominant_orders) + 0.5 if dominant_orders else 5)
                
                # Plot even/odd ratio on twin axis
                line2 = ax3_twin.plot(topo_times, even_odd_ratios, 'r-', linewidth=2,
                                     marker='s', markersize=6, label='Even/Odd Ratio', alpha=0.8)
                ax3_twin.set_ylabel('Even/Odd Power Ratio', fontsize=12, color='r')
                ax3_twin.tick_params(axis='y', labelcolor='r')
                ax3_twin.axhline(y=1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
                
                ax3.set_title('Field Topology (Legendre Multipoles)', fontsize=13, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add horizontal reference lines for multipole orders
                for l, label, color in [(0, 'Monopole (l=0)', 'blue'), 
                                        (1, 'Dipole (l=1)', 'orange'), 
                                        (2, 'Quadrupole (l=2)', 'green')]:
                    ax3.axhline(y=l, color=color, linestyle='--', alpha=0.2, linewidth=1.5)
                    ax3.text(times[-1]*0.02, l, f' {label}', fontsize=8, va='center', color=color, alpha=0.7)
                
                # Classification text box
                final_topo = results['topology_evolution'][-1]
                if final_topo:
                    topo_text = f'{final_topo["field_type"]}\n'
                    topo_text += f'l = {final_topo["dominant_order"]}\n'
                    # Handle confidence as either float or string
                    try:
                        conf_val = float(final_topo["confidence"])
                        topo_text += f'Confidence: {conf_val:.2f}'
                    except (ValueError, TypeError):
                        topo_text += f'Confidence: {final_topo["confidence"]}'
                    ax3.text(0.98, 0.98, topo_text, transform=ax3.transAxes,
                            fontsize=9, verticalalignment='top', horizontalalignment='right',
                            fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                    edgecolor='gray', alpha=0.9, linewidth=1))
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax3.legend(lines, labels, loc='best', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Topology data\nnot available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Field Topology Evolution', fontsize=13, fontweight='bold')
        
        # ========================================================================
        # Panel 4: Frame Dragging at Horizon
        # ========================================================================
        ax4 = axes[1, 0]
        ax4.plot(times, results['frame_dragging_efficiency'], 'g-', linewidth=2)
        ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, linewidth=2, label='BZ Theory = 0.500')
        ax4.set_xlabel('Time (M)', fontsize=12)
        ax4.set_ylabel('ΩF/ΩH', fontsize=12)
        ax4.set_title('Frame Dragging at Horizon', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Stats
        if results['frame_dragging_efficiency']:
            valid_omega = [x for x in results['frame_dragging_efficiency'] if not np.isnan(x)]
            if valid_omega:
                avg_omega = np.mean(valid_omega)
                std_omega = np.std(valid_omega)
                deviation_percent = abs(avg_omega - 0.5) / 0.5 * 100
                
                omega_text = f'⟨ΩF/ΩH⟩ = {avg_omega:.4f} ± {std_omega:.4f}\n'
                omega_text += f'Theory = 0.5000\n'
                omega_text += f'Δ = {deviation_percent:.1f}%'
                
                ax4.text(0.02, 0.98, omega_text, transform=ax4.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                edgecolor='gray', alpha=0.9, linewidth=1))
        
        ax4.legend()
        
        # ========================================================================
        # Panel 5: Flux Conservation
        # ========================================================================
        ax5 = axes[1, 1]
        flux_normalized = np.array(results['magnetic_flux_conservation'])
        if len(flux_normalized) > 0:
            flux_normalized = flux_normalized / flux_normalized[0]
        ax5.plot(times, flux_normalized, 'm-', linewidth=2)
        ax5.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Perfect Conservation')
        ax5.set_xlabel('Time (M)', fontsize=12)
        ax5.set_ylabel('Φ/Φ₀ (normalized)', fontsize=12)
        
        # Quantitative title
        max_flux_ratio = max(flux_normalized) if len(flux_normalized) > 0 else 1
        min_flux_ratio = min(flux_normalized) if len(flux_normalized) > 0 else 1
        
        if max_flux_ratio > 5:
            ax5.set_title(f'Magnetic Flux: {max_flux_ratio:.1f}× Amplification', 
                        fontsize=13, fontweight='bold', color='red')
        elif min_flux_ratio < 0.5:
            ax5.set_title(f'Magnetic Flux: {(1-min_flux_ratio)*100:.0f}% Dissipation',
                        fontsize=13, fontweight='bold')
        else:
            flux_variation = (max_flux_ratio - min_flux_ratio) / 1.0 * 100
            ax5.set_title(f'Magnetic Flux: {flux_variation:.1f}% Variation',
                        fontsize=13, fontweight='bold')
        
        # Stats
        flux_text = f'Initial: 1.00\n'
        flux_text += f'Final:   {flux_normalized[-1]:.2f}\n'
        flux_text += f'Max:     {max_flux_ratio:.2f}\n'
        flux_text += f'Min:     {min_flux_ratio:.2f}'
                
        ax5.text(0.02, 0.98, flux_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                        edgecolor='gray', alpha=0.9, linewidth=1))
            
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # ========================================================================
        # Panel 6: Summary
        # ========================================================================
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Keep this panel quantitative - just numbers, no interpretation:
        summary_text = "SUMMARY\n"
        summary_text += "─" * 25 + "\n\n"

        summary_text += f"Time: {times[0]:.0f} - {times[-1]:.0f} M\n"
        summary_text += f"Snapshots: {len(times)}\n\n"

        summary_text += f"ΔB/B_0: {decay_percent:+.1f}%\n\n"

        if results.get('hemisphere_flux_evolution'):
            final_flux = results['hemisphere_flux_evolution'][-1]
            if final_flux:
                summary_text += f"Field: {final_flux['field_type']}\n"
                summary_text += f"Φ_N: {final_flux['flux_north']:+.2e}\n"
                summary_text += f"Φ_S: {final_flux['flux_south']:+.2e}\n\n"

        if results.get('topology_evolution'):
            final_topo = results['topology_evolution'][-1]
            if final_topo:
                # Use Legendre-based metrics instead of n_crossings
                summary_text += f"Dominant l: {final_topo['dominant_order']}\n"
                summary_text += f"Type: {final_topo['field_type']}\n"
                # Handle confidence as either float or string
                try:
                    conf_val = float(final_topo['confidence'])
                    summary_text += f"Confidence: {conf_val:.2f}\n\n"
                except (ValueError, TypeError):
                    summary_text += f"Confidence: {final_topo['confidence']}\n\n"

        if valid_omega:
            summary_text += f"⟨ΩF/ΩH⟩: {avg_omega:.4f}\n"
            summary_text += f"σ: {std_omega:.4f}\n"
            summary_text += f"Δ: {deviation_percent:.1f}%"

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                        edgecolor='gray', alpha=0.95, linewidth=1.5))
        
        plt.tight_layout()
        
        # Save
        safe_field_type = field_type.lower().replace('/', '_').replace(' ', '_')
        filename = os.path.join(self.output_dir, 
                            f"magnetic_hair_loss_{safe_field_type}.png")
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved hair loss analysis: {filename}")
        
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

def handle_animation_requests(analyzer, args, dump_files):
    """
    Handle all animation-related requests
    Returns True if handled (should exit), False otherwise
    """
    if args.animations:
        print("\n=== GENERATING ALL ANIMATIONS ===")
        analyzer.generate_all_animations(
            dump_files,
            fps=args.fps,
            early_resolution_boost=True,
            problem_name=args.problem
        )
        print(f"✓ All animations saved to: {args.output}")
        return True
    
    if args.stag_anim:
        print("\n=== GENERATING VELOCITY & STAGNATION ANIMATION ===")
        filename = f"velocity_stagnation_{args.problem}.mp4"
        analyzer.create_velocity_and_stagnation_animation(
            dump_files, output_file=filename, fps=args.fps
        )
        output_path = os.path.join(args.output, filename)
        print(f"✓ Saved to: {output_path}")
        return True
    
    if args.energy_anim:
        print("\n=== GENERATING ENERGY ZONES ANIMATION ===")
        filename = f"energy_zones_{args.problem}.mp4"
        analyzer.create_energy_zones_animation(
            dump_files, output_file=filename, fps=args.fps
        )
        output_path = os.path.join(args.output, filename)
        print(f"✓ Saved to: {output_path}")
        return True
    
    if args.mag_anim:
        print("\n=== GENERATING MAGNETIC TOPOLOGY ANIMATION ===")
        filename = f"magnetic_topology_{args.problem}.mp4"
        analyzer.create_magnetic_topology_animation(
            dump_files, output_file=filename, fps=args.fps
        )
        output_path = os.path.join(args.output, filename)
        print(f"✓ Saved to: {output_path}")
        return True
    
    return False

def handle_analysis_requests(analyzer, args, dump_files):
    """
    Handle standard analysis requests.
    Dispatches the appropriate analysis based on command-line arguments.
    
    Returns True if handled (should exit), False otherwise
    
    Args:
        analyzer: MagnetizedAnalysis instance
        args: Parsed command-line arguments
        dump_files: List of dump files to analyze
    """
    # ========================================================================
    # Handle hair loss analysis
    # ========================================================================
    if args.hair_loss:
        print(f"\n{'='*70}")
        print("BLACK HOLE 'HAIR LOSS' ANALYSIS (SOMA 2017)")
        print(f"{'='*70}")
        print(f"Problem configuration: {args.problem}")
        
        # Determine field type label from problem name
        field_labels = {
            'bz_monopole': 'Monopole',
            'bz_dipole': 'Dipole',
            'bz_split_monopole': 'Split Monopole',
            'monopole_1d': 'Monopole',
            'monopole_2d': 'Monopole'
        }
        expected_field = field_labels.get(args.problem, 'Auto')
        
        # Run hair loss analysis
        hair_results = analyzer.analyze_magnetic_hair_loss(dump_files, sample_every=args.sample)
        
        # Use detected field type or expected from problem name
        if hair_results['times']:
            detected_type = analyzer.detect_field_type(dump_files[-1])
            print(f"\nExpected field type: {expected_field}")
            print(f"Detected field type: {detected_type}")
            
            # Use expected field type for labeling (from --problem flag)
            field_type_label = expected_field
        else:
            field_type_label = "Unknown"
        
        # Generate enhanced plots with problem name
        analyzer.plot_hair_loss_analysis(hair_results, field_type=field_type_label, problem_name=args.problem)
        
        # Generate frame dragging animation
        print(f"\n{'='*70}")
        print("GENERATING FRAME DRAGGING ANIMATION")
        print(f"{'='*70}")
        filename = f"frame_dragging_{args.problem}.mp4"
        analyzer.create_frame_dragging_animation(
            dump_files, 
            output_file=filename, 
            fps=args.fps,
            sample_every=args.sample,
            problem_name=args.problem
        )
        
        
        # Print validation summary
        if hair_results['frame_dragging_efficiency']:
            valid_omega = [x for x in hair_results['frame_dragging_efficiency'] 
                          if x is not None and not np.isnan(x)]
            if valid_omega:
                avg_omega = np.mean(valid_omega)
                std_omega = np.std(valid_omega)
                deviation = abs(avg_omega - 0.5) / 0.5 * 100
                print(f"\n{'='*70}")
                print("SUMMARY")
                print(f"{'='*70}")
                print(f"Average ΩF/ΩH = {avg_omega:.4f} ± {std_omega:.4f}")
                print(f"Theory (monopole) = 0.500")
                print(f"Deviation: {deviation:.1f}%")
                if deviation < 5:
                    print("✓ Excellent agreement with BZ mechanism")
                elif deviation < 10:
                    print("✓ Good agreement with BZ mechanism")
                else:
                    print("⚠ Significant deviation - check resolution/boundary conditions")
            else:
                print("\n⚠ ΩF/ΩH data contains only NaN values")
        
        print(f"\n✓ Results saved to: {args.output}")
        return True  # Handled, should exit
    
    # ========================================================================
    # Handle standard analysis for specified problem types
    # ========================================================================
    if args.problem in ['monopole_1d', 'monopole_2d', 'bz_monopole', 'bz_dipole', 'bz_split_monopole']:
        print(f"\n{'='*70}")
        print(f"STANDARD ANALYSIS FOR {args.problem.upper()}")
        print(f"{'='*70}")
        
        # Detect field type
        detected_type = analyzer.detect_field_type(dump_files[-1])
        print(f"Detected field type: {detected_type}")
        
        # Run 2D monopole analysis
        results = analyzer.analyze_2d_monopole(dump_files, sample_every=args.sample)
        analyzer.plot_2d_monopole_results(results, field_type=detected_type, problem_name=args.problem)
        
        # Generate animations for full analysis
        print(f"\n{'='*70}")
        print(f"GENERATING ANIMATIONS FOR {args.problem.upper()}")
        print(f"{'='*70}")
        analyzer.generate_all_animations(
            dump_files,
            fps=10,
            early_resolution_boost=True,
            problem_name=args.problem
        )

        print(f"\n✓ Analysis complete!")
        print(f"Results saved to: {args.output}")
        return True  # Handled, should exit
    
    # If no specific analysis was handled, return False to allow default behavior
    return False


def setup_argparse():
    """
    Simplified argument parser - only keep what you actually use
    """
    parser = argparse.ArgumentParser(
        description='Magnetized Accretion Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all animations
  python magnetized_analysis.py --animations
  
  # Generate specific animation
  python magnetized_analysis.py --stag-anim --fps 15
  
  # Full analysis
  python magnetized_analysis.py --problem bz_monopole
        """
    )
    
    # Core options
    parser.add_argument('--problem', type=str, default='bz_monopole',
                       choices=['monopole_1d', 'monopole_2d', 'bz_monopole', 'bz_dipole', 'bz_split_monopole'],
                       help='Problem type / field configuration to analyze')
    parser.add_argument('--output', type=str, default='./magnetized_plots',
                       help='Output directory for results')
    parser.add_argument('--hair-loss', action='store_true',
                       help='Run black hole "hair loss" analysis (SOMA 2017 style)')
    
    # Animation options TODO: add frame dragging animation
    anim_group = parser.add_argument_group('Animation Options')
    anim_group.add_argument('--animations', action='store_true',
                           help='Generate all animations')
    anim_group.add_argument('--stag-anim', action='store_true',
                           help='Generate velocity/stagnation animation only')
    anim_group.add_argument('--energy-anim', action='store_true',
                           help='Generate energy zones animation only')
    anim_group.add_argument('--mag-anim', action='store_true',
                           help='Generate magnetic topology animation only')
    anim_group.add_argument('--fps', type=int, default=10,
                           help='Frames per second for animations (default: 10)')
    
    # Analysis options (if you use them)
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--sample', type=int, default=5,
                               help='Sample every N-th dump file for analysis')
    
    return parser


def main():
    """
    Streamlined main function - delegates to helper functions
    """
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize analyzer
    analyzer = MagnetizedAnalysis(output_dir=args.output)
    
    # Get dump files
    dump_files = get_dump_files()
    if not dump_files:
        print("ERROR: No dump files found!")
        print("Looking for: dumps/dump[0-9][0-9][0-9]")
        return
    
    print(f"Found {len(dump_files)} dump files")
    print(f"Range: {dump_files[0]} to {dump_files[-1]}")
    
    # ========================================================================
    # PRIORITY 1: Handle animation requests (exit after completion)
    # ========================================================================
    if handle_animation_requests(analyzer, args, dump_files):
        return  # Animation generated, exit
    
    # ========================================================================
    # PRIORITY 2: Handle analysis requests (hair-loss, standard, etc.)
    # ========================================================================
    if handle_analysis_requests(analyzer, args, dump_files):
        return  # Analysis complete, exit
    
    # ========================================================================
    # DEFAULT: If no flags provided, inform user
    # ========================================================================
    print(f"\nNo analysis specified. Use --help to see available options.")
    print(f"Common commands:")
    print(f"  --problem bz_monopole           # Run standard analysis")
    print(f"  --hair-loss --problem bz_dipole # Run hair loss analysis")
    print(f"  --animations                    # Generate all animations")


if __name__ == "__main__":
    main()
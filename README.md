# Anisotropic Womersley Flow Solver

## This repository contains the code used in the following paper:
* **Paper Title:** A transverse picoNewton force revealed in anisotropic Womersley flow
* **Author:** Khalid M. Saqr
* **Contact:** k[dot]saqr[at]aast[dot]edu

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).

[![CC BY-NC-ND 4.0](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Context and Motivation

The fundamental fluid-dynamic quantity that regulates endothelial mechanotransduction in the human arterial system remains a highly debated subject. Historically, Wall Shear Stress (WSS) has been utilized to characterize the boundary traction exerted by flowing blood on the endothelial lining. However, WSS is strictly a surface traction metric and inherently fails to represent the volumetric, non-conservative inertial structure of arterial blood flow within the near-wall boundary layer.

Furthermore, blood is a complex suspension that exhibits direction-dependent stress under physiological shear rates. This suggests that the classical Womersley solution of the Navier-Stokes equations—which assumes a purely Newtonian, isotropic fluid—omits constitutive mechanisms that are critically relevant to near-wall hemodynamics. 

This repository provides the complete numerical and computational framework developed to derive and analyze an anisotropic extension of Womersley flow. By introducing a tensorial viscosity into the incompressible Navier-Stokes equations, the provided Python models evaluate the nonlinear interaction of velocity and vorticity within a precisely defined near-wall control volume. 

The results demonstrate that anisotropic viscosity produces a non-trivial spectral signature in the transverse forcing, maintaining power across higher-order cardiac harmonics. While macroscopic geometric drivers dominate the bulk flow at the fundamental cardiac frequency, they are subject to significant inertial dampening as the harmonic frequency increases. In contrast, the anisotropy-induced Lamb vector, sustained by the sharp velocity gradients of the oscillatory boundary layer, evades this macroscopic attenuation, revealing a transverse force on the order of picoNewtons (pN).

---

## Mathematical Formulation and Governing Equations



The computational codebase rigorously anchors its execution on the spectral collocation solution of the anisotropic Navier-Stokes momentum equations, structured in a cylindrical coordinate system $(r, \theta, z)$. 

### The Anisotropic Womersley System

The solver evaluates the complex velocity amplitudes for the axial component ($\hat{u}_ {z,h}$) and the azimuthal (swirl) component ($\hat{u}_ {\theta,h}$) across a spectrum of harmonic frequencies designated by $h$. The pressure gradient forcing amplitude for each harmonic is denoted as $a_h$. The differential equations are governed by two distinct linear operators, $L_0$ and $L_1$, which dictate the spatial derivatives in the radial direction.

The coupled system is defined by the following matrices:

```math
\hat{A}_{zz} = (i \cdot f_h \cdot \alpha^2) I - L_0
```

```math
\hat{A}_{z\theta} = -\beta L_1
```

Where:
* $\alpha$ represents the dimensionless Womersley number, characterizing the pulsatile flow frequency relative to viscous effects.
* $\beta$ represents the anisotropy ratio, introducing the tensorial viscosity component.
* $f_h$ is the normalized frequency for harmonic $h$.

### Boundary Conditions

The coupled ordinary differential equations enforce strict physiological boundary conditions to ensure physical validity:

1. **Centerline Symmetry** ($r=0$): A Neumann boundary condition is applied to the axial velocity ($d\hat{u}_ {z,h}/dr = 0$), and a Dirichlet regularity condition is applied to the azimuthal velocity ($\hat{u}_ {\theta,h} = 0$).
2. **Wall No-Slip** ($r=1$): A strict Dirichlet no-slip condition is enforced at the vessel wall for both components ($\hat{u}_ {z,h} = 0$ and $\hat{u}_ {\theta,h} = 0$).

### Gromeka-Lamb Decomposition and the Lamb Vector



To quantify the inertial forcing independent of classical shear, the framework utilizes the Gromeka-Lamb decomposition of the Navier-Stokes equations. This defines the Lamb vector ($l = u \times \omega$), which represents the non-conservative inertial field capturing local velocity-vorticity coupling. 

Using the spatial derivatives computed via the spectral matrices, the vorticity components are calculated as:

* **Axial Vorticity:**

```math
\omega_z = \frac{1}{r} \frac{d(r u_\theta)}{dr}
```

* **Azimuthal Vorticity:**

```math
\omega_\theta = - \frac{du_z}{dr}
```

The transverse (radial) component of the Lamb vector is subsequently computed as:

```math
l_r = u_\theta \omega_z - u_z \omega_\theta
```

### Time-Domain Reconstruction

Because the Lamb vector is a mathematically non-linear entity, the individual frequency-domain harmonics cannot be superposed after the vector product. The algorithm reconstructs the physical (real) time-domain signals prior to calculating the nonlinear interactions using the following summation:

```math
u_{j}(r,t) = \text{Re} \left( \sum_{h=1}^{6} \hat{u}_{j,h}(r) e^{i 2\pi h t} \right)
```

### Endothelial Control Volume Integration

The local near-wall inertial force acting upon a single endothelial cell ($F_{EC}$) is resolved by scaling the Lamb vector magnitude by the volumetric force density scale ($\rho U_0^2 / R$). This continuous density field is computationally integrated across a standard endothelial cell pillbox control volume ($V_{EC} = 10^{-15} \text{ m}^3$) with a specific cellular footprint ($A_{EC} = 100 \times 10^{-12} \text{ m}^2$). This yields a cumulative force metric natively calibrated in picoNewtons (pN).

---

## Numerical Methods and Code Construction



Standard finite volume methodologies introduce excessive numerical diffusion when resolving the incredibly sharp gradients present in near-wall oscillatory boundary layers. Therefore, this project relies strictly on high-order numerical techniques.

### Chebyshev-Gauss-Lobatto Spectral Discretization
The radial domain is discretized using a mapped spectral collocation method. The default grid density is set to $N=150$ nodes. As verified in the supplementary grid-independence studies, $N \ge 140$ reduces the residual magnitude to $\mathcal{O}(10^{-14})$, ensuring double-precision spectral convergence. The standard Chebyshev interval $x \in [-1, 1]$ is systematically mapped to the normalized physical radial domain $r \in [0, 1]$ via the algebraic transformation $r = (1 - x) / 2$.

### Differentiation and Sparse Linear Algebra
Spatial derivatives are formulated using dense Chebyshev spectral differentiation matrices. The resulting discretized, coupled equations form a highly structured block matrix system. This system is solved using direct sparse linear algebra routines (`scipy.sparse.linalg.spsolve`) to allow for rapid parameter sweeps across the $(\alpha, \beta)$ phase space. Barycentric polynomial interpolation is utilized to evaluate fields without introducing interpolation artifacts.

### Spatial Integration and Spectral Decomposition
The control volume integration leverages the trapezoidal rule (`scipy.integrate.trapezoid`), constrained strictly to the localized collocation nodes residing within the physical height of the endothelial cell. Finally, `numpy.fft.rfft` is applied to the aggregated time-domain force signals to extract the power retained across the distinct physiological harmonics ($H_1, H_2, H_3$).

---

## Repository Structure

The repository is divided into two sequential computational notebooks, highly optimized for execution in Google Colaboratory or local Jupyter environments.

### `picoNewton_v1.ipynb` (Mono-Harmonic Sweeps)
This script conducts the foundational parametric sweeps to map the theoretical phase space.
* Iterates the Womersley number ($\alpha$) from 1 to 25.
* Iterates the anisotropy ratio ($\beta$) from 0.0 to 0.2.
* Calculates the resulting dimensionless velocity and vorticity fields.
* Isolates and plots the maximum amplitude of azimuthal velocity ($|\hat{u}_{\theta}|_{\max}$) and axial vorticity ($|\hat{\omega}_z|_{\max}$).
* Computes the baseline near-wall transverse inertial force.

### `picoNewton_v2.ipynb` (Multi-Harmonic Physiological Realities)
This script handles the complex, real-world physiological waveforms.
* Injects digitized flow waveforms representing six specific arterial sites: Aortic Root, Thoracic Aorta, Femoral, Carotid, Iliac, and Brachial arteries.
* Reconstructs the full nonlinear time-domain signals.
* Executes the exact near-wall spatial integration over the $10^{-15} \text{ m}^3$ control volume.
* Outputs the spectral decomposition tables, tracking the transverse picoNewton forces under distinct cardiac harmonic constraints.

---

## Installation, Setup, and Execution

These notebooks are self-contained and require standard Python scientific computing libraries. The computational environment requires:

* `numpy` (Matrix manipulation and discrete Fourier transforms)
* `scipy` (Sparse matrix operations, linear algebra, and numerical integration)
* `matplotlib` (Contour mapping and publication-grade data visualization)
* `pandas` (Tabular spectral summaries)
* `tqdm` (Progress tracking for nested parameter sweeps)

### Typography Dependencies

To accurately reproduce the publication-grade figures containing specific LaTeX fonts, the host machine or Colab instance must install the following TeX distribution packages. The notebooks contain a shell command to execute this automatically at runtime:

```bash
sudo apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

### Reproducing the Paper's Results

Both notebooks are structured with a master `main()` function block. Running the notebook sequentially from top to bottom will automatically apply the required Matplotlib `mathptmx` plot styling, initialize the numerical solver, and generate the respective figures exactly as they appear in the manuscript.

To manually invoke the core engine within a custom Python script or separate analysis pipeline, use the following syntax:

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import trapezoid

# 1. Initialize the solver with spectrally converged grid density (N=150)
solver = WomersleySolver(N=150)

# 2. Replicate Figures 1 and 2 (Mono-harmonic parametric sweeps)
make_figure1_and_2(solver)

# 3. Replicate Figures 3 through 6 (Physiological temporal maps)
make_figures_3_to_6(solver)

# 4. Generate the Final Endothelial Transverse Force Table (pN)
run_genius_spectral_analysis(solver)
```

Upon successful execution, the pipeline automatically generates a local `./figures/` directory containing both high-resolution `.png` and vector-based `.pdf` formats of the simulated data.# Anisotropic Womersley Flow Solver

## Code Nomenclature

## 1. Grid and Spatial Discretization Variables
| Variable | Description | Mathematical Notation |
| :--- | :--- | :--- |
| `D` | Dense Chebyshev spectral differentiation matrix | $D$ |
| `N` | Number of Chebyshev-Gauss-Lobatto collocation nodes | $N$ |
| `r` | Dimensionless radial coordinate mapped to physical vessel | $r \in [0, 1]$ |
| `x` | Standard Chebyshev collocation nodes interval | $x \in [-1, 1]$ |

## 2. Physical and Physiological Parameters
| Variable | Description | Mathematical Notation |
| :--- | :--- | :--- |
| `A_ EC` | Prescribed endothelial cell footprint area | $A_{EC}$ |
| `alpha` | Dimensionless Womersley number | $\alpha$ |
| `beta` | Anisotropy ratio (tensorial viscosity magnitude) | $\beta$ |
| `G_ 0` | Baseline or fundamental pressure gradient | $G_0$ |
| `R` | Physical arterial radius | $R$ |
| `rho` | Blood density | $\rho$ |
| `U_ 0` | Characteristic velocity scale | $U_0$ |
| `V_ EC` | Endothelial cell pillbox control volume | $V_{EC}$ |

## 3. Spectral and Harmonic Variables
| Variable | Description | Mathematical Notation |
| :--- | :--- | :--- |
| `a_ h` | Complex pressure gradient amplitude for a specific harmonic | $a_h$ |
| `f_ h` | Normalized frequency for a specific harmonic | $f_h$ |
| `h` | Cardiac harmonic index | $h$ |

## 4. Differential Operators
| Variable | Description | Mathematical Notation |
| :--- | :--- | :--- |
| `L_ 0` | Linear differential operator for axial velocity spatial derivatives | $L_0$ |
| `L_ 1` | Linear differential operator for azimuthal velocity spatial derivatives | $L_1$ |

## 5. Flow Field and Force Variables
| Variable | Description | Mathematical Notation |
| :--- | :--- | :--- |
| `F_ EC` | Cumulative near-wall transverse inertial force per endothelial cell | $F_{EC}$ |
| `l_ r` | Transverse (radial) component of the non-conservative Lamb vector | $l_r$ |
| `omega_ theta` | Azimuthal (swirl) vorticity component | $\omega_\theta$ |
| `omega_ z` | Axial vorticity component | $\omega_z$ |
| `u_ theta_ h` | Complex amplitude of the azimuthal (swirl) velocity | $\hat{u}_{\theta,h}$ |
| `u_ z_ h` | Complex amplitude of the axial velocity | $\hat{u}_{z,h}$ |


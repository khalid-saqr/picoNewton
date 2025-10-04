# Anisotropic Womersley Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official open-source Python implementation for the paper:

**"A transverse piconewton-scale force field revealed in anisotropic Womersley flow"** *by Khalid M. Saqr* [Placeholder for Paper DOI/Link]

---

## Overview

This code provides the definitive numerical solution for the pulsatile flow of an incompressible, anisotropic Newtonian fluid in a rigid cylindrical vessel. It is based on an exact extension of the classic Womersley flow model to account for the tensorial nature of blood viscosity.

The central thesis, proven by this code, is that axial-to-transverse momentum transfer, induced by blood's anisotropic viscosity, generates a significant, radially-directed force field on the endothelium. [cite_start]This work is situated within a new paradigm that considers physiologic blood flow not as laminar, but as an inherently complex, non-Kolmogorov turbulent state[cite: 1]. [cite_start]The transverse force identified here is proposed as a fundamental, analytically tractable component of the near-wall "vortex force field" that has been previously identified as a key regulator of mechanobiological stimulation in complex 3D flows[cite: 2].

The script runs a complete, two-stage analysis to demonstrate this phenomenon:
1.  **Stage 1:** A mono-harmonic parametric sweep to establish the fundamental physics.
2.  **Stage 2:** An extended comparative analysis using high-fidelity, digitized physiological waveforms for six major human arteries.

## Theoretical Framework

The solver is based on the frequency-domain solution of the incompressible Navier-Stokes equations in cylindrical coordinates, assuming an axisymmetric flow ($u_r=0, \partial/\partial\theta = 0$).

### Governing Equations

The axial ($u_z$) and azimuthal ($u_\theta$) momentum equations are:

$$
\rho \frac{\partial u_{z}}{\partial t}=-\frac{\partial p}{\partial z}+\frac{1}{r}\frac{\partial}{\partial r}(r\tau_{zr}) \quad(1)
$$

$$
\rho \frac{\partial u_{\theta}}{\partial t}=\frac{1}{r^{2}}\frac{\partial}{\partial r}(r^{2}\tau_{\theta r}) \quad(2)
$$

### Anisotropic Constitutive Law

The key innovation is the use of a tensorial kinematic viscosity, where the off-diagonal terms ($\nu_{z\theta}, \nu_{\theta z}$) couple the axial and azimuthal momentum:

$$
\begin{bmatrix} \tau_{zr} \\ \tau_{\theta r} \end{bmatrix} = \rho \begin{bmatrix} \nu_{zz} & \nu_{z\theta} \\ \nu_{\theta z} & \nu_{\theta\theta} \end{bmatrix} \begin{bmatrix} \partial_r u_z \\ \partial_r u_\theta - u_\theta/r \end{bmatrix} \quad(3)
$$

### Dimensionless Formulation

The system is non-dimensionalized and solved in the frequency domain for each harmonic, $h$. This transforms the time derivatives ($\partial/\partial t$) into multiplications by $i \omega_h$, resulting in a system of coupled ordinary differential equations:

$$
i f_{h}\alpha^{2}\hat{U}_{z}^{\ast} = a_{h} + L_{0}^{\ast}\hat{U}_{z}^{\ast} + \beta L_{1}^{\ast}\hat{U}_{\theta}^{\ast} \quad(4)
$$

$$
i f_{h}\alpha^{2}\hat{U}_{\theta}^{\ast} = \gamma L_{0}^{\ast}\hat{U}_{z}^{\ast} + \delta L_{1}^{\ast}\hat{U}_{\theta}^{\ast} \quad(5)
$$

where $\alpha$ is the Womersley number, $a_h$ is the dimensionless pressure gradient amplitude for the $h$-th harmonic, and $\beta, \gamma, \delta$ are the dimensionless anisotropy ratios. The linear differential operators are defined as $L_0 f = f'' + f'/r$ and $L_1 f = L_0 f - f/r^2$.

### Transverse Force Calculation

The resulting transverse (radial) force on the endothelium is calculated from the Gromeka-Lamb vector, evaluated in the near-wall fluid layer:

$$
\mathbf{f}_r = \rho(\mathbf{u} \times \boldsymbol{\omega}) \cdot \mathbf{e}_r = \rho(u_\theta \omega_z - u_z \omega_\theta) \quad(6)
$$

---

## Code Implementation Details

The numerical solution is encapsulated in the `WomersleySolver` class.

-   **Numerical Method:** The solver employs a **spectral collocation method** using Chebyshev-Gauss-Lobatto nodes. This high-order method provides exponential convergence for smooth solutions, ensuring high accuracy in calculating velocity and vorticity fields. The discretization and operator construction are performed in the `_setup_discretization` method.
-   **System Assembly:** For each harmonic, the coupled ODEs are formulated as a $2(N+1) \times 2(N+1)$ sparse block-matrix system, as shown in the `solve_harmonic` method.
-   **Solution:** The sparse linear system is solved efficiently using a direct LU factorization via `scipy.sparse.linalg.spsolve`.
-   **Physical Scaling:** The code uses a physically consistent scaling framework. A reference dimensional pressure gradient (`G0_ref`) is defined, and from this, case-specific characteristic velocities (`U0_case`) are derived. [cite_start]This ensures that the comparison between different arteries is physically meaningful, a critical detail that addresses the shortcomings of previous analyses that used inconsistent scaling[cite: 3, 4].

## How to Use and Parametrize the Code

The script `final_verified_solver.py` is designed to be run non-interactively to ensure full reproducibility of the paper's results.

```bash
python final_verified_solver.py

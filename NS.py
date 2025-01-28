"""
Created on Fri Jan 24 23:09:26 2025

@author: Shahab Golshan
"""

import numpy as np
import math

# ---------------------------
# Parameters
# ---------------------------
nx = 64       # number of grid points in x
ny = 64       # number of grid points in y
lx = 1.0      # domain length in x
ly = 1.0      # domain length in y
dx = lx / (nx - 1)
dy = ly / (ny - 1)

# Physical parameters
rho = 1.0     # density
nu = 0.01     # viscosity

# Time parameters
dt = 0.001    # time step
nt = 500      # number of time steps

# Write VTK at every nOutputStep steps
nOutputStep = 1  

# ---------------------------
# Initialization
# ---------------------------
u = np.zeros((ny, nx), dtype=np.float64)  # velocity in x-direction
v = np.zeros((ny, nx), dtype=np.float64)  # velocity in y-direction
p = np.zeros((ny, nx), dtype=np.float64)  # pressure

un = np.zeros_like(u)
vn = np.zeros_like(v)
pn = np.zeros_like(p)

# Create coordinate arrays (for VTK)
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)

def build_up_b(u, v, rho, dt, dx, dy):
    """
    Build the right-hand side of the Poisson equation for pressure.
    """
    b = np.zeros_like(u)
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            dudx = (u[j, i+1] - u[j, i-1]) / (2*dx)
            dvdy = (v[j+1, i] - v[j-1, i]) / (2*dy)
            dudy = (u[j+1, i] - u[j-1, i]) / (2*dy)
            dvdx = (v[j, i+1] - v[j, i-1]) / (2*dx)
            
            b[j, i] = rho * (
                1.0/dt * (dudx + dvdy) 
                - dudx**2 
                - 2.0*dudy*dvdx
                - dvdy**2
            )
    return b

def pressure_poisson(p, b, dx, dy):
    """
    Solve Poisson equation for pressure using Jacobi iterations (simple demo).
    """
    pn = np.empty_like(p)
    
    for _ in range(50):  # number of Jacobi iterations
        pn[:] = p[:]
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                p[j, i] = ((pn[j, i+1] + pn[j, i-1])*dy**2 +
                           (pn[j+1, i] + pn[j-1, i])*dx**2 -
                           b[j, i]*dx**2*dy**2) / (2*(dx**2+dy**2))

        # Boundary conditions for p
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x=1
        p[:, 0]  = p[:, 1]   # dp/dx = 0 at x=0
        p[-1, :] = 0.0       # p = 0 at y=1 or dp/dy=0
        p[0, :]  = p[1, :]   # dp/dy = 0 at y=0

def cavity_step(u, v, p, dt, dx, dy, b, rho, nu):
    """
    One time step of 2D Navier-Stokes with boundary conditions.
    """
    un[:] = u[:]
    vn[:] = v[:]
    
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            u[j, i] = (un[j, i] 
                       - un[j, i]*(dt/dx)*(un[j, i] - un[j, i-1])
                       - vn[j, i]*(dt/dy)*(un[j, i] - un[j-1, i])
                       - dt/(2*rho*dx)*(p[j, i+1] - p[j, i-1])
                       + nu*((dt/dx**2)*(un[j, i+1] - 2*un[j, i] + un[j, i-1])
                             + (dt/dy**2)*(un[j+1, i] - 2*un[j, i] + un[j-1, i])) )

            v[j, i] = (vn[j, i]
                       - un[j, i]*(dt/dx)*(vn[j, i] - vn[j, i-1])
                       - vn[j, i]*(dt/dy)*(vn[j, i] - vn[j-1, i])
                       - dt/(2*rho*dy)*(p[j+1, i] - p[j-1, i])
                       + nu*((dt/dx**2)*(vn[j, i+1] - 2*vn[j, i] + vn[j, i-1])
                             + (dt/dy**2)*(vn[j+1, i] - 2*vn[j, i] + vn[j-1, i])) )

    # Lid velocity
    u[-1, :] = 1.0
    v[-1, :] = 0.0
    # Other walls
    u[0, :] = 0.0
    v[0, :] = 0.0
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    u[:, -1] = 0.0
    v[:, -1] = 0.0


def write_vtk_structured(filename, x, y, u, v, p):
    """
    Write structured grid data in legacy VTK format so it can be opened in ParaView.
    """
    ny, nx = u.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Lid-driven cavity result\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx} {dy} 1\n")
        f.write(f"POINT_DATA {nx*ny}\n")

        # Velocity
        f.write("VECTORS velocity float\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{u[j,i]} {v[j,i]} 0.0\n")

        # Pressure
        f.write("\nSCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{p[j,i]}\n")

# ---------------------------
# Time stepping (write VTK each time step)
# ---------------------------
for n in range(nt):
    # Build RHS for pressure
    b = build_up_b(u, v, rho, dt, dx, dy)
    # Solve for pressure
    pressure_poisson(p, b, dx, dy)
    # Update velocity
    cavity_step(u, v, p, dt, dx, dy, b, rho, nu)
    
    # Write VTK at every nOutputStep
    if n % nOutputStep == 0:
        filename = f"cavity_fd_{n:04d}.vtk"
        write_vtk_structured(filename, x, y, u, v, p)
        print(f"Time step {n}: wrote {filename}")

print("Finite-difference lid-driven cavity flow simulation completed.")

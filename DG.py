# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 23:09:26 2025

@author: Shahab Golshan
"""

import torch
import torch.nn as nn
import numpy as np

# ---------------------------
# Neural Network Model
# ---------------------------
class NavierStokesNet(nn.Module):
    """
    A network that takes in (x,y) and outputs (u,v,p).
    """
    def __init__(self, hidden_dim=64, num_hidden_layers=5):
        super(NavierStokesNet, self).__init__()
        layers = []
        in_features = 2
        out_features = hidden_dim
        
        # Input layer
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.Tanh())  # activation
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(out_features, out_features))
            layers.append(nn.Tanh())
        
        # Final layer: outputs (u, v, p)
        layers.append(nn.Linear(out_features, 3))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y):
        inp = torch.stack([x, y], dim=1)
        out = self.net(inp)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

def navier_stokes_res(u, v, p, x, y, nu, rho):
    """
    Compute PDE residuals for the interior points.
    We'll treat it as steady-state, so partial_t=0.
    PDE:
       (u du/dx + v du/dy) = -1/rho dp/dx + nu Laplacian(u)
       (u dv/dx + v dv/dy) = -1/rho dp/dy + nu Laplacian(v)
       continuity: du/dx + dv/dy = 0
    """
    # Automatic differentiation
    grads_u = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    grads_v = torch.autograd.grad(v, [x, y], grad_outputs=torch.ones_like(v), create_graph=True)
    grads_p = torch.autograd.grad(p, [x, y], grad_outputs=torch.ones_like(p), create_graph=True)
    
    u_x = grads_u[0]
    u_y = grads_u[1]
    v_x = grads_v[0]
    v_y = grads_v[1]
    p_x = grads_p[0]
    p_y = grads_p[1]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    # PDE residuals
    r1 = u * u_x + v * u_y + (1.0/rho)*p_x - nu * (u_xx + u_yy)
    r2 = u * v_x + v * v_y + (1.0/rho)*p_y - nu * (v_xx + v_yy)
    r3 = u_x + v_y  # incompressibility
    
    return r1, r2, r3

def boundary_conditions_loss(model, x, y, boundary_type="other"):
    """
    Returns MSE of (u - bc_u, v - bc_v) at the boundary.
    boundary_type='top' => (u=1, v=0)
    boundary_type='other' => (u=0, v=0)
    """
    u_pred, v_pred, p_pred = model(x, y)
    if boundary_type == 'top':
        return ((u_pred - 1.0)**2 + (v_pred - 0.0)**2).mean()
    else:
        return ((u_pred - 0.0)**2 + (v_pred - 0.0)**2).mean()

# ---------------------------
# Training Setup
# ---------------------------
# Check for NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")

# Check for Apple Silicon GPU (MPS)
elif torch.backends.mps.is_available():
    device = torch.device("mps")

# Default to CPU
else:
    device = torch.device("cpu")

model = NavierStokesNet(hidden_dim=64, num_hidden_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nu_value = 0.01
rho_value = 1.0

# Generate collocation points in the interior
N_interior = 2000
xy_interior = np.random.rand(N_interior, 2)  # uniform in [0,1]^2
xy_interior = torch.tensor(xy_interior, dtype=torch.float32, device=device, requires_grad=True)

# Generate boundary points
N_boundary = 200

# Left (x=0)
x_b_left = torch.zeros(N_boundary, device=device)
y_b_left = torch.rand(N_boundary, device=device)

# Right (x=1)
x_b_right = torch.ones(N_boundary, device=device)
y_b_right = torch.rand(N_boundary, device=device)

# Bottom (y=0)
x_b_bottom = torch.rand(N_boundary, device=device)
y_b_bottom = torch.zeros(N_boundary, device=device)

# Top (y=1)
x_b_top = torch.rand(N_boundary, device=device)
y_b_top = torch.ones(N_boundary, device=device)

n_epochs = 5000
print_interval = 500
# We'll also save a VTK file every nOutputEpoch
nOutputEpoch = 10

# Create a grid for visualization / saving predictions
nx = 50
ny = 50
x_lin = np.linspace(0, 1, nx)
y_lin = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_lin, y_lin)
X_torch = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
Y_torch = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)

def write_vtk_structured_dgm(filename, X, Y, u, v, p):
    """
    Write 2D arrays (ny, nx) to VTK for ParaView.
    """
    ny, nx = X.shape
    dx = X[0,1] - X[0,0]
    dy = Y[1,0] - Y[0,0] if ny>1 else 0
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("PINN-lid-driven-cavity\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx} {dy} 1\n")
        f.write(f"POINT_DATA {nx*ny}\n")

        # velocity
        f.write("VECTORS velocity float\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{u[j,i]} {v[j,i]} 0.0\n")

        # pressure
        f.write("\nSCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for j in range(ny):
            for i in range(nx):
                f.write(f"{p[j,i]}\n")

# ---------------------------
# Training loop
# ---------------------------
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    # interior PDE loss
    x_int = xy_interior[:,0]
    y_int = xy_interior[:,1]
    u_int, v_int, p_int = model(x_int, y_int)
    r1, r2, r3 = navier_stokes_res(u_int, v_int, p_int, x_int, y_int, nu_value, rho_value)
    pde_loss = (r1**2 + r2**2 + r3**2).mean()
    
    # boundary losses
    loss_bc = 0.0
    loss_bc += boundary_conditions_loss(model, x_b_left,  y_b_left,  "other")
    loss_bc += boundary_conditions_loss(model, x_b_right, y_b_right, "other")
    loss_bc += boundary_conditions_loss(model, x_b_bottom, y_b_bottom, "other")
    loss_bc += boundary_conditions_loss(model, x_b_top,   y_b_top,   "top")
    
    # total loss
    loss = pde_loss + loss_bc
    loss.backward()
    optimizer.step()
    
    # print info
    if epoch % print_interval == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.6f}, PDE_Loss={pde_loss.item():.6f}, BC_Loss={loss_bc.item():.6f}")
    
    # Write VTK at chosen epoch intervals
    if epoch % nOutputEpoch == 0:
        with torch.no_grad():
            u_pred, v_pred, p_pred = model(X_torch, Y_torch)
        u_grid = u_pred.cpu().numpy().reshape(ny, nx)
        v_grid = v_pred.cpu().numpy().reshape(ny, nx)
        p_grid = p_pred.cpu().numpy().reshape(ny, nx)
        
        filename = f"cavity_dgm_{epoch:04d}.vtk"
        write_vtk_structured_dgm(filename, X, Y, u_grid, v_grid, p_grid)
        print(f"Saved VTK at epoch {epoch}: {filename}")

print("Training finished.")

import torch
import torch.nn as nn
import numpy as np
import time

N_f = 30000  
N_bc = 1000
epochs = 8000

R_wire = 0.1
R_ext = 0.5
J = 1.0

#масштабирование элементов 
U_0 = 0.01            
J_scaled = J / U_0    

A_in = np.pi * R_wire**2
A_out = np.pi * (R_ext**2 - R_wire**2)

def u_analytic(x, y):
    r = torch.sqrt(x**2 + y**2)

    sol_outside = (J * R_wire**2 / 2) * torch.log(R_ext / r)

    sol_inside = (J / 4) * (R_wire**2 - r**2) + (J * R_wire**2 / 2) * torch.log(torch.tensor(R_ext / R_wire))

    return torch.where(r <= R_wire, sol_inside, sol_outside)

torch.manual_seed(57)
np.random.seed(57)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Выбрано: {device}")

history = {
    'total': [],
    'integral': [],
    'bc': []
}

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_Tanh_stack = nn.Sequential(
            nn.Linear(2, 64), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, 64), 
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x, y):
        x_y = torch.cat([x, y], dim=1)
        return self.linear_Tanh_stack(x_y)  

def get_points(N_f, N_bc):
    N_in = N_f // 2
    N_out = N_f - N_in

    #внутри
    r_in = R_wire * torch.sqrt(torch.rand(N_in, 1, device=device))
    theta_in = 2 * np.pi * torch.rand(N_in, 1, device=device)

    x_in = (r_in * torch.cos(theta_in)).requires_grad_(True)
    y_in = (r_in * torch.sin(theta_in)).requires_grad_(True)

    #снаружи
    r_out = torch.sqrt(R_wire**2 + (R_ext**2 - R_wire**2) * torch.rand(N_out, 1, device=device))
    theta_out = 2 * np.pi * torch.rand(N_out, 1, device=device)

    x_out = (r_out * torch.cos(theta_out)).requires_grad_(True)
    y_out = (r_out * torch.sin(theta_out)).requires_grad_(True)

    #на границе
    theta_bc = 2 * np.pi * torch.rand(N_bc, 1, device=device)

    x_bc = R_ext * torch.cos(theta_bc)
    y_bc = R_ext * torch.sin(theta_bc)
    
    return x_in, y_in, x_out, y_out, x_bc, y_bc

def calculate_error(model, n_test=100):

    x = torch.linspace(-R_ext, R_ext, n_test, device=device)
    y = torch.linspace(-R_ext, R_ext, n_test, device=device)

    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    r_flat = torch.sqrt(x_flat**2 + y_flat**2)

    mask = r_flat <= R_ext

    x_test = x_flat[mask].reshape(-1, 1)
    y_test = y_flat[mask].reshape(-1, 1)

    u_pred = model(x_test, y_test) * U_0
    u_true = u_analytic(x_test, y_test)

    u_max = torch.max(torch.abs(u_true))

    return (torch.mean(torch.abs(u_pred - u_true)) / u_max).item()

def get_loss(x_in, y_in, x_out, y_out, x_bc, y_bc):

    #внутри
    u_in = model(x_in, y_in)

    u_x_in = torch.autograd.grad(u_in, x_in, grad_outputs=torch.ones_like(u_in), create_graph=True)[0]
    u_y_in = torch.autograd.grad(u_in, y_in, grad_outputs=torch.ones_like(u_in), create_graph=True)[0]

    energy_in = 0.5 * (u_x_in**2 + u_y_in**2) - J_scaled * u_in

    loss_in = torch.mean(energy_in) * A_in

    #снаружи
    u_out = model(x_out, y_out)
    
    u_x_out = torch.autograd.grad(u_out, x_out, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
    u_y_out = torch.autograd.grad(u_out, y_out, grad_outputs=torch.ones_like(u_out), create_graph=True)[0]
    
    energy_out = 0.5 * (u_x_out**2 + u_y_out**2)

    loss_out = torch.mean(energy_out) * A_out

    loss_integral = loss_in + loss_out

    #на границе
    u_bc_pred = model(x_bc, y_bc)
    loss_bc = torch.mean(u_bc_pred**2)

    total_loss = loss_integral + 300 * loss_bc

    return total_loss, loss_integral, loss_bc

start = time.time()

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.1)

for epoch in range(epochs):
    x_in, y_in, x_out, y_out, x_bc, y_bc = get_points(N_f, N_bc)

    optimizer.zero_grad()

    loss, loss_integral, loss_bc = get_loss(x_in, y_in, x_out, y_out, x_bc, y_bc)
    loss.backward()

    optimizer.step()
    scheduler.step()

    history['total'].append(loss.item())
    history['integral'].append(loss_integral.item())
    history['bc'].append(loss_bc.item())

    if epoch % 1000 == 0:
        error = calculate_error(model)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d} | LR: {current_lr:.1e} | Loss: {loss.item():.4f} | Loss_integral: {loss_integral.item():.4f} | | Loss_bc: {loss_bc.item():.4f} | Error: {error:.4f}")

end = time.time()
print(f"\nTime {end - start:.4f} сек")
print(f"Final error (PINN): {calculate_error(model):.4e}")

torch.save(history, "train_history.pt")
torch.save(model.state_dict(), "pinn_poisson.pth")
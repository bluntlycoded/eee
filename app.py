import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def system_dynamics(t, y, params):
    """
    Defines the system of differential equations for the DI ecosystem.
    
    Args:
        t (float): Current time.
        y (list): Current state vector [x_I, x_H].
        params (dict): Dictionary of system parameters.

    Returns:
        list: Derivatives [dx_I/dt, dx_H/dt].
    """
    x_I, x_H = y
    
    # Unpack parameters
    a_i = params['a_i']
    b_ik = params['b_ik']
    a_h = params['a_h']
    b_hi = params['b_hi']
    K_p = params['K_p']
    x_I_desired = params['x_I_desired']
    
    # --- Disturbance (Security Breach) ---
    # A strong negative impulse at t=5
    disturbance_w_A = 0
    if 5 <= t < 5.2: # Impulse duration
        disturbance_w_A = -2.0
        
    # --- Controller ---
    # The governance policy is active only if K_p > 0
    error = x_I_desired - x_I
    control_u = K_p * error
    
    # --- Differential Equations ---
    # Issuer Trust Dynamics: Decays naturally, reinforced by active holders,
    # impacted by disturbances and corrected by the controller.
    dx_I_dt = -a_i * x_I + b_ik * x_H + disturbance_w_A + control_u
    
    # Holder Adoption Dynamics: Decays naturally, reinforced by high issuer trust.
    dx_H_dt = -a_h * x_H + b_hi * x_I * x_H  # Adoption depends on trust
    
    return [dx_I_dt, dx_H_dt]

def run_and_plot_simulation():
    """
    Runs the simulation for both controlled and uncontrolled scenarios
    and generates the plots as seen in the paper.
    """
    # --- Shared Simulation Parameters ---
    t_span = [0, 50]
    t_eval = np.linspace(t_span[0], t_span[1], 500)
    initial_conditions = [1.0, 1.0] # [x_I(0), x_H(0)]
    
    # Parameters from Appendix A, with plausible values for holder dynamics
    base_params = {
        'a_i': 0.1,         # Issuer trust decay rate
        'b_ik': 0.1,        # Trust gain from holder activity
        'a_h': 0.5,         # Holder adoption decay rate (churn)
        'b_hi': 0.5,        # Adoption gain from issuer trust
        'x_I_desired': 1.0, # Desired equilibrium for trust
    }
    
    # --- Run Uncontrolled Scenario ---
    params_uncontrolled = base_params.copy()
    params_uncontrolled['K_p'] = 0.0 # No controller
    
    sol_uncontrolled = solve_ivp(
        fun=system_dynamics, 
        t_span=t_span, 
        y0=initial_conditions, 
        t_eval=t_eval, 
        args=(params_uncontrolled,)
    )

    # --- Run Controlled Scenario ---
    params_controlled = base_params.copy()
    params_controlled['K_p'] = 0.75 # Controller is active
    
    sol_controlled = solve_ivp(
        fun=system_dynamics, 
        t_span=t_span, 
        y0=initial_conditions, 
        t_eval=t_eval, 
        args=(params_controlled,)
    )

    # --- Plotting: Figure 3 (System Response) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle('Ecosystem Response to Security Breach at t=5', fontsize=16, fontweight='bold')

    # Top Subplot: Issuer Trust Score
    ax1.plot(sol_uncontrolled.t, sol_uncontrolled.y[0], 'r-', label='Uncontrolled System')
    ax1.plot(sol_controlled.t, sol_controlled.y[0], 'b-', label='Controlled System')
    ax1.axhline(y=1.0, color='k', linestyle='--', label='Equilibrium')
    ax1.set_ylabel('Issuer Trust Score $x_I(t)$', fontsize=12)
    ax1.legend()
    ax1.set_ylim(-0.2, 1.4)
    ax1.set_title('Issuer Trust Dynamics', fontsize=14)

    # Bottom Subplot: Holder Adoption Level
    ax2.plot(sol_uncontrolled.t, sol_uncontrolled.y[1], 'r-', label='Uncontrolled System')
    ax2.plot(sol_controlled.t, sol_controlled.y[1], 'b-', label='Controlled System')
    ax2.axhline(y=1.0, color='k', linestyle='--', label='Equilibrium')
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Holder Adoption Level $x_H(t)$', fontsize=12)
    ax2.set_ylim(-0.2, 1.4)
    ax2.set_title('Holder Adoption Dynamics', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('simulation_response.png', dpi=300)
    print("Saved plot as 'simulation_response.png'")

    # --- Plotting: Figure 4 (Conceptual Lyapunov Function) ---
    fig2, ax_lyap = plt.subplots(figsize=(10, 6))
    t_lyap = np.linspace(5, 50, 200)
    V_lyap = np.exp(-0.2 * (t_lyap - 5))
    
    ax_lyap.plot(t_lyap, V_lyap, color='green', lw=2, label='$V(t)$ for Controlled System')
    ax_lyap.plot([0, 5], [0, 0], 'k-') # V(t) = 0 before disturbance
    ax_lyap.set_xlabel('Time (t)', fontsize=12)
    ax_lyap.set_ylabel('Lyapunov Function Value $V(t)$', fontsize=12)
    ax_lyap.set_title('Conceptual Lyapunov Function Convergence', fontsize=16, fontweight='bold')
    ax_lyap.set_xlim(0, 50)
    ax_lyap.set_ylim(0, 1.1)
    ax_lyap.legend()
    
    plt.tight_layout()
    plt.savefig('lyapunov_function_plot.png', dpi=300)
    print("Saved plot as 'lyapunov_function_plot.png'")
    
    plt.show()


if __name__ == '__main__':
    run_and_plot_simulation()

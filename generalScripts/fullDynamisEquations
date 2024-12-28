import numpy as np

def sim_equations(t, full_state, param):
    """
    Compute the state derivatives for both target and chaser spacecraft
    and their relative dynamics in a CR3BP environment.
    
    Parameters:
        t (float):				Current time [s]
        full_state (np.array):  Full state vector [target state, chaser state, relative state].
        param (Param): 			Instance of the Param dataclass, containing all necessary parameters.

    Returns:
        dSimState (np.array): 	Derivatives of the full state.
    """
    
    # Initialisation of the parameters
    target_state = full_state[1:6]
    chaser_state = full_state[6:12]
    relative_state = full_state[12:18]

    # Environment
    sun_versor = sunPositionVersor(t, param)

    # Spacecraft Guidance Algorithm (for relative dynamics)
    dRelState = relativeDynamicsLVLH(t, relative_state, chaser_state, param, target_state)
    # Uncomment for guidance:
    # if param['guidance_bool']:
    #     # Compute inverse dynamics
    #     uID = inverse_dynamics(t, lambda t, target_state, chaser_state: relative_dynamics_model(t, target_state, chaser_state, param), full_state, param)
    # 
    #     # Augment with APF
    #     uAPF = 0
    # 
    #     # Resolve the guidance problem merging with SMC
    #     uTOT = uID + uAPF

    # Propagation of CR3BP dynamics
    dstate_Target = CR3BP(t, target_state, param)

    # If guidance is active, propagate chaser dynamics with guidance
    # if param['guidance_bool']:
    #     dstate_chaser = CR3BP(t, chaser_state, param, uTOT)
    # else:
    dstate_Chaser = CR3BP(t, chaser_state, param)

    # Full output generation
    dSimState = np.concatenate([dstate_Target.flatten(), dstate_Chaser.flatten(), dRelState.flatten()])

    return dSimState

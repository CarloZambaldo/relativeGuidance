import gymnasium as gym

class noAutoResetWrapper(gym.Wrapper):
    """
    This wrapper is used to prevent automatic reset using DummyVecEnv

    """
    def __init__(self, env):
        super().__init__(env)
        self.needs_reset = False  # Flag per controllare se l'ambiente necessita di un reset manuale

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            # avoid automatic reset by changin the terminated and truncated
            self.needs_reset = True
            terminated = False  # Evita che DummyVecEnv resetti automaticamente
            truncated = False

        info["terminated"] = terminated
        info["truncated"] = truncated

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        The reset is done only if it is marked to be reset
        
        """
        if self.needs_reset:
            self.needs_reset = False  # Ripristina il flag
            return self.env.reset(**kwargs)
        else:
            return self.env.reset(**kwargs)
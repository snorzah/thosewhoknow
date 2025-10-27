from HetMRFWScenario import HetMRFWScenario
from use_vmas_env import use_vmas_env
use_vmas_env(
    render=True,
    num_envs=8,
    n_steps=100,
    device="cpu",
    scenario=HetMRFWScenario(),
    continuous_actions=True
)
# render_interactively(HetMRFWScenario(),
#                      control_two_agents=False)
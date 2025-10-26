from vmas import make_env, render_interactively
from HetMRFWScenario import HetMRFWScenario
env = make_env(
    scenario=HetMRFWScenario(),
    num_envs=32,
    device='cpu',
    seed=1337,
)
actions = env.get_random_actions()
obs, rews, dones, info = env.step(actions)
print(HetMRFWScenario.__name__)

from vmas.examples.use_vmas_env import use_vmas_env
use_vmas_env(
    render=True,
    save_render=True,
    num_envs=32,
    n_steps=100,
    device='cpu',
    scenario_name=HetMRFWScenario.__name__,
    continuous_actions=True,
    random_action=True
)

# render_interactively(HetMRFWScenario(),
#                      control_two_agents=False)
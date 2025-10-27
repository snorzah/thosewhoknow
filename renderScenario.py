from HetMRFWScenario import HetMRFWScenario
from use_vmas_env import use_vmas_env
from vmas import make_env

env = make_env(
    scenario=HetMRFWScenario(),
    num_envs=1,
    device="cpu",
)
actions = env.get_random_actions()
obs,rews, dones, info = env.step(actions)

for obs_per_agent in obs:
    print(obs_per_agent)
print(f"observation dimension holonomic: {len(obs[0])}")
print(f"observation dimension kinematicBicycle: {len(obs[-1])}")

# use_vmas_env(
#     render=True,
#     num_envs=8,
#     n_steps=100,
#     device="cpu",
#     scenario=HetMRFWScenario(),
#     continuous_actions=True
# )
# render_interactively(HetMRFWScenario(),
#                      control_two_agents=False)
from gym.envs.registration import register

register(
    id='bs_env-v0',
    entry_point='gym_bs_env.envs:BSEnv',
)
#register(
#    id='bs_env-extrahard-v0',
#    entry_point='gym_foo.envs:BS_extra_hardEnv',
#)
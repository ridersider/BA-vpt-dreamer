import minedojo
env = minedojo.make('open-ended', image_size=(128, 128))
obs = env.reset()
print('MineDojo OK — obs keys:', list(obs.keys()))
print('RGB shape:', obs['rgb'].shape)
env.close()
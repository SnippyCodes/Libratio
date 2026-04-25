import matplotlib.pyplot as plt

steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
losses = [0.000000, 0.000000, 0.000002, 0.000007, 0.000016, 0.000021, 0.000034, 0.000040, 0.000049, 0.000057]
rewards = [0.185827, 0.351165, 0.489893, 0.609637, 0.611199, 0.757431, 0.778288, 0.742079, 0.846850, 0.824075]

plt.figure(figsize=(10, 5))
plt.plot(steps, rewards, marker='o', color='g', label='Mean Reward')
plt.title('GRPO Training Run 2: Mean Fleet Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('reward_curve_run2.png')
print("Saved reward_curve_run2.png")

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, marker='o', color='r', label='Training Loss')
plt.title('GRPO Training Run 2: Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_curve_run2.png')
print("Saved loss_curve_run2.png")

import numpy as np
import matplotlib.pyplot as plt


def vote(players, x, y):
    stencil = (np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]) + (x, y) + players.shape[:2]) % players.shape[:2]
    group = players[tuple(stencil.T)]
    players[tuple(stencil.T)] = np.where(np.sum(group, axis=0) > 2, 1, 0)


def play(L, Nreps, P):
    players = (np.random.uniform(size=(L, L, Nreps)) < P).astype(int)
    means = [np.mean(players, axis=(0, 1))]
    while np.any(np.std(players, axis=(0, 1))):
        vote(players, *(np.random.randint(n) for n in players.shape[:2]))
        means.append(np.mean(players, axis=(0, 1)))
    disc_rounds = np.array([np.min(np.where(np.round(m) == m)) for m in np.transpose(means)])
    print(f"Avg. time to agree: {np.mean(disc_rounds):.1f} +/- {np.std(disc_rounds) / (np.sqrt(np.mean(disc_rounds) - 1)):.2} h")
    return disc_rounds, np.array(means)


L, Nreps, P = 16, 1024, 0.1
disc_rounds, means = play(L=L, Nreps=Nreps, P=P)

# Distribution of discussion times has quite long tail to the right:
plt.figure()
plt.plot(means)
plt.xlabel("time [h]")
plt.ylabel("fraction of flat-earthers")
plt.figure()
plt.hist(disc_rounds, bins=20)
plt.xlabel("discussion time")

# Average discussion time grows algebraically with number of participants:
ls = np.arange(4, 24)
rounds_vs_L = [play(ll, Nreps, P)[0].mean() for ll in ls]
plt.figure()
plt.plot(ls, rounds_vs_L, "o")
plt.xlabel("L")
plt.ylabel("discussion time")
plt.xscale("log")
plt.yscale("log")

# Mean estimator is stable, but standard error of the mean does not seem to converge with Nreps
ns = np.logspace(5, 12, 16, base=2).astype(int)
rounds_vs_Nreps = [play(L, n, P)[0].mean() for n in ns]
plt.figure()
plt.plot(ns, rounds_vs_Nreps, "o")
plt.xlabel("# repetitions")
plt.ylabel("discussion time")
plt.xscale("log")
plt.show()

# Discussion time is sharply peaked around 50:50 split initial condition:
ps = np.linspace(1 / L**2, 0.5, 20)
rounds_vs_P = [play(L, Nreps, p)[0].mean() for p in ps]
plt.figure()
plt.plot(ps, rounds_vs_P, "o")
plt.xlabel("initial fraction")
plt.ylabel("discussion time")
plt.yscale("log")

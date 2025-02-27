import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.stats as stats
import random
import math
# import seaborn as sb

def simulate_unfair_dice(num_rolls=1000, probabilities=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
    # Example probability distribution for an unfair die
    # Generate random rolls based on the specified probabilities
    rolls = np.bincount(np.random.choice(range(6), size=num_rolls, p=probabilities), minlength=6)
    # normalize the rolls
    rolls = rolls / num_rolls
    return rolls

def dice_error(dice, probabilities=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]):
    # Calculate the error between the actual probabilities and the observed probabilities
    return np.sum(np.abs(dice - probabilities))


def sample_simplex(n, total):
    """
    Sample n nonnegative numbers that sum to 'total' uniformly.
    (This is the standard “stick-breaking” method.)
    """
    if n == 1:
        return [total]
    cuts = sorted(random.uniform(0, total) for _ in range(n - 1))
    parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [total - cuts[-1]]
    return parts

def sample_bounded_simplex(n, total, bound, tol=1e-9):
    """
    Sample n nonnegative numbers that sum to 'total' with each part <= bound.
    If total is nearly equal to n*bound (within tolerance), then the only possibility is each equals bound.
    Otherwise we use rejection sampling.
    """
    if abs(total - n * bound) < tol:
        return [bound] * n
    while True:
        parts = sample_simplex(n, total)
        if all(x <= bound + tol for x in parts):
            return parts

def biased_dice(error, tol=1e-9):
    """
    Generate a random list of 6 probabilities that sum to 1 and have total error
    (sum of |p - 1/6|) equal to 'error'. The allowed error is between 0 and 5/3.
    """
    if error < 0 or error > 5/3:
        raise ValueError("error must be between 0 and 5/3 (inclusive).")
    # If error is 0, return the uniform distribution.
    if abs(error) < tol:
        return [1/6] * 6

    E_pos = error / 2.0  # Total amount to add and subtract

    # For any face, a positive deviation cannot exceed 5/6 (so that p <= 1)
    # and a negative deviation (in absolute value) cannot exceed 1/6 (so that p >= 0).
    # Thus, if k faces are positive, we need E_pos <= k*(5/6)  => k >= ceil(3E/5),
    # and if m faces are negative, we need E_pos <= m*(1/6)   => m >= ceil(3E).
    k_min = math.ceil((3 * error) / 5)  # minimum number of positive faces
    m_min = math.ceil(3 * error)          # minimum number of negative faces

    # Find all valid pairs (k, m) with k, m >= 1 and k + m <= 6.
    valid_pairs = []
    for k in range(k_min, 7):
        for m in range(m_min, 7 - k + 1):  # ensure k+m <= 6
            if k + m <= 6 and k >= 1 and m >= 1:
                valid_pairs.append((k, m))
    if not valid_pairs:
        # Fallback: force all faces to deviate (k+m = 6)
        k = math.ceil((3 * error) / 5)
        m = 6 - k
    else:
        k, m = random.choice(valid_pairs)

    # Randomly choose which indices will have nonzero deviation.
    indices = list(range(6))
    nonzero_count = k + m
    nonzero_indices = random.sample(indices, nonzero_count)
    pos_indices = random.sample(nonzero_indices, k)
    neg_indices = [i for i in nonzero_indices if i not in pos_indices]

    # For the k positive deviations, each must be in [0, 5/6].
    pos_bound = 5/6
    if abs(E_pos - k * pos_bound) < tol:
        pos_deviations = [pos_bound] * k
    else:
        pos_deviations = sample_bounded_simplex(k, E_pos, pos_bound, tol)

    # For the m negative deviations, each (as a positive number) must be in [0, 1/6].
    neg_bound = 1/6
    if abs(E_pos - m * neg_bound) < tol:
        neg_deviations = [neg_bound] * m
    else:
        neg_deviations = sample_bounded_simplex(m, E_pos, neg_bound, tol)

    # Build the deviation vector (delta = p - 1/6).
    deltas = [0.0] * 6
    for i, d in zip(pos_indices, pos_deviations):
        deltas[i] = d      # positive deviation
    for i, d in zip(neg_indices, neg_deviations):
        deltas[i] = -d     # negative deviation

    # Now the probabilities:
    probs = [1/6 + d for d in deltas]
    # (In exact arithmetic the p's sum to 1; here we normalize to be safe.)
    total = sum(probs)
    probs = [p / total for p in probs]

    return probs


def uniform_dice(min_error=0.01, max_error=0.1):
    return biased_dice(random.uniform(min_error, max_error))

def dirichlet_dice(alpha=1.0):
    return np.random.dirichlet([alpha] * 6)

num_fakedice = 10000
alpha = 10**2
error_range = (0, 30)
fair_dice = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
unfair_dice = [[1, 0, 0, 0, 0, 0] for _ in range(num_fakedice)]
bins = 20
# print(dice_error(np.array([2/6, 0/6, 1/6, 1/6, 1/6, 1/6]))) # 0.3333333333333333

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # leave space for slider

# hist,a,b = plt.hist(stddevs, bins=50)

rolls_slider = plt.axes((0.25, 0.07, 0.65, 0.03))
rolls_slider = Slider(rolls_slider, "Rolls", 1, 10000, valinit=2000)
min_bad_slider = plt.axes((0.25, 0.12, 0.65, 0.03))
max_bad_slider = plt.axes((0.25, 0.17, 0.65, 0.03))
min_bad_slider = Slider(min_bad_slider, "Min Bad", 0, 1.66, valinit= 0.1)
max_bad_slider = Slider(max_bad_slider, "Max Bad", 0, 1.66, valinit= 0.1)
scale_slider = plt.axes((0.25, 0.02, 0.65, 0.03))
scale_slider = Slider(scale_slider, "Scale", 10, 200, valinit=20)
# alpha_slider = plt.axes((0.25, 0.15, 0.65, 0.03))
# alpha_slider = Slider(alpha_slider, "Alpha", 0.1, 10, valinit=2)

def update(val):
    ax.clear()

    # percent_error = [dice_error(d)*100 for d in dice]
    # chiresults = [stats.chisquare([0, 0, 8, 7, 4, 5], f_exp=simulate_unfair_dice(num_rolls=24, probabilities=d)) for d in dice]
    # pvalue = [c.pvalue for c in chiresults]
    tests = [simulate_unfair_dice(num_rolls=int(rolls_slider.val)) for _ in range(num_fakedice)]
    errors = [dice_error(t)*100 for t in tests]
    density = stats.kde.gaussian_kde(errors, bw_method=0.2)
    x = np.arange(0., scale_slider.val, scale_slider.val/50)
    ax.plot(x, density(x))

    fake_tests = [simulate_unfair_dice(num_rolls=int(rolls_slider.val), probabilities=d) for d in unfair_dice]
    fake_errors = [dice_error(t)*100 for t in fake_tests]
    density = stats.kde.gaussian_kde(fake_errors, bw_method=0.2)
    ax.plot(x, density(x))




    # ax.xlabel('Error (0%-167%)')
    # ax.ylabel('Frequency')

    fig.canvas.draw_idle()  # refresh plot

def update_alpha(val):
    global unfair_dice
    unfair_dice = [uniform_dice(min_bad_slider.val, max_bad_slider.val) for _ in range(num_fakedice)]
    # unfair_dice = [[1, 0, 0, 0, 0, 0] for _ in range(num_fakedice)]
    update(1)


# alpha_slider.on_changed(update_alpha)  # connect slider to update function
min_bad_slider.on_changed(update_alpha)
max_bad_slider.on_changed(update_alpha)
rolls_slider.on_changed(update_alpha)  # connect slider to update function
scale_slider.on_changed(update_alpha)  # connect slider to update function
update_alpha(2)
update(100)

plt.show()

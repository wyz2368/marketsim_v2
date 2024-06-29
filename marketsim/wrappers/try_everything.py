import torch
import torch.distributions as dist
from collections import defaultdict

from gymnasium.spaces import Box
import numpy as np

# print(min(-1, 0.5))

# a = np.array([1,2,3,4,5])
# b = np.array([2,3,4,5,6])
#
# print(b)


def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()

agents = {}
arrival_index = 0
arrivals = defaultdict(list)
arrival_times = sample_arrivals(5e-3, int(100))
print(arrival_times)
for agent_id in range(25):
    arrivals[arrival_times[arrival_index].item()].append(agent_id)
    arrival_index += 1

first_100 = arrival_times[:100]
print("100:", sorted(first_100))

print(arrivals)

# """
# tensor([ 8., 31.,  2.,  8.,  4., 11., 30.,  2., 18.,  8.,  4.,  7., 25.,  8.,
#          1.,  0., 11.,  1.,  0.,  3., 18.,  7.,  7.,  3., 15.,  5., 12.,  9.,
#          1., 38.,  1.,  7., 32., 13.,  8., 17.,  0.,  0.,  9.,  1.,  9., 18.,
#          8.,  1.,  9.,  0.,  4.,  8.,  2.,  7.,  0.,  8.,  9.,  3.,  5., 12.,
#          7.,  3.,  0.,  3.,  8., 31., 20.,  2.,  0., 13., 21.,  7.,  8., 10.,
#          8.,  9., 14.,  1., 20., 34., 48.,  3.,  5.,  2.,  7., 10., 12.,  0.,
#          8.,  8.,  3.,  3.,  4.,  4., 15., 37.,  9., 33.,  2., 18., 11.,  3.,
#          0.,  9.])
# defaultdict(<class 'list'>, {8.0: [0, 3, 9], 31.0: [1], 2.0: [2, 7], 4.0: [4], 11.0: [5], 30.0: [6], 18.0: [8]})
# """
#
# print("----------------------")
# #
# arrival_index_MM = 0
# arrivals_MM = defaultdict(list)
# arrival_times_MM = sample_arrivals(0.2, 100)
# print(arrival_times_MM)
# arrivals_MM[arrival_times_MM[arrival_index_MM].item()].append(11)
# arrival_index += 1
#
# print(arrivals_MM)

# a = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
# b = a.sample()
# print(b)
# print(type(b))


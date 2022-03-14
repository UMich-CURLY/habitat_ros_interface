import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
import scipy.stats as ss
import numpy as np
place_num = 20
np.random.seed(100)
popularity_order = np.arange(0,20)
np.random.shuffle(popularity_order)
# popularity_order = list(range(0,place_num))
# x = np.arange(0,place_num)
x = np.arange(0,20)
xU, xL = x + 0.5, x - 0.5
std_dev = 5
prob = ss.norm.cdf(xU, loc = 15, scale = std_dev) - ss.norm.cdf(xL, loc = 15, scale = std_dev)
prob = prob / prob.sum()
prob_res = []
for i in range(0,20):
    prob_res.append(prob[popularity_order[i]])


nums = np.random.choice(x, size = 10000, p = prob_res)

fig, ax = plt.subplots()

dict = {i:prob_res[i] for i in range(20)}
dict_value = list(dict.values())
dict_key = list(dict.keys())

dict_key = sorted(dict_key, key=lambda x: dict[x], reverse=True)
x_label = [7, 15, 19, 0, 2, 3, 13, 9, 10, 17, 1, 8, 6, 16, 11, 12, 5, 18, 14, 4]
hmm = [4,5,11,6,1,10,9,2,0,7,15,19,3,13,17,8,16,12,18,14]
popularity_order = dict_key
prob_res_1 = []
for i in range(0,20):
    prob_res_1.append(prob_res[popularity_order[i]])

plt.figure(figsize=(14,10))
nums = np.random.choice(x, size = 10000, p = prob_res_1)
plt.hist(nums, bins = len(x)+1,range = (-0.5,20.5), density = True)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'ultralight',
        'size': 40,
        }
# plt.rc('text', usetex=True)
# plt.rcParams["font.family"] = ["Modern Roman"]
plt.xticks(range(0,20),labels=x_label)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel("Node numbers",fontdict=font)
plt.ylabel("Probability",fontdict=font)
# plt.ylabel("Probability")
plt.savefig("./temp/place_num.eps")
plt.savefig('./temp/place_num.png')
plt.show()
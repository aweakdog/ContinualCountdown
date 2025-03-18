# ContinualCountdown
This is a repo for the experiments on Countdown task with continual learning setting.

1.group by the number of Available Numbers [2,3,4] with the sample size of 100000 each group

2.group the magnitude of the max target number [333, 666, 999] with the sample size of 100000 each group

3.group the set of the used operators [{+}, {+ -}, {+ - *}, {+ - * /}] with the sample size of 100000 each group

to record the following metrics of each group durning training:
a. success rate
b. normalized weight change
c. loss function
d. normalized gradient
e. activations
f. response length
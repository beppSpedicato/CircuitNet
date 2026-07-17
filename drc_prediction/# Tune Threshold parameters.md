# Tune Threshold parameters
## 1 Introduction
In the drc violation prediction pipeline the threshold parameters is used in the test phase to calculate the presence of a violation in a certain pixel and it creates the needed value to compute the relative confusion matrix. It's baseline is 0.1, and since it is very low, it penalizes zones in which the violation is less relevant. We retryed the test pipeline using different threshold values while evaluating the BiasedLoss 15k step model.

## 2 Experiment
The values under test were the following: 0.1, 0.25, 0.50, 0.75.
The results are those:
|  # | Tag                      | Avg NRMS | Avg SSIM |  AUC ROC | TPR @ FPR=5% | accuracy |
| -: | ------------------------ | -------: | -------: | -------: | -----------: | -------: |
|  0 | tune th biased 75%       | 0.217567 | 0.784107 | 0.942380 |     0.770293 | 0.963925 |
|  1 | tune th biased 50%       | 0.217567 | 0.784107 | 0.933020 |     0.780682 | 0.964542 |
|  2 | tune th biased 25%       | 0.217567 | 0.784107 | 0.932823 |     0.800284 | 0.954057 |
|  5 | Baseline Biased 10%      | 0.217567 | 0.784107 | 0.906719 |     0.754848 | 0.929905 |

The AUC under ROC curve increases with threshold values and also accuracy does, though the value of (TPR; 5% FPR) approaches the peak using a threshold of 25%. 

## 3 Conclusion
In the original paper (CircuitNet 1.0), the threshold is set to 10%, which is reasonable due to the fact that we can spot more region, which is preferred to prevent FN. To se this, a second step is to plot the confusion matrix in order to see the overall prediction categorization
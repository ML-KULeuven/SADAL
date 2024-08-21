# SADAL : Semi-supervised Anomaly Detector combining Active learning and Learning to reject

`SADAL` (Semi-supervised Anomaly Detector combining Active learning and Learning to reject) is a GitHub repository containing the **SADAL** algorithm.
It refers to the paper titled *Combining Active Learning and Learning to Reject for Anomaly Detection* published at ECAI 2024.



## Abstract

Anomaly detection attempts to identify instances in the data that do not conform to the expected behavior. Because it is often difficult to label instances, the problem is tackled in an unsupervised way by employing data-driven heuristics to identify anomalies. However, the heuristics are imperfect which can degrade a detector's performance. One way to mitigate this problem is using Active Learning to collect labels that help correct cases where the employed heuristics are incorrect. Alternatively, one can allow the detector to abstain (i.e., say 'I do not know') whenever it is likely to make mispredictions at test time, which is called Learning to Reject (LtR). However, while both have been studied in the context of anomaly detection, they have not been considered in conjunction. Although they both need labels to accomplish their task, integrating these two ideas is challenging for two reasons. First, their label selection strategies are intertwined but they acquire different types of labels. Second, it is unclear how to best divide the limited budget between labeling instances that help AL and those that help LtR. In this paper, we introduce SADAL, the first semi-supervised detector that allocates the label budget between AL and LtR by relying on a reward-based selection function. Experimentally on 25 datasets, we show that our approach outperforms several baselines by achieving a better performance.

## Contents and usage

The folder contains:
- SADAL.py, the file containing the SADAL algorithm;
- SSIF.py, the base semi-supervised detector used by the SADAL algorithm;
- Notebook.ipynb, a notebook showing how to use SADAL on an artificial 2D dataset;
- Supplement.pdf, a pdf with the supplementary material used for the paper.

To use SADAL, import the github repository or simply download the files. You can also find the benchmark datasets at this [link](https://github.com/Minqi824/ADBench)[1]. 


## Budget allocation for Active Learning and Learning to reject in Anomaly Detection (BALLAD)

Given a dataset **X**, we use SSIF [[paper](https://epubs.siam.org/doi/epdf/10.1137/1.9781611978032.77)][2] as a semi-supervised anomaly detector to learn from partially labeled examples. We simulate k allocation rounds. We initialize the problem by (1) training SSIF with no labels and setting a default rejection threshold to the contamination factor (i.e. the percentage of anomalies in the dataset), and (2) collecting random labels for LtR and for AL in the first two allocation rounds. This allows us to compute the initial rewards by measuring how the detector varies from (1) to (2): for LtR, we measure the variation after re-setting the rejection thresholds; for AL, we measure the variation after re-training the detector with the new labels. Then, we start the allocation loop. In each round, we allocate the budget to the option (LtR or AL) with the highest reward, and we update the reward using the new labels. We consider the **entropy reward** as reward function, which looks at the detector’s probabilities, either for prediction (AL), or for rejection (LtR).

Given a dataset **X** with true labels **y**, we first split it into training/test using the proportion 80/20 (the SADAL class splits the training set into training and validation using the proportion 50/50). Then, our algorithm is applied as follows:

```python
from sadal import SADAL
model = SADAL(Xtrain, ytrain, contamination)
test_costs, fnr, fpr, rejection_rate = model.run_allocation_loop(nrounds=10, budget=int(len(Xtrain)*0.2), Xtest, ytest)
```

## Dependencies

The `SADAL` class requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Sklearn](https://scikit-learn.org/stable/)
- [Skopt](https://scikit-optimize.github.io/stable/)
- [Matplotlib](https://matplotlib.org)


## Contact

Contact the author of the paper: [luca.stradiotti@kuleuven.be](mailto:luca.stradiotti@kuleuven.be).


## References

[1] Han, Songqiao, et al. "Adbench: Anomaly detection benchmark." Advances in Neural Information Processing Systems 35 (2022): 32142-32159.

[2] Stradiotti, Luca, Lorenzo Perini, and Jesse Davis. "Semi-Supervised Isolation Forest for Anomaly Detection." Proceedings of the 2024 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2024.

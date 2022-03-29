# ML-Assign3
It takes huge time to prepare the data from UN-english.txt.gz. Attempts were made both at MLTGPU server and at my local laptop.
It took 1 day of running and did not finish. It was the same for training either categoricalNB or SVC model. The processing time
for the UN-english-sample.txt was still a lot. Instead, UN-english-sample-small.txt is taken for the training and testing.

## CategoricalNB
After training the model, the fellowing statistics were obtained:

    Categorical NB Performance Metrics
    ==================================
    Accuracy: 	 0.17497730142661838
    Precision: 	 0.17497730142661838
    recall: 	 0.17497730142661838
    F1 score: 	 0.17497730142661838

The result is somewhat not statifactory, all the scores are about 18%. The above score is generated with paratmter as 'macro'.


## SVM-SVC
The SVC model predicted other labels apart from the true lables. It triggered the following warning:

    "UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with 
    no predicted samples. Use `zero_division` parameter to control this behavior."

 and the following scores are obtained:

          precision    recall  f1-score   support

           0       0.00      0.00      0.00        30
           1       0.00      0.00      0.00       103
           2       0.00      0.00      0.00        71
           3       0.00      0.00      0.00        83
           4       0.00      0.00      0.00        30
           5       0.00      0.00      0.00        21
           7       0.00      0.00      0.00         3
           8       0.00      0.00      0.00        90
           9       0.00      0.00      0.00        66
          10       0.19      1.00      0.32       289
          11       0.00      0.00      0.00        61
          12       0.00      0.00      0.00         2
          13       0.00      0.00      0.00       215
          14       0.00      0.00      0.00       171
          15       0.00      0.00      0.00       206
          16       0.00      0.00      0.00        23
          17       0.00      0.00      0.00        28
          19       0.00      0.00      0.00        23
          20       0.00      0.00      0.00         4

    accuracy                           0.19      1519
   macro avg       0.01      0.05      0.02      1519
weighted avg       0.04      0.19      0.06      1519




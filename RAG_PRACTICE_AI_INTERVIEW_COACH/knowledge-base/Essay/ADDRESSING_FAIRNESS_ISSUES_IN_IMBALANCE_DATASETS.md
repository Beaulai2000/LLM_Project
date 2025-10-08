### Addressing Fairness Issues in imbalance datasets

_Po-Yu Lai_

American University

#### Abstract

The presence of imbalanced datasets often leads to biased machine learning models that prioritize the majority class, undermining the representation and importance of minority class data. This report investigates techniques to tackle such bias by employing advanced Support Vector Machine (SVM) models, focusing on reducing the influence of majority class while enhancing minority class detection to ensuring robust overall performance. Through a series of experiments conducted on both synthetic and real-world datasets, the findings highlight the efficacy of these approaches in addressing class imbalance, promoting equitable prediction, and improving decision-making reliability.

**Index Terms-** Imbalanced datasets, Support Vector Machine (SVM), Minority class, Majority class

**1\. Introduction**

Support Vector Machine (SVM) is a widely used machine learning technique designed for classification tasks. It operates by identifying an optimal hyperplane that separates data points into distinct classes while maximizing the margin between them \[1\]. This separation is guided by support vectors, which are the critical data points defining the margins of the hyperplane. Over time, variations of SVM have emerged, such as Quadratic Support Vector Machine (QSVM) and Twin Support Vector Machine (TSVM), which adapt the traditional approach to handle specific challenges, including imbalanced datasets, by introducing more flexible decision boundaries \[2\].

In addition to these variations, Universum points and weighted methods offer innovative approaches to addressing fairness in classification tasks. Universum points introduce auxiliary data that neither belong to the majority nor minority classes but provide structural information, helping to refine the decision boundary \[3\].

This study investigates the effectiveness of QSVM, TSVM, Universum points in addressing fairness issues in imbalanced datasets. By evaluating their performance on synthetic and real-world data, this report provides insights into their potential to improve classification outcomes through key metrics such as accuracy rate and recall.

**2\. Preliminaries**

In this work, we utilize multiple SVM-based models, including Standard SVM, Quadratic SVM (QSVM), Universum SVM (USVM), Twin SVM (TSVM), and Universum Twin SVM (UTSVM) to address binary classification problems in the context of imbalanced datasets. To enhance computational efficiency and reduce redundancy, Principal Component Analysis (PCA) is applied to perform dimensionality reduction.

**2.1 Support Vector Machine (SVM)**

Support Vector Machine (SVM) is a powerful supervised learning technique designed for binary classification tasks. It identifies a hyperplane that maximizes the margin between two classes while minimizing classification errors. The introduction of a slack variable (𝜉) allows the model to tolerate some misclassifications, while the penalty parameter (𝐶) regulates the trade-off between a wide margin and classification accuracy. The objective function for SVM can be formulated as follows \[1\]:

Subject to:

**2.2 Quadratic Support Vector Machine (QSVM)**

Quadratic Support Vector Machine (QSVM) builds upon the traditional SVM framework by integrating a quadratic term into the objective function. This modification allows QSVM to better capture nonlinear decision boundaries, making it suitable for datasets with more complex structures. The QSVM optimization problem can be represented as follows \[2\]:

Subject to:

**2.3 Universum Support Vector Machine (USVM)**

Universum Support Vector Machine (USVM) incorporates Universum points, a set of auxiliary data that does not belong to either target class but provides valuable structural information. By leveraging Universum points, USVM enhances the generalization capability of SVM models and refines decision boundaries. The optimization problem for USVM is formulated as follows:

Subject to:

where C1​ and C2 are regularization parameters, and are slack variables for Universum points.

**2.4 Twin Support Vector Machine (TSVM)**

Twin Support Vector Machine (TSVM) modifies the original SVM by generating two non-parallel hyperplanes, each closer to its respective class while maintaining separation from the other class. TSVM solves two separate optimization problems, each corresponding to one class \[5\].

Class 1:

Subject to:

Class 2:

Subject to:

Where, C1​ and C2 are penalty terms, and represent slack variables to manage misclassifications.

**2.5 Quadratic Twin Support Vector Machine (QTSVM)**

Quadratic Twin Support Vector Machine (QTSVM) extends TSVM by incorporating quadratic terms into its objective function. This addition enables QTSVM to model nonlinear boundaries more effectively. Similar to TSVM, QTSVM solves two optimization problems \[6\]:

Class 1:

Subject to:

Class 2:

Subject to:

Where, and ​ are quadratic regularization parameters, and C1, C2 penalize misclassifications.

**2.6 Universum Twin Support Vector Machine (U-TSVM)**

Universum Twin Support Vector Machine (U-TSVM) is an extension of Twin SVM that incorporates Universum points-data that do not belong to either class but provide structural information about the feature space. The U-QTSVM optimization problem can be represented as follows \[7\]:

Class 1:

Subject to:

Class 2:

Subject to:

**2.7 Evaluation Metrics**

The performance of the proposed SVM-based methods is evaluated using the following standard metrics by \[8\]:

2.7.1 **Accuracy**: Accuracy measures the proportion of correctly classified instances relative to the total number of instances. It is calculated as:

where TP represents true positives, TN true negatives, FP false positives, and FN false negatives.

2.7.2 Recall: Recall, also known as sensitivity or true positive rate, assesses the model's ability to correctly identify all positive instances. It is defined as:

These two metrics are critical for evaluating the model's overall performance and its effectiveness in detecting the minority class in imbalanced datasets.

**3\. Experiment**

To evaluate the performance of SVM models, we use five datasets, including both synthetic and real-world data: fraud detection, income classification, age classification (senior or not), smoking classification (smoker or not), and mushroom classification (edible or not). Each dataset initially consists of a balanced set of 1000 samples for the majority class and 1000 samples for the minority class. To simulate class imbalance, we randomly select 800 majority class samples and 400 minority class samples as the training set (without replacement). The remaining 200 samples from each class are reserved for the test set. Various SVM models are trained on the imbalanced training set, and their performance is evaluated on the test set. Table 1 provides an overview of the test samples for all datasets.

**Table 1** Number of training and testing samples for all the datasets used

| ID  | Dataset3 | Minority (1) | Majority (-1) |
| --- | --- | --- | --- |
| 1   | Fraud<br><br>(Balanced) | Fraud (1000) | Not Fraud (1000) |
| 2   | Fraud<br><br>(Imbalanced) | Fraud (400) | Not Fraud (800) |
| 3   | Income<br><br>(Balanced) | Over 50k (1000) | Under 50K (1000) |
| 4   | Income<br><br>(Imbalanced) | Over 50k (400) | Under 50K (800) |
| 5   | Age<br><br>(Balanced) | Over 70 (1000) | Under 70 (1000) |
| 6   | Age<br><br>(Imbalanced) | Over 70 (400) | Under 70 (800) |
| 7   | Smoking<br><br>(Balanced) | Smoker (1000) | Not Smoker (1000) |
| 8   | Smoking<br><br>(Imbalanced) | Smoker (400) | Not Smoker (800) |
| 9   | Mushroom<br><br>(Balanced) | Edible (1000) | Not Edible (1000) |
| 10  | Mushroom<br><br>(Imbalanced) | Edible (400) | Not Edible (800) |

**3.1 Performance on balanced datasets**

We evaluate the performance of SVM, QSVM, USVM, TSVM, QTSVM, and U-TSVM on both balanced and imbalanced datasets to examine whether the incorporation of flexible decision boundaries and Universum points improves classification accuracy, enhances minority class detection, and overall model performance.

**Table 2** classification performance of Fraud balanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 85.83% | 83% | 89% |
| USVM | 82.17% | 76% | 88% |
| _TSVM_ | 18.33% | 24% | 13% |
| UTSVM | 83.17% | 77% | 89% |
| QSVM | 74.5% | 96% | 53% |
| UQSVM | 83.17% | 77% | 89% |
| QTSVM | 49.83% | 4%  | 95% |

**Table 3** classification performance of Income balanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 92.83% | 88% | 98% |
| USVM | 91.83% | 84% | 100% |
| _TSVM_ | 96.33% | 93% | 99% |
| UTSVM | 88% | 81% | 95% |
| QSVM | 52% | 5%  | 99% |
| UQSVM | 76.83% | 68% | 86% |
| QTSVM | 67.17% | 46% | 88% |

**Table 4** classification performance of Age balanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 66.83% | 70% | 64% |
| USVM | 66.33% | 71% | 61% |
| _TSVM_ | 65.67% | 70% | 62% |
| UTSVM | 66.67% | 71% | 63% |
| QSVM | 68% | 65% | 71% |
| UQSVM | 66.5% | 67% | 66% |
| QTSVM | 66.5% | 84% | 49% |

**Table 5** classification performance of Smoking balanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 76% | 57% | 95% |
| USVM | 76.17% | 57% | 96% |
| _TSVM_ | 76% | 57% | 95% |
| UTSVM | 75.67% | 60% | 91% |
| QSVM | 63.5% | 87% | 40% |
| UQSVM | 76.33% | 58% | 95% |
| QTSVM | 52.67% | 98% | 8%  |

**Table 6** classification performance of Mushroom balanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 62% | 65% | 59% |
| USVM | 60.5% | 62% | 59% |
| _TSVM_ | 61.83% | 63% | 61% |
| UTSVM | 61.5% | 62% | 61% |
| QSVM | 62.83% | 62% | 64% |
| UQSVM | 63.83% | 66% | 62% |
| QTSVM | 64.17% | 63% | 65% |

The evaluation of SVM, QSVM, USVM, TSVM, QTSVM, and U-TSVM on balanced datasets reveals varying performance across different models and tasks. Traditional SVM demonstrates consistent accuracy with balanced recall values across datasets, serving as a strong baseline.

Models incorporating Universum points (USVM, U-TSVM, UQSVM) show improved flexibility and often achieve competitive recall for both classes, as observed in the Income (Table 3) and Smoking datasets (Table 5). For instance, USVM achieves 100% recall on one class in the Income dataset while maintaining over 90% accuracy.

In contrast, models like QSVM and QTSVM display significant variations in performance, often favoring one class. For example, in the Fraud dataset (Table 2), QTSVM achieves 95% recall for one class but sacrifices overall accuracy. Similarly, in the Smoking dataset (Table 5), QSVM prioritizes one class with 87% recall but underperforms on the other. Despite these fluctuations, QSVM performs well in the Age (Table 4) and Mushroom (Table 6) datasets, where accuracy and recall remain more balanced.

Overall, the results highlight the strengths of traditional SVM and the added flexibility of Universum-based methods, while also indicating that advanced models like QSVM and QTSVM require careful tuning to achieve consistent and balanced performance.

**3.2 Performance on Imbalanced datasets**

We evaluate the performance of SVM, QSVM, USVM, TSVM, QTSVM, and U-TSVM on both balanced and imbalanced datasets to examine whether the incorporation of flexible decision boundaries and Universum points improves classification accuracy, enhances minority class detection, and overall model performance.

**Table 7** classification performance of Fraud imbalanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 70.5% | 96% | 45% |
| USVM | 59% | 99% | 19% |
| _TSVM_ | 18% | 21% | 15% |
| UTSVM | 82.25% | 80% | 85% |
| QSVM | 50% | 100% | 0%  |
| UQSVM | 72.25% | 93% | 52% |
| QTSVM | 78.75% | 72% | 85% |
| UQTSVM | 50% | 100% | 0%  |

**Table 8** classification performance of Income imbalanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 90% | 99% | 81% |
| USVM | 91.5% | 100% | 83% |
| _TSVM_ | 95.5% | 100% | 91% |
| UTSVM | 88.5% | 93% | 84% |
| QSVM | 50.75% | 100% | 0.1% |
| UQSVM | 58.25% | 99% | 17% |
| QTSVM | 69.5% | 90% | 49% |
| UQTSVM | 72.25% | 86% | 58% |

**Table 9** classification performance of Age imbalanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 63% | 89% | 37% |
| USVM | 65.5% | 83% | 47% |
| _TSVM_ | 61.5% | 89% | 34% |
| UTSVM | 72.5% | 71% | 73% |
| QSVM | 50% | 100% | 0%  |
| UQSVM | 50% | 99% | 1%  |
| QTSVM | 50% | 100% | 0%  |
| UQTSVM | 50% | 100% | 0%  |

**Table 10** classification performance of Smoking imbalanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 66.75% | 86% | 47% |
| USVM | 72.25% | 78% | 67% |
| _TSVM_ | 69.5% | 85% | 54% |
| UTSVM | 73.25% | 61% | 85% |
| QSVM | 50% | 100% | 0%  |
| UQSVM | 65% | 88% | 42% |
| QTSVM | 69.25% | 75% | 64% |
| UQTSVM | 60.5% | 94% | 28% |

**Table 11** classification performance of Mushroom imbalanced data

| Model | Accuracy | Recall (-1) | Recall (1) |
| --- | --- | --- | --- |
| SVM | 55% | 91% | 19% |
| USVM | 57.75% | 81% | 34% |
| _TSVM_ | 54.25% | 94% | 14% |
| UTSVM | 58.75% | 52% | 66% |
| QSVM | 50% | 100% | 0%  |
| UQSVM | 50% | 100% | 0%  |
| QTSVM | 61.25% | 42% | 80% |
| UQTSVM | 60% | 90% | 30% |

The evaluation on imbalanced datasets reveals that traditional SVM achieves high accuracy but struggles with minority class detection, favoring the majority class.

Models like UTSVM and QTSVM demonstrate improved minority class recall, particularly in the Fraud (Table 7) and Smoking (Table 10) datasets, achieving up to 85% minority recall while maintaining reasonable accuracy.

In contrast, QSVM and UQSVM display significant bias, achieving perfect majority recall (100%) but failing to detect the minority class (0%) across multiple datasets, such as Age (Table 9) and Mushroom (Table 11). TSVM performs well in the Income dataset (Table 8), achieving balanced accuracy and recall, while USVM shows moderate improvements in minority class detection across various datasets.

Overall, models incorporating flexible decision boundaries and Universum points (e.g., UTSVM, USVM) perform better in addressing class imbalance, though their effectiveness varies by dataset.

**4\. Discussion**

The evaluation of SVM, QSVM, USVM, TSVM, QTSVM, and U-TSVM on both balanced and imbalanced datasets highlights the strengths and limitations of flexible decision boundaries and Universum points in classification tasks.

On balanced datasets, traditional SVM and its variants such as USVM and UTSVM perform consistently well, achieving competitive accuracy and balanced recall values for both classes. For instance, in the Income balanced dataset (Table 3), SVM achieved 92.83% accuracy, while USVM and UTSVM achieved similarly strong performances, demonstrating the reliability of these approaches when no significant imbalance exists. However, models like QSVM and QTSVM occasionally prioritized one class, as seen in the Fraud balanced dataset (Table 2), where QSVM achieved high majority class recall but struggled with minority class detection.

On imbalanced datasets, the results reveal a more noticeable distinction in model behavior. Traditional SVM tends to favor the majority class, achieving high accuracy but underperforming on minority class recall, as seen in the Fraud (Table 7) and Age (Table 9) datasets. Models incorporating Universum points (e.g., USVM, UTSVM) and flexible boundaries (e.g., QTSVM) exhibit improved minority class recall. For instance, UTSVM achieved 85% recall for the minority class in the Fraud dataset while maintaining overall accuracy, significantly outperforming traditional SVM.

However, some advanced models, such as QSVM and QTSVM, exhibit inconsistent behavior, particularly in datasets like Smoking (Table 10), where they show extreme bias toward the majority class. These results indicate that while flexible boundaries and Universum points can enhance performance, their impact depends on the dataset characteristics and model implementation.

**5\. Conclusion**

Imbalanced datasets remain a significant challenge for machine learning, as models often favor the majority class while underperforming on the minority class, which is often the focus in real-world applications. This study examined the performance of various SVM-based models, including QSVM, USVM, TSVM, QTSVM, and U-TSVM, on both balanced and imbalanced datasets.

The results demonstrate that on balanced datasets, most models achieve strong overall accuracy and balanced class recall. On imbalanced datasets, methods incorporating Universum points (e.g., USVM, UTSVM) and flexible decision boundaries (e.g., QTSVM) exhibit improved minority class performance, particularly in datasets like Fraud and Smoking. However, certain models show bias toward the majority class, indicating that their effectiveness depends on the dataset's structure and appropriate tuning.

In conclusion, flexible boundaries and Universum points provide promising approaches for addressing class imbalance and improving minority class detection. Future work can focus on optimizing these methods for more complex datasets to further enhance classification performance.

**6\. reference**

\[1\] Hearst, M. A., Dumais, S. T., Osuna, E., Platt, J., & Others. (1998). Support vector machines.

\[2\] Author(s). (2008). _Quadratic kernel-free non-linear support vector machine_. **Volume 41**, 15-30.

\[3\] Qi, Z., Tian, Y., & Shi, Y. (2012). Twin support vector machine with Universum data. _Neural Networks, 36_, 112-119.

\[4\] Ding, S., Yu, J., Qi, B., & Huang, H. (2014). An overview on twin support vector machines. _Artificial Intelligence Review, 41_(2), 145-176.

\[5\] Gao, Q. Q., Bai, Y. Q., & Zhan, Y. R. (2019). Quadratic kernel-free least square twin support vector machine for binary classification problems. _Journal of the Operations Research Society_, _70_(4), 567-581.

\[6\] Xu, Y., Chen, M., & Li, G. (2016). Least squares twin support vector machine with Universum data for classification. _International Journal of Systems Science, 47_(12), 2915-2926.

\[7\] Long, M., Cao, Y., Cao, Z., et al.: 'Transferable representation learning with deep adaptation networks', IEEE Trans. Pattern Anal. Mach. Intell., 2018, 41, (12), pp. 3071-3085

\[8\] López-Rojas, E. A. (2016). _Synthetic Financial Datasets For Fraud Detection_ \[Data set\]. Kaggle. Retrieved from <https://www.kaggle.com/datasets/ealaxi/paysim1>

\[9\] Sawhney, P. (2023). _Mushroom Dataset (Binary Classification)_ \[Data set\]. Kaggle. Retrieved from <https://www.kaggle.com/datasets/prishasawhney/mushroom-dataset-binary-classification>

\[10\] De Tomasi, L. (2018). _Income Classification_ \[Data set\]. Kaggle. Retrieved from <https://www.kaggle.com/datasets/lorenzodetomasi/income-classification>

\[11\] Sawhney, P. (2023). _Age Prediction Dataset (Binary Classification)_ \[Data set\]. Kaggle. Retrieved from <https://www.kaggle.com/datasets/prishasawhney/age-prediction-dataset-binary-classification>

\[12\] kukuroo3. (2021). _Body Signal of Smoking: Find Smokers by Vital Signs (Binary Classification)_ \[Data set\]. Kaggle. Retrieved from <https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking>
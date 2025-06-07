# Report

| Training instance | Eng. Name  | Regularizer | Optimizer | Early Stopping | Dropout Rate | Accuracy | F1 score | Recall | Precision |
|-------------------|------------|-------------|-----------|----------------|--------------|----------|----------|--------|-----------|
| 1                 | Lievin     | L1          | Adamax    | 16             | 0.2          | 0.6707   | 0.3721   | 0.2595 | 0.6575    |
| 2                 | Deolinda   | L2          | Adam      | 4              | 0.35         | 0.6951   | 0.4485   | 0.3297 | 0.7011    |
| 3                 | Ivan       | L2          | SGD       | 10             | 0.5          | 0.6687   | 0.2488   | 0.1459 | 0.8438    |
| 4                 | Diana      | L1          | RMSprop   | 4              | 0.4          | 0.6809   | 0.4075   | 0.2919 | 0.6750    |
| 5                 | Owen       | L2          | Adagrad   | 3              | 0.3          | 0.6240   | 0.0000   | 0.0000 | 0.0000    |

## Shema Ivan

As part of our group project, I was responsible for Training Instance 3, using the following techniques:
- **Regularizer**: L1 regularization, which helps in feature selection by penalizing the absolute magnitude of coefficients.
- **Optimizer**: SGD (Stochastic Gradient Descent), a widely used optimizer known for its simplicity and effectiveness in handling large-scale data.
- **Early Stopping**: Set at 10 epochs to prevent overfitting by monitoring validation loss.
- **Dropout Rate**: 0.5 to reduce overfitting by randomly deactivating 50% of neurons during training.
These techniques aimed to find a balance between generalization and model complexity.

### Summary of Findings and their Significance
1. **Balanced Performance**: My model achieved relatively balanced results across the metrics—second-highest precision and moderate accuracy, though recall was on the lower side.
2. **Effect of L1 Regularization**: L1 seemed effective in reducing overfitting without overly compromising performance.
3. **Trade-offs observed**: While precision was strong (good at identifying relevant results), the lower recall suggests the model missed a number of true positives.

### Insights from Experiments
- **SGD's Behavior**: Requires careful tuning—learning rate and batch size are crucial. It tends to converge slowly but steadily.
- **Dropout at 0.5**: Helped prevent overfitting but might have contributed to lower recall by being too aggressive in deactivating neurons.
- **L1 Regularization**: Reduced model complexity, which helped improve precision but might have hurt recall by discarding potentially useful features.

### Challenges Faced
1. **Tuning SGD**: Unlike adaptive optimizers (e.g., Adam), SGD required manual tuning of learning rate and batch size for effective convergence.
2. **Recall vs Precision Tradeoff**: It was challenging to improve recall without hurting precision or overfitting the model.
3. **Training Stability**: With high dropout and L1 regularization, training sometimes became unstable or slow to converge.
4. **Model Sensitivity**: The model's performance was sensitive to minor changes in hyperparameters, which required extensive experimentation.

### Model Evaluation
In this experiment, my model used SGD with L2 regularization, achieving high accuracy (0.8438) and strong precision (0.6687), but low recall (0.1459) and F1 score (0.2488). This indicates the model was good at avoiding false positives but still failed to capture many actual positives.
- **Deolinda’s model** outperformed all, with the highest F1 score (0.5110) and recall (0.4219). Her use of Adam, L2 regularization, and a balanced dropout rate enabled better generalization and detection of positives.
- **Owen’s model** performed poorly despite perfect precision (1.000), due to extremely low recall (0.0054) and F1 score (0.0108). It likely predicted almost no positives, making it ineffective. His choice of Adagrad and low dropout may have limited learning.
In short, Deolinda achieved the best balance, my model was second-best with high accuracy and precision, and Owen’s model lacked recall, severely impacting its usefulness.

## Ganza Owen Yhaan

### Model Comparison and Evaluation
As part of our group project, I was responsible for Training Instance 5, using the following techniques:
- **Regularizer**: L2 regularization, which penalizes the squared magnitude of coefficients to prevent overfitting and maintain model stability.
- **Optimizer**: Adagrad, an adaptive optimizer that adjusts the learning rate based on past gradients, chosen to handle the varying scales of water quality features.
- **Early Stopping**: Set at 3 epochs to prevent overfitting by monitoring validation loss, aiming for quick termination if performance plateaus.
- **Dropout Rate**: 0.3 to reduce overfitting by randomly deactivating 30% of neurons during training, seeking a balance between regularization and model capacity.
These techniques were selected to balance generalization and computational efficiency given the dataset's characteristics.

#### Summary of Findings and their Significance
- **Balanced Performance**: My model achieved a moderate accuracy (0.6240), but zero F1 Score, Recall, and Precision, indicating a strong bias toward the majority class due to class imbalance.
- **Effect of L2 Regularization**: L2 helped stabilize the model but may have overly penalized weights, contributing to the inability to predict the minority class.
- **Trade-offs Observed**: The zero Recall and F1 suggest the model missed all "Potable" cases, likely prioritizing "Not Potable" accuracy, while the zero Precision indicates no correct positive predictions were made.

#### Insights from Experiments
- **Adagrad's Behavior**: Adagrad’s adaptive nature was intended to suit the dataset, but its slow convergence, especially with a low learning rate (0.001), limited effective training within 3 epochs.
- **Dropout at 0.3**: This rate aimed to prevent overfitting but, combined with early stopping, may have reduced the model’s capacity to learn the minority class.
- **L2 Regularization**: Reduced overfitting but potentially discarded useful features for the minority class, exacerbating the imbalance issue.

#### Challenges Faced
- **Tuning Adagrad**: The optimizer required careful learning rate tuning, and the low rate (0.001) likely hindered convergence.
- **Recall vs Precision Tradeoff**: Addressing the zero Recall was challenging without class weighting, as the model favored the majority class.
- **Training Stability**: Early stopping at 3 epochs and L2 regularization sometimes led to unstable or premature convergence.
- **Model Sensitivity**: Performance was highly sensitive to early stopping and learning rate, requiring further experimentation.

#### Model Evaluation
In this experiment, my model used Adagrad with L2 regularization, achieving Accuracy 0.6240, but zero F1 Score, Recall, and Precision. This indicates the model excelled at predicting the majority "Not Potable" class due to class imbalance, but failed to identify any "Potable" cases, rendering it ineffective for this task.
- **Lievin’s model** (Training Instance 1) performed well, with Accuracy 0.6707 and F1 Score 0.3721, using Adamax, L1 regularization, patience 10, and dropout 0.2. The higher patience and adaptive optimizer likely helped balance the classes, making it more effective at detecting positives.
- **Ivan’s model** (Training Instance 3) also outperformed mine, with Accuracy 0.6687 and F1 Score 0.2488, using SGD, L2 regularization, patience 10, and dropout 0.5. The longer patience and higher dropout likely mitigated imbalance better, improving recall (0.1459) compared to my zero recall.
In short, Lievin achieved the best F1 score and balance, Ivan followed with solid accuracy and moderate F1, while my model’s zero recall and F1, due to unaddressed imbalance and early stopping, made it the least effective. To improve, I’d increase patience to 10, adjust the learning rate to 0.005, and add class weights to address the imbalance.

## Lievin Murayire

As part of our group project, I was responsible for Training Instance 1 and implemented the following configuration:
- **Regularizer (L1)**: L1 regularization to encourage sparsity and help with feature selection.
- **Optimizer (Adamax)**: Adamax was chosen for its adaptive learning rate behavior, useful when tuning is limited.
- **Dropout Rate**: 0.2 applied to mitigate overfitting.
- **Early Stopping**: Triggered at 16 epochs to avoid overfitting.
These settings aimed to produce a model that generalizes well without being overly complex.

### Model Performance (lievin_adamax_dropout_model)
- **Accuracy**: 0.6707
- **Precision**: 0.6575
- **Recall**: 0.2595
- **F1 Score**: 0.3721
- **Interpretation**: The model is very precise when it predicts a positive (potable water), but it misses many true positives. This results in a low recall and moderate F1 score.

### Summary of Findings
- **Precision Strength**: My model showed second-highest precision, indicating it avoided many false positives.
- **L1 Regularization**: Likely contributed to reducing noise and complexity, improving precision but hurting recall.
- **Trade-Off**: High selectivity (precision) came at the cost of sensitivity (recall), lowering the F1 score.

### Insights from Experiments
- **Dropout**: At 0.2, it controlled overfitting but might have suppressed feature interactions necessary for higher recall.
- **Optimizer Behavior**: Adamax handled the learning process well without much fluctuation.
- **Model Sensitivity**: Minor hyperparameter changes often caused shifts in model performance, especially on recall.

### Challenges Faced
- **Balancing Recall and Precision**: Increasing recall typically decreased precision.
- **Hyperparameter Tuning**: L1 regularization and dropout needed careful adjustment to avoid underfitting.

### Comparison with Peers
- **Deolinda (deolinda_adam_l2_dropout)**
  - **Accuracy**: 0.6829
  - **Precision**: 0.6480
  - **Recall**: 0.4219
  - **F1 Score**: 0.5110
  - **Highlights**: Deolinda's model had the best overall balance. Adam optimizer and L2 regularization likely allowed her model to generalize better.
- **Ivan (ivan_sgd_l2_dropout)**
  - **Accuracy**: 0.6687
  - **Precision**: 0.8438
  - **Recall**: 0.1459
  - **F1 Score**: 0.2488
  - **Highlights**: Ivan’s model achieved the highest precision but suffered from very low recall, making it overly conservative and missing too many positives.

### Final Thoughts
While Deolinda’s model achieved the most balanced performance, mine came close with strong precision and stable training. Ivan's model was overly cautious. To boost recall and F1 score, I would experiment with L2 regularization and slightly reduce dropout.

## Deolinda Bogore

As part of our group project, I was responsible for Training Instance 2, using the following techniques:
- **Regularizer**: L2 regularization, which helps prevent overfitting by penalizing large weights and encourages simpler models.
- **Optimizer**: Adam, an adaptive optimizer that combines momentum and adaptive learning rates, enabling efficient and stable convergence on water quality features.
- **Early Stopping**: Set at 4 epochs to avoid overfitting by halting training once validation loss stopped improving.
- **Dropout Rate**: 0.35 to randomly deactivate 35% of neurons during training, reducing overfitting and promoting generalization.
These choices aimed to balance learning complexity with robustness given the class imbalance in water quality data.

### Summary of Findings and Their Significance
- **Highest F1 Score (0.4485)**: My model achieved the best F1 score in the group, indicating a balanced ability to detect both potable and non-potable water cases.
- **Strong Precision (0.7011)**: The model made positive “potable” predictions with relatively low false positives, critical for minimizing unnecessary alarms.
- **Improved Recall (0.3297)**: The model successfully identified a meaningful portion of actual potable water samples, outperforming most teammates.
- **Balanced Performance**: The combination of L2 regularization, Adam optimizer, and moderate dropout contributed to a model that generalized well without overfitting.

### Insights from Experiments
- **Adam Optimizer**: Its adaptive nature helped the model quickly and reliably converge despite the noisy and imbalanced water quality features.
- **L2 Regularization and Dropout**: Together, these prevented overfitting by penalizing complexity and forcing the model to rely on robust patterns.
- **Early Stopping**: Prevented the model from overfitting but may have slightly limited recall, as the model might have stopped before fully learning minority class patterns.

### Challenges Faced
- **Understanding why evaluation metrics like Accuracy, Precision, Recall, and F1 Score change each time the model training cell is rerun.**
- **Early Stopping Sensitivity**: Stopping at 4 epochs helped prevent overfitting but may have cut off learning before the model fully recognized minority class features.
- **Dropout Tuning**: Balancing dropout to avoid both underfitting and overfitting required experimentation.

### Comparison with Teammates
- **Lievin (Training Instance 1)**
  - **Accuracy**: 0.6707
  - **F1 Score**: 0.3721
  - **Precision**: 0.6575
  - **Recall**: 0.2595
  - **Why my model is better**: My model’s F1 score (0.4485) and recall (0.3297) are higher, meaning it detects more potable water cases without losing precision. Adam optimizer (mine) appears better suited than Adamax (Lievin’s) for the dataset, improving learning stability and convergence.
- **Ivan (Training Instance 3)**
  - **Accuracy**: 0.6687
  - **F1 Score**: 0.2488
  - **Precision**: 0.8438
  - **Recall**: 0.1459
  - **Why my model is better**: Ivan’s model has much lower recall and F1, meaning it misses many potable water samples despite high precision. Ivan’s high dropout (0.5) might have limited learning capacity, whereas my moderate dropout (0.35) balanced generalization and learning better.

### Model Evaluation
Overall, my model showed the best balance between precision and recall among the team, reflected in the highest F1 score (0.4485). This makes it the most reliable. The Adam optimizer combined with L2 regularization and moderate dropout allowed stable and effective learning on this challenging dataset. I will increase early stopping patience to let the model learn more from the minority class.

## Diana

### Model Comparison and Evaluation
As part of our group project, I was responsible for Training Instance 4, using the following techniques:
- **Regularizer**: L1 regularization, which helps in feature selection by penalizing the absolute magnitude of coefficients.
- **Optimizer**: RMSprop, a widely used adaptive optimizer known for its effectiveness in handling non-stationary objectives.
- **Early Stopping**: Set at 4 epochs to prevent overfitting by monitoring validation loss.
- **Dropout Rate**: 0.4 to reduce overfitting by randomly deactivating 40% of neurons during training.
These techniques aimed to find a balance between generalization and model complexity.

#### Summary of Findings and their Significance
- **Balanced Performance**: My model achieved relatively balanced results across the metrics—moderate accuracy, a decent F1 score, and good precision, though recall was on the lower side.
- **Effect of L1 Regularization**: L1 seemed effective in reducing overfitting and improving precision through feature selection.
- **Trade-offs observed**: While precision was strong (good at identifying relevant results), the lower recall suggests the model missed a number of true positives.

#### Insights from Experiments
- **RMSprop's Behavior**: Requires careful tuning—learning rate and momentum are crucial. It tends to converge steadily but needed adjustment for optimal performance.
- **Dropout at 0.4**: Helped prevent overfitting but might have contributed to lower recall by being too aggressive in deactivating neurons.
- **L1 Regularization**: Reduced model complexity, which helped improve precision but might have hurt recall by discarding potentially useful features.

#### Challenges Faced
- **Tuning RMSprop**: Unlike other optimizers, RMSprop required manual tuning of its learning rate for effective convergence.
- **Recall vs Precision Tradeoff**: It was challenging to improve recall without hurting precision or overfitting the model.
- **Training Stability**: With high dropout and early stopping, training sometimes became unstable or slow to converge.
- **Model Sensitivity**: The model's performance was sensitive to minor changes in hyperparameters, which required extensive experimentation.

#### Model Evaluation
In this experiment, my model used RMSprop with L1 regularization, achieving high accuracy (0.6809) and strong precision (0.6750), but low recall (0.2919) and F1 score (0.4075). This indicates the model was good at avoiding false positives but still failed to capture many actual positives due to class imbalance.
- **Ivan’s model** outperformed mine slightly, with the highest accuracy (0.6687) and a moderate F1 score (0.2488). His use of SGD, L2 regularization, and a high dropout rate (0.5) with a patience of 10 enabled better generalization, though recall (0.1459) was lower, suggesting a trade-off.
- **Deolinda’s model** was the best, with the highest F1 score (0.4485) and recall (0.3297). Her use of Adam, L2 regularization, and a balanced dropout rate (0.35) with a patience of 4 likely optimized convergence and balanced class prediction.
In short, Deolinda achieved the best balance with the highest F1 score, Ivan showed strong accuracy but lower F1 due to imbalance handling, and my model offered a solid middle ground with good precision but room for recall improvement.

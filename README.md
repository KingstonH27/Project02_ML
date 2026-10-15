# Polynomial Regression Model

A simple machine learning project that implements **polynomial regression using gradient descent** to learn relationships between input and target values. It uses SymPy to calculate derivatives and Pandas to load data.
ChatGPT was used for assistance on this project, however I wrote almost the whole thing manually from scratch.

## Requirements

```
pip install pandas sympy
```

Place `model.py` and `housing.csv` in the same directory.

## Configuration

Modify these variables near the top of `model.py`:

| Variable               | Description                         |
| ---------------------- | ----------------------------------- |
| `answerCol`            | CSV column containing target values |
| `inputCol`             | CSV column containing input values  |
| `rowCount`             | Number of rows used for training    |
| `finishThreshold`      | Maximum acceptable prediction error |
| `maxTrainingSteps`     | Maximum number of training steps    |
| `learningRate`         | Controls the size of weight updates |
| `optimizeLearningRate` | Automatically tests learning rates  |

**Note:** Column indices start at `0`.

## How It Works

The model uses a third-degree polynomial:

[\
y=b+\frac{w_1}{10}x+\frac{w_2}{100}x^2+\frac{w_3}{1000}x^3\
]

- **Normalization:** Scales inputs to the range `-4` to `4`.
- **Random weights:** Initializes the polynomial's coefficients.
- **Gradient descent:** Updates weights to minimize squared prediction error.
- **Learning rate optimization:** Tests different learning rates and selects the one with the lowest average training steps.
- **Stopping condition:** Training ends when every training example meets the error threshold or the maximum step count is reached.

## Running the Program

1. Configure the dataset and training settings.
2. Run `model.py`.
3. Review initial predictions, trained predictions, error percentage, training time, and final weights.
4. Enter a numeric value when prompted to predict its output.
5. Press `Ctrl+C` to exit.

## Limitations

- Supports one input variable and a fixed third-degree polynomial.
- Trains on the first `rowCount` rows.
- Does not evaluate performance on unseen data.
- Results depend on random initialization and learning rate.

Clarify the prediction input scaleDocument CSV column and row assumptions
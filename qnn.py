import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, transpile, execute, Aer, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
import random

TRAINABLE_PARAMS = 20


def process_data():
    """
    Load the Iris dataset, normalize the data, and split it into training and testing sets.

    Returns:
    x_train, x_test, y_train, y_test: Training and testing data.
    """
    global data, x_train, y_train
    data = sklearn.datasets.load_iris()
    type(data)
    df = pd.DataFrame(data=data.data, columns=data.feature_names)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data)) * np.pi

    normalized_df = df.apply(NormalizeData)
    X = normalized_df
    y = pd.DataFrame(data.target, columns=["target"])
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    return x_train, x_test, y_train, y_test


def approximate_gradient(parameters, qnn, input_data, target, epsilon=1e-20):
    """
    Approximate the gradient of a loss function at the input predictions.

    Parameters:
    - parameters: Model parameters to compute the gradient.
    - qnn: Quantum neural network.
    - input_data: Input data.
    - target: Target labels.
    - epsilon: A small value to compute finite differences (default is 1e-20).

    Returns:
    - gradient: An approximate gradient of the same shape as the input parameters.
    """
    print(parameters)
    gradient = [None] * len(parameters)
    for i in range(len(parameters)):
        parameters_plus = np.copy(parameters)
        parameters_minus = np.copy(parameters)

        # Perturb the predictions by adding and subtracting epsilon to the i-th component
        parameters_plus[i] += epsilon
        parameters_minus[i] -= epsilon

        predict_plus = predict(parameters_plus, qnn, input_data)
        predict_minus = predict(parameters_minus, qnn, input_data)

        loss_plus = log_loss(y_true=target, y_pred=[predict_plus], labels=[0, 1, 2])
        loss_minus = log_loss(y_true=target, y_pred=[predict_minus], labels=[0, 1, 2])

        # Calculate the finite difference for the i-th component
        gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient


import random
import numpy as np


def initialize_adam_optimizer(parameters):
    # Initialize variables for the Adam optimizer
    m = [0] * len(parameters)  # Initialize first moment estimate
    v = [0] * len(parameters)  # Initialize second moment estimate
    beta1 = 0.9  # Decay rate for first moment estimate
    beta2 = 0.999  # Decay rate for second moment estimate
    epsilon = 1e-8  # Smoothing term to prevent division by zero

    return m, v, beta1, beta2, epsilon


def update_parameters_with_adam(params, loss_gradient, m, v, t, beta1, beta2, epsilon, learning_rate):
    """
    Update model parameters using the Adam optimizer.

    Parameters:
    - params: Model parameters.
    - loss_gradient: Gradient of the loss.
    - m: First moment estimate.
    - v: Second moment estimate.
    - t: Time step (iteration count).
    - beta1: Decay rate for the first moment estimate.
    - beta2: Decay rate for the second moment estimate.
    - epsilon: Smoothing term to prevent division by zero.
    - learning_rate: Initial learning rate.

    Returns:
    updated_params: Updated model parameters.
    m: Updated first moment estimate.
    v: Updated second moment estimate.
    """
    # Update the first moment estimate
    m = [(beta1 * m[i] + (1 - beta1) * loss_gradient[i]) for i in range(len(params))]

    # Update the second moment estimate
    v = [(beta2 * v[i] + (1 - beta2) * (loss_gradient[i] ** 2)) for i in range(len(params))]

    # Bias-corrected first moment estimate
    m_hat = [m[i] / (1 - beta1 ** t) for i in range(len(params))]

    # Bias-corrected second moment estimate
    v_hat = [v[i] / (1 - beta2 ** t) for i in range(len(params))]

    # Update the parameters with the adaptive learning rate
    updated_params = [params[i] - (learning_rate / (np.sqrt(v_hat[i]) + epsilon)) * m_hat[i] for i in
                      range(len(params))]

    # Enforce the [0, π] constraint
    updated_params = [max(0, min(np.pi, p)) for p in updated_params]

    # Assign a random value in [0, π] if a parameter reaches 0 or π
    updated_params = [p if 0 < p < np.pi else random.uniform(0, np.pi) for p in updated_params]

    return updated_params, m, v


def build_qc():
    """
    Build a Quantum Circuit (qc) and define a parameterized quantum circuit.

    Returns:
    qc: Quantum Circuit.
    """
    q = QuantumRegister(11)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)

    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(13)]

    for i in range(4):
        qc.rx(features[i], i)

    k = 0

    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.crx(params[k], i, i + 1)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(4):
        qc.crx(params[k], i + 1, 4 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.crx(params[k], i + 4, i + 5)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(3):
        qc.crx(params[k], i + 5, 8 + j)
        print(f'k: {k}, i: {i}, j: {j + 5}')
        k += 1
    qc.measure(q[8:], c)
    return qc


def build_qc_big():
    """
        Build a Quantum Circuit (qc) and define a parameterized quantum circuit.

        Returns:
        qc: Quantum Circuit.
        """
    q = QuantumRegister(11)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)

    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(TRAINABLE_PARAMS)]

    for i in range(4):
        qc.rx(features[i], i)

    k = 0
    for j in range(4):
        # Add parameterized quantum convolution layers
        for i in range(3):
            qc.crx(params[k], i, i + 1)
            print(f'k: {k}, i: {i}')
            k += 1
        qc.crx(params[k], i + 1, 4 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    for j in range(3):
        # Add parameterized quantum convolution layers
        for i in range(3):
            qc.crx(params[k], i + 4, i + 5)
            print(f'k: {k}, i: {i}')
            k += 1
        qc.crx(params[k], i + 5, 8 + j)
        print(f'k: {k}, i: {i}, j: {j + 5}')
        k += 1

    qc.measure(q[8:], c)
    return qc


def build_qc_ent():
    """
        Build a Quantum Circuit (qc) and define a parameterized quantum circuit.

        Returns:
        qc: Quantum Circuit.
        """
    q = QuantumRegister(11)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)

    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(28)]

    for i in range(4):
        qc.rx(features[i], i)

    k = 0
    for j in range(4):
        # Add parameterized quantum convolution layers
        for i in range(3):
            qc.cry(params[k], i, i + 1)
            qc.cx(i, i + 1)
            print(f'k: {k}, i: {i}')
            k += 1
        qc.crx(params[k], i + 1, 4 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    for j in range(3):
        # Add parameterized quantum convolution layers
        for i in range(3):
            qc.cry(params[k], i + 4, i + 5)
            qc.cx(i + 4, i + 5)
            print(f'k: {k}, i: {i}')
            k += 1
        qc.crx(params[k], i + 5, 8 + j)
        print(f'k: {k}, i: {i}, j: {j + 5}')
        k += 1

    qc.measure(q[8:], c)
    return qc


def build_qc_ent_small():
    """
        Build a Quantum Circuit (qc) and define a parameterized quantum circuit.

        Returns:
        qc: Quantum Circuit.
        """
    q = QuantumRegister(11)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)

    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(13)]

    for i in range(4):
        qc.rx(features[i], i)

    k = 0
    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.cry(params[k], i, i + 1)
        qc.cx(i, i + 1)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(4):
        qc.cry(params[k], i + 1, 4 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.cry(params[k], i + 4, i + 5)
        qc.cx(i + 4, i + 5)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(3):
        qc.cry(params[k], i + 5, 8 + j)
        print(f'k: {k}, i: {i}, j: {j + 5}')
        k += 1
    qc.measure(q[8:], c)
    return qc


def build_qc_ent_small_conv():
    """
        Build a Quantum Circuit (qc) and define a parameterized quantum circuit.

        Returns:
        qc: Quantum Circuit.
        """
    q = QuantumRegister(15)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)

    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(22)]

    for i in range(4):
        qc.rx(features[i], i)

    k = 0
    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.cry(params[k], i, i + 1)
        qc.cx(i, i + 1)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(4):
        qc.cry(params[k], i + 1, 4 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.cry(params[k], i + 4, i + 5)
        qc.cx(i + 4, i + 5)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(4):
        qc.cry(params[k], i + 5, 8 + j)
        print(f'k: {k}, i: {i}, j: {j + 4}')
        k += 1

    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.cry(params[k], i + 8, i + 9)
        qc.cx(i + 8, i + 9)
        print(f'k: {k}, i: {i}')
        k += 1

    for j in range(3):
        qc.cry(params[k], i + 9, 12 + j)
        print(f'k: {k}, i: {i}, j: {j + 5}')
        k += 1
    qc.measure(q[12:], c)
    return qc


def loss_function(qnn, input_data, target_data):
    """
    Calculate the loss function using predictions from a quantum neural network.

    Parameters:
    - qnn: Quantum neural network.
    - input_data: Input data.
    - target_data: Target labels.

    Returns:
    loss: The computed loss.
    """
    # Perform forward pass to obtain predictions
    predictions = predict(qnn, input_data)

    # Compute the loss (e.g., mean squared error)
    loss = log_loss(predictions, target_data)

    return loss


def bind_parameter_values(params, qnn, input_sample):
    """
    Bind parameter values to a quantum neural network and input sample.

    Parameters:
    - params: List of parameter values.
    - qnn: Quantum neural network.
    - input_sample: Input data sample.

    Returns:
    assignedQNN: Quantum neural network with bound parameters.
    """
    params = input_sample + list(params)
    assignedQNN = qnn.assign_parameters(params)
    return assignedQNN


def process_measurement_results(counts, shots):
    """
    Process measurement results and calculate qubit probabilities.

    Parameters:
    - counts: Measurement outcomes.
    - shots: Number of measurement shots.

    Returns:
    qubit_probabilities: List of qubit probabilities.
    """
    # Initialize a list to store the probabilities for each qubit
    qubit_probabilities = [0] * 3
    total_ones = 0
    # Iterate through each qubit
    for qubit in range(3):
        # Count the outcomes where the qubit is '1'
        qubit_counts_1 = sum(counts[outcome] for outcome in counts if outcome[qubit] == '1')
        total_ones += qubit_counts_1
        qubit_probabilities[qubit] = qubit_counts_1
    if total_ones > 0:
        qubit_probabilities = [x / total_ones for x in qubit_probabilities]
    else:
        qubit_probabilities = [1 / 3, 1 / 3, 1 / 3]

    # Return the list of probabilities
    return qubit_probabilities


# Define a prediction function that runs the QNN
def predict(params, qnn, input_sample):
    """
    Perform quantum circuit execution and process measurement results to make predictions.

    Parameters:
    - params: Model parameters.
    - qnn: Quantum neural network.
    - input_data: Input data.

    Returns:
    prediction: Predicted values.
    """
    # Apply the input data to the QNN
    predictions = []
    backend = Aer.get_backend('qasm_simulator')

    assignedQNN = bind_parameter_values(params=params, qnn=qnn, input_sample=input_sample)
    transpiled_circuit = transpile(assignedQNN, backend)
    shots = 100
    job = execute(transpiled_circuit, backend=backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    # Process the measurement results and obtain predictions
    prediction = process_measurement_results(counts, shots)
    return prediction


# Define a function to update parameters
def update_parameters(params, loss_gradient, learning_rate):
    """
    Update model parameters using gradient descent.

    Parameters:
    - params: Model parameters.
    - loss_gradient: Gradient of the loss.
    - learning_rate: Learning rate for gradient descent.

    Returns:
    updated_params: Updated model parameters.
    """
    updated_params = [p - learning_rate * g for p, g in zip(params, loss_gradient)]

    # Enforce the [0, π] constraint
    updated_params = [max(0, min(np.pi, p)) for p in updated_params]

    # Assign a random value in [0, π] if a parameter reaches 0 or π
    updated_params = [p if 0 < p < np.pi else random.uniform(0, np.pi) for p in updated_params]

    return updated_params


def train_qnn(x_train, y_train, batch_size=32):
    # Build qnn circuit
    qc = build_qc_ent_small_conv()

    # Initialize model parameters
    params = [1.5] * TRAINABLE_PARAMS
    num_epochs = 20

    input_data = x_train.values.tolist()[:50]
    target_data = y_train.values.tolist()[:50]
    num_samples = len(input_data)

    # Initialize the Adam optimizer variables
    t = 0
    m, v, beta1, beta2, epsilon = initialize_adam_optimizer(
        params)  # Initialize these variables before the training loop

    for epoch in range(num_epochs):
        total_loss = 0

        # Split the training data into batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_input = input_data[batch_start:batch_end]
            batch_target = target_data[batch_start:batch_end]

            # Initialize batch gradient
            batch_gradient = [0.0] * len(params)

            batch_loss = 0

            for input_sample, target in zip(batch_input, batch_target):
                predictions = predict(params, qc, input_sample)
                loss = log_loss(y_true=target, y_pred=[predictions], labels=[0, 1, 2])
                loss_gradient = approximate_gradient(params, qc, input_sample, target)

                # Accumulate the gradient for this batch
                batch_gradient = [g + dg for g, dg in zip(batch_gradient, loss_gradient)]
                total_loss += loss
                batch_loss += loss
                print(f'avg batch loss: {batch_loss / batch_size}')

            # Update model parameters with the average gradient for the batch
            batch_gradient = [g / batch_size for g in batch_gradient]
            # m, v, beta1, beta2, epsilon = initialize_adam_optimizer(params)
            # params = update_parameters_with_adam(params, batch_gradient, m, v, beta1, beta2, epsilon, 1e-3, 0.3)

            # Increment the time step (iteration count)
            t += 1

            # Update parameters using the Adam optimizer
            params, m, v = update_parameters_with_adam(params, batch_gradient, m, v, t, beta1, beta2, epsilon, 0.3)

            # params = update_parameters(params, batch_gradient, learning_rate=1e-10)

        avg_loss = total_loss / num_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")


def main():
    """
    Main function to execute the entire script.

    Returns:
    None
    """
    x_train, x_test, y_train, y_test = process_data()
    train_qnn(x_train, y_train, batch_size=3)

main()

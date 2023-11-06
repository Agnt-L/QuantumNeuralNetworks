import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from qiskit import QuantumCircuit, transpile, execute, Aer, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter



def process_data():
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
    return (x_train, x_test, y_train, y_test)


def approximate_gradient(parameters, qnn, input_data, target, epsilon=1e-20):
    """
    Approximate gradient of a loss function at the input predictions.

    Parameters:
    - loss_function: A function that takes a predictions as input and returns a scalar loss.
    - predictions: The predictions at which to compute the gradient.
    - epsilon: A small value to compute finite differences (default is 1e-6).

    Returns:
    - gradient: An approximate gradient predictions of the same shape as the input predictions.
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

        loss_plus = log_loss(y_true=target, y_pred=[predict_plus], labels=[0,1,2])
        loss_minus = log_loss(y_true=target, y_pred=[predict_minus], labels=[0,1,2])

        # Calculate the finite difference for the i-th component
        gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient


def build_qc():
    global qc
    q = QuantumRegister(11)
    c = ClassicalRegister(3)
    qc = QuantumCircuit(q, c)
    # Define the input feature vector
    features = [Parameter(f'f_{i}') for i in range(4)]
    params = [Parameter(f'p_{i}') for i in range(28)]
    for i in range(4):
        qc.rx(features[i], i)
    k = 0
    """# Add parameterized quantum convolution layers
    for i in range(3):
        qc.crx(params[k], i, i+1)
        print(f'k: {k}, i: {i}')
        k+=1
    
    for j in range(4):
        qc.crx(params[k],i+1,4+j)
        print(f'k: {k}, i: {i}, j: {j+4}')
        k+=1
    
    # Add parameterized quantum convolution layers
    for i in range(3):
        qc.crx(params[k], i + 4, i+5)
        print(f'k: {k}, i: {i}')
        k+=1
    
    for j in range(3):
        qc.crx(params[k], i+5, 8+j)
        print(f'k: {k}, i: {i}, j: {j+5}')
        k+=1"""
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
    print(qc.parameters)
    return qc


def bind_parameters_to_circuit(qnn, params, values):
    # Check if the number of parameters matches the number of values
    if len(params) != len(values):
        raise ValueError("The number of parameters must match the number of values.")

    # Create a parameter binding dictionary
    parameter_binding = {param: value for param, value in zip(params, values)}

    # Bind parameters to the circuit
    bound_circuit = qnn.bind_parameters(parameter_binding)

    return bound_circuit


def loss_function(qnn, input_data, target_data):
    # Perform forward pass to obtain predictions
    predictions = predict(qnn, input_data)

    # Compute the loss (e.g., mean squared error)
    loss = log_loss(predictions, target_data)

    return loss


def bind_parameter_values(params, qnn, input_sample):
    params = input_sample + list(params)
    assignedQNN = qnn.assign_parameters(params)
    return assignedQNN


def process_measurement_results(counts, shots):
    # Initialize a list to store the probabilities for each qubit
    qubit_probabilities = [0] * 3
    total_ones = 0
    # Iterate through each qubit
    for qubit in range(3):
        # Count the outcomes where the qubit is '1'
        qubit_counts_1 = sum(counts[outcome] for outcome in counts if outcome[qubit] == '1')
        total_ones += qubit_counts_1
        qubit_probabilities[qubit] = qubit_counts_1
    qubit_probabilities =  [x/total_ones for x in qubit_probabilities]

    # Return the list of probabilities
    return qubit_probabilities


# Define a prediction function that runs the QNN
def predict(params, qnn, input_data):
    # Apply the input data to the QNN
    predictions = []
    backend = Aer.get_backend('qasm_simulator')

    assignedQNN = bind_parameter_values(params=params, qnn=qnn, input_sample=input_sample)
    transpiled_circuit = transpile(assignedQNN, backend)
    shots = 1000
    job = execute(transpiled_circuit, backend=backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    # Process the measurement results and obtain predictions
    prediction = process_measurement_results(counts, shots)
    return prediction


# Define a function to update parameters
def update_parameters(params, loss_gradient, learning_rate):
    # Iterate through QNN parameters
    for i in range(len(params)):
        params[i] = params[i] - learning_rate * loss_gradient[i]
    return params

def train_qnn():
    global input_sample, target
    #to be trained
    params = [1.5] * 28

    # Main training loop
    input_data = x_train.values.tolist()
    target_data = y_train.values.tolist()
    num_epochs = 2
    print(qc.parameters)
    for epoch in range(1):
        print("ho")
        total_loss = 0
        for input_sample, target in zip(input_data, target_data):
            # Forward pass
            predictions = predict(params, qc, input_sample)
            loss = log_loss(y_true=target, y_pred=[predictions], labels=[0,1,2])
            # predictions = predict(qc, [input_sample])

            # Compute the loss gradient (you need to implement this)
            loss_gradient = approximate_gradient(params, qc, input_data, target)
            print(f'gradient: {loss_gradient}')
            # loss_gradient = compute_loss_gradient(predictions, target)

            # Update QNN parameters
            new_params = update_parameters(params, loss_gradient, 1e-19)
            params = new_params

            print(f"Params {params}, Loss: {loss}")

            # Accumulate the loss
            # total_loss += loss_function(qnn, [input_sample], [target])

        # avg_loss = total_loss / len(input_data)
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss}")


def main():
    x_train, x_test, y_train, y_test = process_data()
    qc = build_qc()
    train_qnn()
    print("hi")

main()
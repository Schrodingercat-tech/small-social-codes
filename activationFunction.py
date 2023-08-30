import numpy as np
import plotly.graph_objects as go

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.5):
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)

x = np.linspace(-2, 2, 100)  # Adjusted the range for better visualization
activations = [sigmoid(x), tanh(x), relu(x), leaky_relu(x), elu(x), softmax(x)]
activation_labels = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Softmax"]

traces = []

for activation, label in zip(activations, activation_labels):
    trace = go.Scatter(x=x, y=activation, mode='lines', name=label)
    traces.append(trace)

layout = go.Layout(
    title="Activation Functions",
    xaxis=dict(title="Input", showgrid=True),
    yaxis=dict(title="Output", showgrid=True),
    plot_bgcolor='rgb(10,50,50)',   # Set plot background color
    paper_bgcolor='rgb(200,100,0)',  # Set paper (around the plot) background color
    xaxis_gridcolor='rgb(2,10,0)', # Set x-axis grid color
    yaxis_gridcolor='rgb(2,10,0)', # Set y-axis grid color
    xaxis_gridwidth=1,      # Set x-axis grid width
    yaxis_gridwidth=1,      # Set y-axis grid width
    height= 1000,
    width= 1500
)

fig = go.Figure(data=traces, layout=layout)
fig.show()


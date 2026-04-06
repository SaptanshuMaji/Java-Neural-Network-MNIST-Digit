# Java-Neural-Network-MNIST-Digit
Fully implemented digit recognition in Java with no external libraries, trained and tested on MNIST dataset. Uses Swing GUI for real time handwritten digit recognition. 

# Features:
Built only using JDK (zero dependencies).
Manually implemented backpropagation, matrix manipulation. Gemini 3.1 Pro was used to help understand swing and was also implemented manually.
Uses adaptive movement estimation (ADAM) optimization. 
User can draw in real time, and the model will guess the number, and also give the top 3 choices as outputs (percentages of probability).

# Structure and Architecture:
This is a multi-layer perceptron.
Input: 784 neurons (grayscale 28x28 pixels)
Hidden layer: 256 neurons with leaky ReLu to avoid classic dying neuron problem.
Output: 10 neurons with softmax activation.
Loss func: cross entropy with label smoothing 
Uses ADAM optimization with automated decay (base learning rate is 0.001).
Logic for weight updates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2$$
$$\hat{w} = w - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

# How to run:
Clone the repository: git clone [https://github.com/SaptanshuMaji/Java-Neural-Network-MNIST-Digit.git](https://github.com/SaptanshuMaji/Java-Neural-Network-MNIST-Digit.git)
Compile and run (yes that's it, shocking): Open the project in IntelliJ Idea or compile on your terminal.

# Screenshots: ![MNIST demo](screenshots/demo.png)


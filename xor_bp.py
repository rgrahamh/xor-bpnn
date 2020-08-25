from math import exp
from random import random

class Node:
	def __init__(self, w, b):
		self.weights = w
		self.bias = b
	feed_forward_val = 0
	err = 0
	

class Network:
	#ni (->input_nodes) = An array of the input nodes
	#nh (->hidden_nodes) = An array of the hidden nodes
	#no (->output_nodes) = An array of the output nodes
	def __init__(self, ni, nh, no):
		self.input_nodes = ni
		self.hidden_nodes = nh
		self.output_nodes = no


def main():
	#train_vals[Inputs[0, 1], Result[true, false]]
	train_vals = [[[0, 0], [0, 1]], [[1, 0], [1, 0]], [[0, 1], [1, 0]], [[1, 1], [0, 1]]]
	test_vals = [[[0, 0], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [0, 1]], [[1, 1], [1, 0]]]

	#Input nodes
	ni1 = Node([random(), random()], random())
	ni2 = Node([random(), random()], random())
	#ni1 = Node([0.2, -0.3], 1)
	#ni2 = Node([0.4, 0.3], 1)

	#Hidden nodes
	hi1 = Node([random(), random()], random())
	hi2 = Node([random(), random()], random())
	#hi1 = Node([0.3, -0.2], 0.1)
	#hi2 = Node([0.5, -0.4], -0.1)

	#Output nodes
	oi1 = Node([random(), random()], random())
	oi2 = Node([random(), random()], random())
	#oi1 = Node([0, 0], -0.2)
	#oi2 = Node([0, 0], 0.3)
	
	#Building the network
	net = Network([ni1, ni2], [hi1, hi2], [oi1, oi2])

	for train_val in range(0, 15000):
		for train_case in train_vals:
			#Update input val
			for input_num in range(0, 2):
				net.input_nodes[input_num].bias = train_case[0][input_num]

			#Calculate forward prop
			for calc_num in range(0, 2):
				Nj = net.hidden_nodes[calc_num].bias
				for input_num in range(0, 2):
					Nj += net.input_nodes[input_num].bias * net.input_nodes[input_num].weights[calc_num]
				net.hidden_nodes[calc_num].feed_forward_val = 1/(1+exp(-1 * Nj))

			for calc_num in range(0, 2):
				Nj = net.output_nodes[calc_num].bias
				for input_num in range(0, 2):
					Nj += net.hidden_nodes[input_num].feed_forward_val * net.hidden_nodes[input_num].weights[calc_num]
				net.output_nodes[calc_num].feed_forward_val = 1/(1+exp(-1 * Nj))

			#Testing forward prop:
			#print("======================")
			#print("\nForward prop values:")
			#print("BotLeft:", net.hidden_nodes[0].feed_forward_val)
			#print("BotRight:", net.hidden_nodes[1].feed_forward_val)
			#print("TopLeft:", net.output_nodes[0].feed_forward_val)
			#print("TopRight:", net.output_nodes[1].feed_forward_val)
			
			#Calculate error
			for calc_num in range(0, 2):
				net.output_nodes[calc_num].err = net.output_nodes[calc_num].feed_forward_val * (1 - net.output_nodes[calc_num].feed_forward_val) * (net.input_nodes[calc_num].bias - net.output_nodes[calc_num].feed_forward_val)
			
			for calc_num in range(0, 2):
				net.hidden_nodes[calc_num].err = net.hidden_nodes[calc_num].feed_forward_val * (1 - net.hidden_nodes[calc_num].feed_forward_val) * ((net.output_nodes[0].err * net.hidden_nodes[calc_num].weights[0]) + (net.output_nodes[1].err * net.hidden_nodes[calc_num].weights[1]))
			
			#print("\nError calcs:")
			#print("BotLeft:", net.hidden_nodes[0].err)
			#print("BotRight:", net.hidden_nodes[1].err)
			#print("TopLeft:", net.output_nodes[0].err)
			#print("TopRight:", net.output_nodes[1].err)

			for input_num in range(0, 2):
				for calc_num in range(0, 2):
					net.input_nodes[input_num].weights[calc_num] += net.hidden_nodes[calc_num].err * net.input_nodes[input_num].bias
				net.hidden_nodes[input_num].bias += net.hidden_nodes[input_num].err

			for input_num in range(0, 2):
				for calc_num in range(0, 2):
					net.hidden_nodes[input_num].weights[calc_num] += net.output_nodes[calc_num].err * net.hidden_nodes[input_num].feed_forward_val
				net.output_nodes[input_num].bias += net.output_nodes[input_num].err
			
			#print("\nBiases:")
			#print("BotLeft:", net.hidden_nodes[0].bias)
			#print("BotRight:", net.hidden_nodes[1].bias)
			#print("TopLeft:", net.output_nodes[0].bias)
			#print("TopRight:", net.output_nodes[1].bias)

			#print("Input Weight 1->1:", net.input_nodes[0].weights[0])
			#print("Input Weight 1->2:", net.input_nodes[0].weights[1])
			#print("Input Weight 2->1:", net.input_nodes[1].weights[0])
			#print("Input Weight 2->2:", net.input_nodes[1].weights[1])
			#print("Hidden Weight 1->1:", net.hidden_nodes[0].weights[0])
			#print("Hidden Weight 1->2:", net.hidden_nodes[0].weights[1])
			#print("Hidden Weight 2->1:", net.hidden_nodes[1].weights[0])
			#print("Hidden Weight 2->2:", net.hidden_nodes[1].weights[1])


	for test_case in test_vals:
		for input_num in range(0, 2):
			net.input_nodes[input_num].bias = test_case[1][input_num]

		for calc_num in range(0, 2):
			Nj = net.hidden_nodes[calc_num].bias
			for input_num in range(0, 2):
				Nj += net.input_nodes[input_num].bias * net.input_nodes[input_num].weights[calc_num]
			net.hidden_nodes[calc_num].feed_forward_val = 1/(1+exp(-1 * Nj))

		for calc_num in range(0, 2):
			Nj = net.output_nodes[calc_num].bias
			for input_num in range(0, 2):
				Nj += net.hidden_nodes[input_num].feed_forward_val * net.hidden_nodes[input_num].weights[calc_num]
			net.output_nodes[calc_num].feed_forward_val = 1/(1+exp(-1 * Nj))

		print("Case:", test_case[0])
		print("Expected: ", test_case[1][0])
		print("Gotten: ", net.output_nodes[0].feed_forward_val)
		print("Expected: ", test_case[1][1])
		print("Gotten: ", net.output_nodes[1].feed_forward_val)

main()

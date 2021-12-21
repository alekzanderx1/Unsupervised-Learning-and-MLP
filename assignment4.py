import numpy as np

### Assignment 4 ###

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)


class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		activation = np.dot(input,self.w) + self.b
		# storing input value to use during backpropagation
		self.input = input
		return activation

	def backward(self, gradients):
		#Write backward pass here
		input = self.input
		wDelta =  np.dot(np.transpose(input),gradients) 
		inputDelta = np.dot(gradients,np.transpose(self.w))
		# updating the weight using the learning rate and gradients
		self.w = self.w - self.lr*wDelta
		self.b = self.b - self.lr*gradients
		return inputDelta


class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		sigmoid = 1 / (1 + np.exp(-input))
		# storing sigmoid value to use during backpropagation
		self.sigmoid = sigmoid
		return sigmoid

	def backward(self, gradients):
		#Write backward pass here
		sigmoid = self.sigmoid
		derivative = (sigmoid * (1 - sigmoid))
		result = gradients * derivative
		return result


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		diffs = (centroids - datapoint)**2
		return np.sqrt(diffs.sum(axis=1))

	def train(self, X):
		#training logic here
		#input is array of features (no labels)

		# initialize centroids by chosing k random elements from the input
		centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]

		# to keep track of which element is in which cluster to return the output
		# using a map for contant time lookup when required
		indexToClusterMap = {}
		
		clusters = []
		iterations = 0
		while iterations != self.t:
			# initialize the clusters for this iteration
			currentClusters = []
			for idx in range(self.k):
				currentClusters.append(set())

			# find distance of each point from centroids 
			# and assign the point to cluster with minimum distance 
			for i in range(len(X)):
				closestCentroid = np.argmin(self.distance(centroids,X[i]))
				indexToClusterMap[i]= closestCentroid
				currentClusters[closestCentroid].add(i)

			# compare previous and current clusters
			# if they are the same, solution is reached and hence break out
			clustersChanged = False
			if len(clusters) != 0:
				for idx in range(len(currentClusters)):
					if clusters[idx] != currentClusters[idx]:
						clustersChanged = True
						break

				# no change in clusters, hence break
				if clustersChanged == False:
					break

			# find mean of each cluster and update the centroid values
			for idx in range(len(currentClusters)):
				mean = np.mean([X[i] for i in currentClusters[idx]],axis=0) 
				centroids[idx] = mean

			# store current cluster into global clusters to compare in the next iteration
			clusters = currentClusters[:]
			
			# increment iteration count
			iterations += 1

		#return array with cluster id corresponding to each item in dataset
		self.cluster = [indexToClusterMap[index] for index in range(len(X))]
		return self.cluster


class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def train(self, X):
		#training logic here
		#input is array of features (no labels)

		# To keep track of elements in respective cluster
		# we use this to iterate over elements to set the new label when combining two clusters
		clusters = [[feature] for feature in range(len(X))]
		numClusters = len(clusters)

		# Stores cluster label for each row in the input X
		# creating a map for constant time lookup
		elementToClusterMap = {}
		for idx in range(len(X)):
			elementToClusterMap[idx] = idx

		# find distance between each pair of datapoints and store them along with datapoint indices
		distances = []
		for a in range(len(clusters)):
				for b in range(a+1,len(clusters)):
					distance = self.distance(X[a],X[b])
					distances.append((distance,(a,b)))
		# sorting distances in descending order
		distances.sort(reverse=True,key=lambda x: x[0])

		while numClusters != self.k:

			# pop the top of the list, this gets the pair of datapoints with least distance
			leastDistance =	distances.pop()

			# read the datapoints and their respective clusters into variable for clarity
			points = leastDistance[1]
			cluster1 = elementToClusterMap[points[0]]
			cluster2 = elementToClusterMap[points[1]]

			# if the pair already belongs to same cluster, check the next pair
			if cluster1 == cluster2:
				continue

			# merge clusters by copying smaller clusters datapoints into the larger cluster
			# here source is index of smaller cluster and destination is index of larger cluster
			if len(clusters[cluster1]) > len(clusters[cluster2]): 
				destination = cluster1
				source = cluster2
			else:
				destination = cluster2
				source = cluster1
			clusters[destination] += clusters[source]

			# set the cluster label for datapoints in the source cluster to destination clusters label
			for idx in clusters[source]:
				elementToClusterMap[idx] = destination

			# reducing number of clusters as we merged two clusters into one in this iteration
			numClusters = numClusters - 1

		#return array with cluster id corresponding to each item in dataset
		self.cluster = [elementToClusterMap[index] for index in range(len(X))]
		print(np.unique(self.cluster,return_counts=True))
		return self.cluster


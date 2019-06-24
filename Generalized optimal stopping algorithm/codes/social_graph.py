import networkx as nx 
import pickle
import numpy as np

path = "../pickles/"
save = "./graphs/"
AVG_RATING = 3.5
AVG_FRIENDSHIP_SCORE = 9.0
DISCOUNT_FACTOR = 1.1

def load(file):
	return pickle.load(open(file, 'rb'))

def addFriendshipEdges(user_to_tuples):
	nodes = list(user_to_tuples.keys())
	count1 = 0
	count2 = 0
	G = nx.Graph()
	G.add_nodes_from(nodes)
	for i in range(0, len(nodes)-1):
		for j in range(i, len(nodes)):
			#if not G.has_edge(nodes[i], nodes[j]):
			L1 = user_to_tuples[nodes[i]][0]
			L2 = user_to_tuples[nodes[j]][0]
			D1 = user_to_tuples[nodes[i]][1]
			D2 = user_to_tuples[nodes[j]][1]
			score = (len(L1&L2)+len(D1&D2))/len(L1|L2|D1|D2)
			if(score*100 > AVG_FRIENDSHIP_SCORE):
				count1+=1
			p=np.random.randint(0, 10)
			if(p<=4 and score*100 > AVG_FRIENDSHIP_SCORE):
				count2 += 1
				G.add_edge(nodes[i], nodes[j])
	print("Count1 : ", count1)
	print("Count2 : ", count2)
	return G
# user_to_movies = {user1 : [[movie1, rating], [movie2, rating]] , user2 : ... } 
# movies = {number: [name, genres, avg_rating], number:[name, ...] ..}

user_to_tuples = load("user_to_tuples.pickle")
G = addFriendshipEdges(user_to_tuples)
print(nx.info(G))
pickle.dump(G, open(save+"social_graph.pickle", "wb"))
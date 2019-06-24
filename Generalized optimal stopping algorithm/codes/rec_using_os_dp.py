import networkx as nx 
import pickle
import numpy as np
#from graph_creation import *

# GLOBALS
path = "../pickles/"
save = "./graphs/"
AVG_RATING = 3.5
AVG_FRIENDSHIP_SCORE = 10.0
PENALTY = 0.9

def load(file):
	return pickle.load(open(file, 'rb'))

def loadGraph(graphFilename, mappingFilename):
	G,info = pickle.load(open(graphFilename,"rb"))
	userToGraphNodeMapping, graphNodeToUser = pickle.load(open(mappingFilename,"rb"))
	return G,userToGraphNodeMapping,graphNodeToUser

# user_to_movies = {user1 : [[movie1, rating], [movie2, rating]] , user2 : ... } 
user_to_movies = load(path+"userMovies.pickle")
# movies = {number: [name, genres, avg_rating], number:[name, ...] ..}
movies = load(path+"movies.pickle")

# a) Create all 3 graph models

# 1. Erdos Renyi : random graph model
# 2. Watts Strogatz : small world model
# 3. Barbasi Albert : preferential attachment model

g1_user, g1_user_to_node, g1_node_to_user = loadGraph(path+"erdosRenyi.pickle", path+"mappingErdosRenyi.pickle")
g2_user, g2_user_to_node, g2_node_to_user = loadGraph(path+"wattsStrogatz.pickle", path+"mappingWattsStrogatz.pickle")
g3_user, g3_user_to_node, g3_node_to_user = loadGraph(path+"barabasiAlbert.pickle", path+"mappingbarabasiAlbert.pickle")

g1_user = nx.erdos_renyi_graph(len(g1_user.nodes()), 0.01)
g2_user = nx.watts_strogatz_graph(len(g2_user.nodes()), 6, 0.3)
g3_user = nx.barabasi_albert_graph(len(g3_user.nodes()), 4)
# b) Map user to nodes of the graph 
nx.relabel_nodes(g1_user, g1_node_to_user, copy=False)
nx.relabel_nodes(g2_user, g2_node_to_user, copy=False)
nx.relabel_nodes(g3_user, g3_node_to_user, copy=False)

#print("BEFORE FRIEND ATTACHMENT :- ")
#print(nx.info(g1_user), nx.info(g2_user), nx.info(g3_user))

# Print summaries
#print("Movies : ", len(movies), type(movies))
#print("User to movie mapping : ", len(user_to_movies), type(user_to_movies))
#print("Graph summaries :- ", nx.info(g1_user), "\n", nx.info(g2_user), "\n", nx.info(g3_user))

# c) Optimal stopping

'''
Idea : Give weights to each edge based on common liked and disliked movies
edge weight = (|L1 int L2| +|D1 int D2|- |L1 int D2| - |L2 int D1|) / (|L1 union L2 union D1 union D2|)
L and D are liked and disliked sets of movies for each user.

max possible weight = 1
min possible weight = -1 


Now, we pose the problem of finding most effective set of friends to recommend from as a search problem. We use optimal stopping to perform this search and get the set of friends, who maximize my reward.

Loss = ||num movies recommended - num movies out of reco seen and liked|-|num movies recommended - num movies out of reco seen and disliked||, that is movies with rating > 3.

We aim to decrease the loss. Show that with increase in reward, loss actually decreases.

'''

# 1. Create user to movie database
# {user:[(liked movies), (disliked movies)], user2: ..}.
# Assume movie is liked if rating > 3

user_to_tuples = {}

for user in user_to_movies:
	pos = []
	neg = []
	#print(user)
	for movie in user_to_movies[user]:
		if movie[1] > AVG_RATING :
			pos.append(movie[0])
		else:
			neg.append(movie[0])
	user_to_tuples[user] = [set(pos), set(neg)]

pickle.dump(user_to_tuples, open("user_to_tuples.pickle", "wb"))


def addFriendshipEdges(G, user_to_tuples):
	nodes = list(G.nodes())
	count1 = 0
	count2 = 0
	for i in range(0, len(nodes)-1):
		for j in range(i, len(nodes)):
			if not G.has_edge(nodes[i], nodes[j]):
				L1 = user_to_tuples[nodes[i]][0]
				L2 = user_to_tuples[nodes[j]][0]
				D1 = user_to_tuples[nodes[i]][1]
				D2 = user_to_tuples[nodes[j]][1]
				score = (len(L1&L2)+len(D1&D2))/len(L1|L2|D1|D2)
				if(score*100 > AVG_FRIENDSHIP_SCORE):
					count1+=1
				p=np.random.randint(0, 10)
				if(p<=1 and score*100 > AVG_FRIENDSHIP_SCORE):
					count2 += 1
					G.add_edge(nodes[i], nodes[j])
	print("Count1 : ", count1)
	print("Count2 : ", count2)
	return G

def addWeights(G, user_to_tuples):
	
	for edge in G.edges():
		L1, D1 = user_to_tuples[edge[0]]
		L2, D2 = user_to_tuples[edge[1]]
		weight = 100*(len(L1&L2) + len(D1&D2))/(len(L1|L2|D1|D2))
		#print(weight)
		G[edge[0]][edge[1]]['weight'] = weight

	return G

def getGraph(G, user_to_tuples, name):
	try:
		_g = pickle.load(open(save+name+".pickle", "rb"))
		print("here")
		return _g
	except:
		G = addFriendshipEdges(G, user_to_tuples)
		G = addWeights(G, user_to_tuples)
		pickle.dump(G, open(save+name+".pickle", "wb"))
		return G

def traverse_graph(G, node, score, visited, depth):
	final = set([node])
	#print(score)
	for n in G.neighbors(node):
		if n not in visited :
			newscore = (PENALTY**depth)*G[node][n]['weight']+score
			if newscore > 0 and newscore/(depth+1) > score:
				visited.append(n)
				ret = traverse_graph(G, n, (PENALTY**depth)*G[node][n]['weight']+score, visited, depth+1)
				final = final|ret
	return final

def optimalStopping(G, node):
	# Take a graph and a node, return a list of nodes as the optimal set of friends to recommend movies from.
	visited = []
	scores = []
	finalset = set()

	#q = queue.Queue()
	#print("Node ", node, " Neighbours : ", len(list(G.neighbors(node))))
	for n in G.neighbors(node): 
		#q.put([n])
		visited.append(n)
		scores.append(G[node][n]['weight'])
	v = len(visited)
	visited.append(node)
	for i in range(v):
		_friends = []
		_scores = []
		if(G[node][visited[i]]['weight'] >= sum(scores)/len(scores)):
			friends = traverse_graph(G, visited[i], scores[i], visited, 1)
			finalset = finalset|friends


	#print(sorted(visited))
	#print("finalset size : ", len(finalset))
	if(len(finalset) < 3):
		#print(finalset)
		finalset = finalset|set(list(G.neighbors(node)))
	return finalset
 
def recommend(friends, user_to_tuples, movies):
	movie_set = set()
	movie_list = []
	for friend in friends:
		movie_set = movie_set|user_to_tuples[friend][0]
	#print(len(movie_set))
	
	for m in movie_set:
		if movies[m][2] > AVG_RATING :
			movie_list.append((movies[m][2], movies[m][0], m))

	return sorted(movie_list, reverse=True)


def evaluate(recommended_movies, test_movies):
	#print("Actual : ", len(test_movies))
	#print("Recommended Movies: ", len(recommended_movies))
	hit = 0
	for movie in recommended_movies:
		if movie[2] in test_movies:
			hit += 1
	#print("Hits : ", hit)
	try:
		#print("PERFECT HITS : ", hit/len(test_movies))
		phit = hit/len(test_movies)
	except:
		phit = 0
	try:
		#print("ERROR : ", len(recommended_movies) - hit, "\n")
		error = (len(recommended_movies) - hit)/len(recommended_movies)
	except:
		error=0
	return phit, error


def utility(G, user_to_tuples, uid):

	friend_set = optimalStopping(G, uid)
	recommended_movies = recommend(friend_set, user_to_tuples, movies)
	#print("USER : ", uid)
	return evaluate(recommended_movies, user_to_tuples[uid][0])


uids = list(set(list(np.random.randint(1, 600, 100))))
g0_user = pickle.load(open("social_graph.pickle", "rb"))
g0_user = getGraph(g0_user, user_to_tuples, "social_graph")
graphs = [[g0_user, "social_graph"], [g1_user, "random_graph"], [g2_user, "small_world"],[ g3_user, "pref_attachment"]]

print("Penalty : ", PENALTY)
for g in graphs:
	G = getGraph(g[0], user_to_tuples, g[1])
	print("Summary for ", g[1], nx.info(G))
	print("Average clustering : ", nx.average_clustering(G))
	#print("Average shortest path : ", nx.average_shortest_path_length(G))
	print(g[1], " Summary : ", nx.info(G))
	MAE = 0
	PHIT = 0
	for uid in uids:
		HIT, E = utility(G, user_to_tuples, uid)
		PHIT += HIT
		MAE += E 
	print(g[1]+" perfect hit percent is : ", PHIT/len(uids), "\n")
	print(g[1]+" MAE : ", MAE/len(uids), "\n")

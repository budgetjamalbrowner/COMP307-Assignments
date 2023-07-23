import utility as utility
import loader as loader
import numpy as np
import math
import itertools

def main():

    # Paths to the data and solution files.
    vrp_file = "vrp-data/n32-k5.vrp"
    sol_file = "vrp-data/n32-k5.sol"

    # Loading the VRP data file.
    px, py, demand, capacity, depot = loader.load_data(vrp_file)

    # Displaying to console the distance and visualizing the optimal VRP solution.
    vrp_best_sol = loader.load_solution(sol_file)
    best_distance = utility.calculate_total_distance(vrp_best_sol, px, py, depot)
    print("Best VRP Distance:", round(best_distance,1))
    utility.visualise_solution(vrp_best_sol, px, py, depot, "Optimal Solution")

    # Executing and visualizing the nearest neighbour VRP heuristic.
    # Uncomment it to do your assignment!

    nnh_solution = nearest_neighbour_heuristic(px, py, demand, capacity, depot)
    nnh_distance = utility.calculate_total_distance(nnh_solution, px, py, depot)
    print("Nearest Neighbour VRP Heuristic Distance:", round(nnh_distance,1))
    utility.visualise_solution(nnh_solution, px, py, depot, "Nearest Neighbour Heuristic")

    # Executing and visualizing the saving VRP heuristic.
    # Uncomment it to do your assignment!
    
    sh_solution = savings_heuristic(px, py, demand, capacity, depot)
    sh_distance = utility.calculate_total_distance(sh_solution, px, py, depot)
    print("Saving VRP Heuristic Distance:", round(sh_distance,1))
    utility.visualise_solution(sh_solution, px, py, depot, "Savings Heuristic")


def nearest_neighbour_heuristic(px, py, demand, capacity, depot):
    """
    Algorithm for the nearest neighbour heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    # TODO - Implement the Nearest Neighbour Heuristic to generate VRP solutions.
    # calculate distance matrix
    # dist_matrix = utility.calculate_matrix(px, py, savings=False)
    distMatrix = np.zeros((len(px), len(px)))
    for i in range(len(px)):
        for j in range(len(px)):
            if i == j:
                continue
            distMatrix[i][j] =  utility.calculate_euclidean_distance(px, py, i, j)
    tours = []
    remainingNodes = list(range(len(px)))
    remainingNodes.remove(depot)
    while remainingNodes:
        currentNode = depot
        route = [depot]
        totalDemand = 0
        remainingCap = capacity
        while remainingCap > 0:
            minumumDist = math.inf
            nearestNeighbor = None
            for node in remainingNodes:
                if distMatrix[currentNode][node] == math.inf:
                    continue
                if distMatrix[currentNode][node] < minumumDist:
                    minumumDist = distMatrix[currentNode][node]
                    nearestNeighbor = node

            if nearestNeighbor is None:
                break
            if totalDemand + demand[nearestNeighbor] > capacity:
                remainingCopy = [
                    node
                    for node in remainingNodes
                    if demand[node] <= remainingCap
                ]
                remainingCopy.sort(key=lambda node: distMatrix[currentNode][node])
                nearestNeighborNew = next(
                    (
                        node
                        for node in remainingCopy
                        if totalDemand + demand[node] <= capacity
                    ),
                    None,
                )
                if nearestNeighborNew is not None:
                    nearestNeighbor = nearestNeighborNew
                else:
                    break
            totalDemand += demand[nearestNeighbor]
            route.append(nearestNeighbor)
            remainingCap -= demand[nearestNeighbor]
            currentNode = nearestNeighbor
            distMatrix[currentNode][depot] = math.inf 
            remainingNodes.remove(nearestNeighbor)

        route.append(depot)
        tours.append(route)
    return tours


def savings_heuristic(px, py, demand, capacity, depot):

    """
    Algorithm for Implementing the savings heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """
    # TODO - Implement the Saving Heuristic to generate VRP solutions.
    nodes = list(range(len(px)))
    nodes.remove(depot)
    tours = [[depot, n, depot] for n in nodes]

    savingsMatrix = np.zeros((len(px), len(px)))
    for i in range(len(px)):
        for j in range(len(px)):
            if i == j:
                continue
            savingsMatrix[i][j] = utility.calculate_euclidean_distance(px, py, depot, i) + utility.calculate_euclidean_distance(px, py, depot, j) 
            - utility.calculate_euclidean_distance(px, py, i, j)
    while True:
        merges = []
        for i, j in itertools.product(range(len(tours)), range(len(tours))):
            if i == j:
                continue
            totalDemand = sum(demand[node] for node in tours[i][1:-1]) + sum(demand[node] for node in tours[j][1:-1])
            if totalDemand > capacity:
                continue
            merges.append([tours[i], tours[j]])
        if not merges:
            break
        maxSavings = -math.inf
        maxMerge = None
        for merge in merges:
            savings = savingsMatrix[merge[0][-2]][merge[1][1]]
            if savings > maxSavings:
                maxSavings = savings
                maxMerge = merge
        tours.remove(maxMerge[0])
        tours.remove(maxMerge[1])
        tours.append(maxMerge[0][:-1] + maxMerge[1][1:])
    return tours

if __name__ == '__main__':
    main()

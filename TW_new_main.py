import matplotlib.pyplot as plt
from functools import total_ordering

import snap
import networkx as nx
import pydot
import json
import itertools
import time
import numpy as np
import os
import random
import heapq

@total_ordering
class CustomObjHeap(list):
    def __eq__(self, _other):
        return self[3] == _other[3]


    def __lt__(self, _other):
        return self[3] < _other[3]

@total_ordering
class CustomObjHeap1(list):
    def __eq__(self, _other):
        return self[3] == _other[3]


    def __lt__(self, _other):
        return self[3] < _other[3]

def precision_graph_creator(data_path, precision_type):
    """

    :param data_path:
    :param precision_type: whether calculating the precision by score,
                            or number of common nodes
    :return:
    """
    with open(data_path) as f:
        info = json.load(f)
    title = ""
    _dict = dict()
    name_set = set()
    # name_list = set()
    # for _name in info.keys():
    #     name_list.add(_name['algo'])
    # creating dict with all the algo name
    # _dict[name] = [k_arr, _score_arr]
    lamda = None
    func_type = None
    _num_edges = None
    _num_nodes = None
    for name in info.values():
        name_set.add(name['algo'])
        _dict[name['algo']] = [[],[]]
        lamda = name['lamda']
        func_type = name['value_function']
        _num_edges = name['edges_in_graph']
        _num_nodes = name['nodes_in_graph']

    for key, value in info.items():
        algo_name = value['algo']
        k = key.split('=')[1]
        R = value['R']
        score = value['score']
        _dict[algo_name][0].append(k)
        if precision_type == SCORE:
            _dict[algo_name][1].append(score)
        else:
            _dict[algo_name][1].append(R)
    if precision_type == SCORE:
        title = " k over precision (comparing score values)"
        optimal_score = _dict[BRUTE_FORCE][1]
        for algo_name in list(name_set):
            if func_type == 1:
                _dict[algo_name][1] = np.divide(optimal_score,
                                                _dict[algo_name][1])
            else:
                _dict[algo_name][1] = np.divide(_dict[algo_name][1], optimal_score)
    else:
        title = "k over precision\n (comparing the percentage of node which corresponding to the optimal set)"
        optimal_set = _dict[BRUTE_FORCE][1]
        k_list = _dict[BRUTE_FORCE][0]
        for algo_name in list(name_set):
            caculated_score = list()
            for idx, (r1, r2) in enumerate(zip(_dict[algo_name][1], optimal_set)):
                caculated_score.append(len(set(r1).intersection(set(r2))) / (int(k_list[idx]) + 0.0))
            _dict[algo_name][1] = caculated_score

    plt.xlabel("k")
    plt.ylabel("precision")
    legends = []
    text = " lambda = " + str(lamda) + \
           "\n num of nodes : " + _num_nodes +\
           "\n num of edges : " + _num_edges +\
           "\n value function type : " + str(func_type)

    for func_name, value in _dict.items():
        legends.append(func_name)
        plt.plot(value[0], value[1], marker='o')
        plt.legend(legends)
        plt.title(title)
        plt.subplots_adjust(left=0.3)
        plt.gcf().text(0.001, 0.7, text, fontsize=10)
    plt.show()

    # info_arr[func.__name__][0][idx] = num_of_nodes_in_graph
    # info_arr[func.__name__][1][idx] = run_time

    # graph_creator(info_arr, "graph size over precision",
    #                               "\nLamda : " + info["lamda"], "Output/Test5" \
    #               + "/graph_size_over_runtime", False, 1, "Graph Size (Nodes)", "Runtime")

def plot_graph(data_x, data_y, save):
    # plotting the points
    plt.plot(data_x, data_y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)

    # setting x and y axis range
    plt.ylim(1, 8)
    plt.xlim(1, 8)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('Some cool customizations!')

    # function to show the plot
    plt.show()
    if save:
        plt.savefig('name.png')

def get_random_undirected_graph(nodes_num, edges_num):
    return snap.GenRndGnm(snap.PUNGraph, nodes_num, edges_num)

def quality_function_random(Vg):
    return random.uniform(0, 1)

def optimize_intersection(first, second):
    # optimization, small.intersection(large) is faster then large.intersection(small)
    smaller, larger = (first, second) if len(first) < len(second) else (
    second, first)
    return set(smaller).intersection(set(larger))

# representative functions: --------------------------------------------------

# later on I will refactor the name according to the conventions (PEP8)
def func_rep_1_snap(R, graph):
    """
    :param r: group of k node to evaluate, |r| = k
    :param graph: graph
    :return: score of representative for the group of k nodes
    """
    _score = 0.0
    for r in R:
        sim_r_intersection_R = 1.0 # start from 1 as the nodes itself is in sim(node)
        node_r = graph.GetNI(r)
        # .GetId() to get the node id
        sim_r = 1
        for neighbor in  node_r.GetOutEdges():
            sim_r +=1
            if neighbor in R:
                sim_r_intersection_R += 1.0
        # probably a bit slower option _score += sim_r_intersection_R / (node_r.GetOutDeg() + 1)
        _score += sim_r_intersection_R / sim_r

    return _score


def func_rep_1_networkx(R, graph):
    """
    sum : for r in R : |sim(r) intersection R| / |sim(r)|
    :param R: set of k node to evaluate, |r| = k
    :param graph: graph
    :return: score for the group of k nodes
    """
    _score = 0.0
    # converting R items from int to string, as graph node name is in string
    # R = set(str(r) for r in R)

    for r in R:
        r_neighbors = set(graph.neighbors(r)).union({r})
        # r_neighbors = set(graph.neighbors(r))
        # r_neighbors.add(r) less efficient
        # _score += (len(R.intersection(r_neighbors)) + 1.0) / (len(r_neighbors) + 1)
        _score += np.minimum((len(R.intersection(r_neighbors)) + 1.0 / (len(r_neighbors))), 1)

    return _score

def func_rep_2_networkx(R, graph):
    """
    min var(|sim(u) intersection R| / |sim(u)|) for u in Vg
    return representative score for group R
    :param R: set of k nodes
    :param graph: full graph
    :return: score of the group
    """
    R = set(str(r) for r in R)
    r_union_neighbors = set()
    _score_list = list()

    # first getting all the relevant nodes
    for r in R:
        r_union_neighbors.add(r)
        r_union_neighbors.update(graph.neighbors(r))

    # second, getting calculating the score of each node (can make it a bit more efficient using dict)
    for node in r_union_neighbors:
        sim_node = set(graph.neighbors(node))
        sim_node.add(node)
        node_score = len(optimize_intersection(sim_node, R)) / len(sim_node)
        _score_list.append(node_score)

    # thirdly, calculating avg, and reduce it from each node score
    avg = sum(_score_list) / (num_of_nodes_in_graph + 0.0)
    final_score = 0.0
    for _score in _score_list:
        final_score += abs(_score - avg)

    final_score += (num_of_nodes_in_graph - len(r_union_neighbors)) * avg

    return final_score


def func_rep_2_networkx_fixed_ver(R, graph):
    """
    return var( |sim(u) intersection R| / |sim(u)| for u in Vg)
    :param R:
    :param graph:
    :return: var( |sim(u) intersection R| / |sim(u)| for u in Vg)
    """
    _list = []
    # R = set(R)
    for _node in graph.nodes():
        sim_node = set(graph.neighbors(_node))
        sim_node.add(_node)
        num = len(R.intersection(sim_node)) / len(sim_node)
        _list.append(num)
    return np.var(_list)

def func_rep_2_networkx_dict(R, graph):
    R = set(str(r) for r in R)
    r_union_neighbors = set()
    _score_list = list()

    neighbors_dict = dict()
    # first getting all the relevant nodes
    for r in R:
        r_union_neighbors.add(r)
        neighbors = set(graph.neighbors(r))
        neighbors_dict[r] = neighbors
        r_union_neighbors.update(neighbors)

    # second, getting calculating the score of each node (can make it a bit more efficient using dict)
    for node in r_union_neighbors:
        sim_node = neighbors_dict.get(node)
        if sim_node is None:
            sim_node = set(graph.neighbors(node))
        sim_node.add(node)
        node_score = len(optimize_intersection(sim_node, R)) / len(sim_node)
        _score_list.append(node_score)

    # thirdly, calculating avg, and reduce it from each node score
    avg = sum(_score_list) / (num_of_nodes_in_graph + 0.0)
    final_score = 0.0
    for _score in _score_list:
        final_score += abs(_score - avg)

    final_score += (num_of_nodes_in_graph - len(r_union_neighbors)) * avg

    return final_score

def func_rep_3_union_neighbors_networkx(R, graph):
    """
    we want to maximize the number of neighbors
    max |Union(sim(r))| for r in R
    :param R: group of k nodes to evaluate their score
    :param graph: graph
    :return: score for the group R
    """
    final_set = set(R)
    for r in R:
        final_set.update(graph.neighbors(r))
    return len(final_set)
    # final_set = list(R)
    # for r in R:
    #     for _neighbor in graph.neighbors(r):
    #         final_set.append(_neighbor)
    # return len(set(final_set))


# value functions: --------------------------------------------------

def f_val_general_helper(R, graph, lamda, quality_func, rep_func, k):
    """
    :param R: set of k nodes to evaluate
    :param graph: full graph
    :param lamda: double that represent how much of importance
        we want give to representative (between 0 to 1)
    :param quality_func: the quality function
    :param rep_func: the representative function
    :param k: the group size we want (for example, we want the 10 best nodes k=10)
    :return: score for that set
    """

    sum_of_quality = sum(quality_func(r) for r in R)

    part_1 = ((1.0 - lamda) / k) * sum_of_quality
    if rep_func == func_rep_3_union_neighbors_networkx:
        part_2 = (lamda / num_of_nodes_in_graph) * rep_func(R, graph)
        return part_1 + part_2
    elif rep_func == func_rep_1_networkx or rep_func == func_rep_2_networkx or\
          rep_func == func_rep_2_networkx_dict or rep_func == func_rep_2_networkx_fixed_ver:
        part_2 = 1 - (lamda / num_of_nodes_in_graph) * rep_func(R, graph)
        return part_1 + part_2
    else:
        part_2 = (lamda / num_of_nodes_in_graph) * rep_func(R, graph)

    #     return part_1 + part_2
    # # might be a bit faster (or slower ofc, but looping once instead of 2):
    # sum_of_rep = 0.0
    # sum_of_quality = 0.0
    # for r in R:
    #     sum_of_rep += sum_of_rep(r)
    #     sum_of_quality += quality_func(r)

    return part_1 + part_2

def func_val(type ,R, graph, lamda, k):
    quality_func =  lambda x: 0
    if type == 1:
        return f_val_general_helper(R, graph, lamda, quality_func,
                                    func_rep_1_networkx, k)
    elif type == 2:
        return f_val_general_helper(R, graph, lamda, quality_func,
                                    func_rep_2_networkx_fixed_ver, k)
    #func_rep_2_networkx_dict

    elif type == 3:
        return f_val_general_helper(R, graph, lamda, quality_func,
                                    func_rep_3_union_neighbors_networkx, k)



def draw_undirected_graph(graph, file_name, title):
    snap.DrawGViz(graph, snap.gvlNeato, file_name, title,
                  True)



def arg_min_rep(graph, Vg, func_val_type):
    func_rep = None
    if func_val_type == 1:
        func_rep = func_rep_1_networkx
    elif func_val_type == 2:
        func_rep = func_rep_2_networkx
    first = True
    _score = 0.0
    optimal_node = None
    for _node in Vg:
        if first:
            first = False
            _score = func_rep(set(_node), graph)
            optimal_node = _node
        else:
            tmp_score = func_rep(set(_node), graph)
            if tmp_score > _score:
                _score = tmp_score
                optimal_node = _node
    return [_score, optimal_node]


def arg_max_quality(Vg, quality_func):
    _score = 0.0
    first = True
    optimal_node = None
    for _node in Vg:
        if first:
            first = False
            optimal_node = _node
            _score = quality_func(_node)
        else:
            tmp_score = quality_func(_node)
            if tmp_score > _score:
                _score = tmp_score
                optimal_node = _node
    return [_score, optimal_node]

def node_ordered_by_quality(Vg, quality_func):
    sorted_list = sorted(Vg, key=quality_func)
    return sorted_list

def greedy_then_swap_ver_2(func_val_type, graph, k, lamda):
    # is slower compare to the other version
    # running greedy algo
    R, _score, _running_time_greedy, Vg = algo_3_greedy(func_val_type, graph, k, lamda, True)

    optimal_set = set()
    start_time = time.time()
    optimal_set, _score, _running_time_swap = algo_2_swap(func_val_type, graph, k, lamda, (R, _score, Vg))
    return [optimal_set, _score, _running_time_swap + _running_time_greedy]
    # while len(Vg) > 0 :
    #     # pop the one with the optimal(minimal) rep takes too long and dont
    #     # give benefits(maybe less swaps, but finding the vertex with minimal
    #     # rep take more time than it)
    #     # _, _node = arg_min_rep(graph,Vg, func_val_type)
    #     _node = Vg.pop()
    #     # Vg.remove(_node)
    #     for r in R:
    #         new_set = set(R.difference(r)).union(_node)
    #         new_score = func_val(func_val_type, new_set, graph, lamda, k)
    #         if _score < new_score:
    #             _score = new_score
    #             optimal_set = new_set
    #     if len(optimal_set) > 0:
    #         R = optimal_set
    #
    # running_time = time.time() - start_time  # in seconds
    # return [frozenset(R), _score, running_time + _running_time_greedy]

def algo_4_greedy_then_swap_ver_2(func_val_type, graph, k, lamda):
    # running greedy algo
    R, _score, _running_time_greedy, Vg = algo_3_greedy(func_val_type, graph,
                                                        k, lamda, True)
    print("Greedy time is : ", _running_time_greedy)
    start_time = time.time()
    optimal_set = set()
    # part 2
    while len(Vg) > 0 :
        _node = Vg.pop()
        # Vg.remove(_node)
        for r in R:
            new_set = R.difference(r).union(_node)
            # new_set.add(_node) this line literally took me half day to debug...
            # with the above line, it would take almost half hour to run instead of few seconds.
            # new_set = R.copy()
            # new_set.add(_node)
            # new_set.remove(r)
            new_score = func_val(func_val_type, new_set, graph, lamda, k)
            if _score < new_score:
                _score = new_score
                optimal_set = new_set
        if len(optimal_set) > 0:
            R = optimal_set

    running_time = time.time() - start_time  # in seconds
    print("Swap time is : ", running_time)
    return [frozenset(R), _score, running_time + (_running_time_greedy)]

# algo_4_greedy_then_swap_ver_2
def algo_4_greedy_then_swap(func_val_type, graph, k, lamda):
    # running greedy algo
    R, _score, _running_time_greedy, Vg = algo_3_greedy(func_val_type, graph,
                                                        k, lamda, True)
    # print("Greedy time : ", _running_time_greedy)
    final_r, final_score, swap_run_tine = algo_2_swap(func_val_type, graph,k,lamda,(R, _score, Vg))
    # print("Swap time : ", swap_run_tine)

    return [final_r, final_score, swap_run_tine+_running_time_greedy]

def algo_3_greedy_old(func_val_type, graph, k, lamda, return_remaining_vertices = False):
    """
    trying to find the best k size group using greedy approach
    assumption : running time we be at least tenth of second (otherwise might not be accurate)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :param return_graph - whether to return the remaining vertices
    :return: set of k nodes, their score and running time of the algo
    """
    start_time = time.time()
    final_set = set()
    _score = 0.0
    vertices_graph = set(graph.nodes)
    for i in range(k):
        curr_best_node = "default"
        first = True
        for _node in vertices_graph:
            if first:
                first = False
                curr_best_node = _node
                _score = func_val(func_val_type, final_set.union({curr_best_node}),
                                  graph, lamda, k)
                continue

            _score_with_node = func_val(func_val_type, final_set.union({_node}),
                                        graph, lamda, k)

            if _score_with_node > _score:
                curr_best_node = _node
                _score = _score_with_node
        if curr_best_node != "default":
            final_set.add(curr_best_node)
            vertices_graph.remove(curr_best_node)

    _score = func_val(func_val_type, final_set, graph, lamda, k)
    running_time = time.time() - start_time # in seconds
    if return_remaining_vertices:
        return [frozenset(final_set), _score, running_time, vertices_graph]
    return [frozenset(final_set), _score, running_time]



def algo_3_greedy(func_val_type, graph, k, lamda, return_remaining_vertices = False):
    """
    trying to find the best k size group using greedy approach
    assumption : running time we be at least tenth of second (otherwise might not be accurate)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :param return_graph - whether to return the remaining vertices
    :return: set of k nodes, their score and running time of the algo
    """
    start_time = time.time()
    final_set = set()
    _score = 0.0
    vertices_graph = list(graph.nodes)
    for i in range(k):
        curr_best_node = "default"
        first = True
        for _node in vertices_graph:
            if first:
                first = False
                curr_best_node = _node
                _score = func_val(func_val_type, final_set.union({curr_best_node}),
                                  graph, lamda, k)
                continue

            _score_with_node = func_val(func_val_type, final_set.union({_node}),
                                        graph, lamda, k)

            if _score_with_node > _score:
                curr_best_node = _node
                _score = _score_with_node
        if curr_best_node != "default":
            final_set.add(curr_best_node)
            vertices_graph.remove(curr_best_node)

    _score = func_val(func_val_type, final_set, graph, lamda, k)
    running_time = time.time() - start_time # in seconds
    if return_remaining_vertices:
        return [frozenset(final_set), _score, running_time, set(vertices_graph)]
    return [frozenset(final_set), _score, running_time]

# --------------------------------------------------------------------------

def algo_2_swap_for_f3_tw(func_val_type, graph, k, lamda, predefined_k_highest_quality=None):
    Vg = set(graph.nodes)
    R_and_neighbors = list()
    # part 1, k highest quality node
    R = set(node_ordered_by_quality(Vg, quality_func_node_name)[-k:])
    # _k_highest_quality = numpy_k_highest_quality(Vg, k, quality_func_node_name)
    for node in R:
        Vg.remove(node)
        R_and_neighbors.add(node)
        R_and_neighbors = [*R_and_neighbors, *graph.neighbors(node)]
    R_and_neighbors = set(R_and_neighbors)

    _heap = []
    # first creating max - heap with [neighbors, updated_index, score_contribution]
    for node in Vg:
        node_neighbors = set(graph.neighbors(node)) # maybe change name to sim later on.
        node_neighbors.add(node)
        index = 0
        score_contribution = ((1-lamda)* quality_func_node_name(node) / k) +\
                             (lamda * len(node_neighbors - R_and_neighbors)) / num_of_nodes_in_graph
        # the heap library that I used implemented min-heap, without formal (not private) method to convert to max-heap,
        # therefore, I will convert the score from positive to negative (to get max-heap)
        _heap.append(CustomObjHeap([node, node_neighbors, index, -1*score_contribution]))
    heapq.heapify(_heap)

    # n_index = len(R)
    # while len(Vg) > 0:
    #     [node, node_neighbors, index, score_contribution] = heapq.heappop(_heap)
    #     score_contribution = -1*score_contribution
    #     if index >= n_index:
    #         R.append(node)
    #         score += score_contribution
    #     else:
    #         neighbors_to_remove = list()
    #         # might be a be faster with cumulative histogram of neighbors
    #         for i in range(index, len(R)):
    #             _node = R[i]
    #             neighbors_to_remove.append(_node)
    #             neighbors_to_remove = [*neighbors_to_remove, *graph.neighbors(_node)]
    #         # the number of neighbors which not in R (including the node itself)
    #         new_neighbors_contribution = node_neighbors.difference(set(neighbors_to_remove))
    #         if node_neighbors != new_neighbors_contribution:
    #             index = len(R)
    #             score_contribution = ((1-lamda)* quality_func_node_name(node) / k) +\
    #                                  score_contribution -\
    #                                  ((lamda * (len(node_neighbors) - len(new_neighbors_contribution))) / num_of_nodes_in_graph)
    #
    #             node_neighbors = new_neighbors_contribution
    #             heapq.heappush(_heap, CustomObjHeap([node, node_neighbors, index, -1*score_contribution]))
    #             # _heap.append(CustomObjHeap([node, node_neighbors, index, score_contribution]))
    #         else:
    #             R.append(node)
    #             score += score_contribution



    # R = set(R)
    _score = func_val(func_val_type, R, graph, lamda, k)


def algo_3_greedy_for_f3_new_tw(tmp_func, graph, k, lamda, return_remaining_vertices = False):
    """
    trying to find the best k size group using greedy approach
    assumption : running time will be at least tenth of second (otherwise might not be accurate)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :param return_graph - whether to return the remaining vertices
    :return: set of k nodes, their score and running time of the algo
    :note:  because the heap library (heapq) implemented min-heap and I needed
     max-heap, I made the score negative when pushing to the heap
     another possibility would be to change the __lt__ of the custom object
    that I pushed to the heap, but I at the end didn't dp so because I need
     min-heap in different place that use the same object as well
    """
    start_time = time.time()
    _heap = []
    # first creating max - heap with [neighbors, updated_index, score_contribution]
    for node in graph.nodes:
        node_neighbors = set(graph.neighbors(node)) # maybe change name to sim later on.
        node_neighbors.add(node)
        index = 0
        score_contribution = ((1-lamda)* quality_func_node_name(node) / k) + (lamda * len(node_neighbors)) / num_of_nodes_in_graph
        heapq.heappush(_heap, CustomObjHeap([node, node_neighbors, index, -1*score_contribution]))
    heapq.heapify(_heap)
    score = 0
    R = list()
    while len(R) < k:
        [node, node_neighbors, index, score_contribution] = heapq.heappop(_heap)
        score_contribution = -1*score_contribution
        if index >= len(R):
            R.append(node)
            score += score_contribution
        else:
            neighbors_to_remove = list()
            # might be a be faster with cumulative histogram of neighbors
            for i in range(index, len(R)):
                _node = R[i]
                neighbors_to_remove.append(_node)
                neighbors_to_remove = [*neighbors_to_remove, *graph.neighbors(_node)]
            # the number of neighbors which not in R (including the node itself)
            new_neighbors_contribution = node_neighbors.difference(set(neighbors_to_remove))
            if node_neighbors != new_neighbors_contribution:
                index = len(R)
                score_contribution = ((1-lamda)* quality_func_node_name(node) / k) +\
                                     score_contribution -\
                                     ((lamda * (len(node_neighbors) - len(new_neighbors_contribution))) / num_of_nodes_in_graph)

                node_neighbors = new_neighbors_contribution
                heapq.heappush(_heap, CustomObjHeap([node, node_neighbors, index, -1*score_contribution]))
                # _heap.append(CustomObjHeap([node, node_neighbors, index, score_contribution]))
            else:
                R.append(node)
                score += score_contribution

    running_time = time.time() - start_time  # in seconds
    return [frozenset(R), score, running_time]


# -------------------------------------------------------------------------
def algo_3_greedy_for_f1_new_tw(tmp_func, graph, k, lamda, return_remaining_vertices = False):
    """
    trying to find the best k size group using greedy approach
    assumption : running time will be at least tenth of second (otherwise might not be accurate)
    Here the lower the score the better the result (for certain k)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :param return_graph - whether to return the remaining vertices
    :return: set of k nodes, their score and running time of the algo

    """
    start_time = time.time()
    _heap = []
    # first creating max - heap with [neighbors, updated_index, score_contribution]
    for node in graph.nodes:
        node_neighbors = set(graph.neighbors(node)) # maybe change name to sim later on.
        node_neighbors.add(node)
        index = 0
        score_contribution = ((1 - lamda) * (1 - quality_func_node_name(node)) / k) +\
                             lamda * 1 / (len(node_neighbors) * num_of_nodes_in_graph)
        heapq.heappush(_heap,
                       CustomObjHeap1([node, node_neighbors, index,
                                       score_contribution, len(node_neighbors)]))
    heapq.heapify(_heap)
    score = 0
    R = list()
    while len(R) < k:
        [node, node_neighbors, index, score_contribution, len_sim] = heapq.heappop(_heap)
        if index >= len(R):
            R.append(node)
            score += score_contribution
        else:
            neighbors_to_remove = list()
            # might be a be faster with cumulative histogram of neighbors
            for i in range(index, len(R)):
                _node = R[i]
                neighbors_to_remove.append(_node)
                neighbors_to_remove = [*neighbors_to_remove, *graph.neighbors(_node)]
            # the number of neighbors which not in R (including the node itself)
            new_neighbors_contribution = node_neighbors.intersection(set(neighbors_to_remove))
            if len(new_neighbors_contribution) != 0:
                index = len(R)
                score_contribution = ((1-lamda)* (1 - quality_func_node_name(node) / k)) +\
                                     score_contribution +\
                                     ((lamda * len(new_neighbors_contribution)) / (len_sim*num_of_nodes_in_graph))

                node_neighbors = node_neighbors - new_neighbors_contribution
                heapq.heappush(_heap, CustomObjHeap1([node, node_neighbors, index, score_contribution, len_sim]))
            else:
                R.append(node)
                score += score_contribution

    running_time = time.time() - start_time  # in seconds
    return [frozenset(R), score, running_time]

# -----------------------------------------------------------------------------

def algo_3_greedy_for_f2_new_tw(tmp_func, graph, k, lamda, return_remaining_vertices = False):
    """
    trying to find the best k size group using greedy approach
    assumption : running time will be at least tenth of second (otherwise might not be accurate)
    Here the lower the score the better the result (for certain k)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :param return_graph - whether to return the remaining vertices
    :return: set of k nodes, their score and running time of the algo

    """
    start_time = time.time()
    _heap = []
    # first creating max - heap with [neighbors, updated_index, score_contribution]
    for node in graph.nodes:
        node_neighbors = set(graph.neighbors(node)) # maybe change name to sim later on.
        node_neighbors.add(node)
        index = 0
        # score_contribution =
        # heapq.heappush(_heap,
        #                CustomObjHeap1([node, node_neighbors, index,
        #                                score_contribution, len(node_neighbors)]))
    heapq.heapify(_heap)
    score = 0
    R = list()
    while len(R) < k:
        [node, node_neighbors, index, score_contribution, len_sim] = heapq.heappop(_heap)
        if index >= len(R):
            R.append(node)
            score += score_contribution
        else:
            neighbors_to_remove = list()
            # might be a be faster with cumulative histogram of neighbors
            for i in range(index, len(R)):
                _node = R[i]
                neighbors_to_remove.append(_node)
                neighbors_to_remove = [*neighbors_to_remove, *graph.neighbors(_node)]
            # the number of neighbors which not in R (including the node itself)
            new_neighbors_contribution = node_neighbors.intersection(set(neighbors_to_remove))
            if len(new_neighbors_contribution) != 0:
                index = len(R)
                score_contribution = ((1-lamda)* (1 - quality_func_node_name(node) / k)) +\
                                     score_contribution +\
                                     ((lamda * len(new_neighbors_contribution)) / (len_sim*num_of_nodes_in_graph))

                node_neighbors = node_neighbors - new_neighbors_contribution
                heapq.heappush(_heap, CustomObjHeap1([node, node_neighbors, index, score_contribution, len_sim]))
            else:
                R.append(node)
                score += score_contribution

    running_time = time.time() - start_time  # in seconds
    return [frozenset(R), score, running_time]



def quality_func_node_name(v):
    """
    :param v: string that represnt the node "int"
    :return: quality of that node (which corresponding to the node name)
    """
    return float(v)/num_of_nodes_in_graph


def numpy_k_highest_quality(Vg, k, quality_func):
    """
    Running time O(n) - using numpy.partition
    :param Vg: nodes in graph
    :param k: set size to return
    :return: return k highest quality node
    """
    # TODO here
    pass


def algo_2_swap(func_val_type, graph, k, lamda, predefined_k_highest_quality=None):
    """
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :return:
    """
    start_time = time.time()
    if predefined_k_highest_quality is None:
        Vg = set(graph.nodes)

        # part 1, k highest quality node
        R = set(node_ordered_by_quality(Vg, quality_func_node_name)[-k:])
        # _k_highest_quality = numpy_k_highest_quality(Vg, k, quality_func_node_name)
        for node in R:
            Vg.remove(node)

        # R = set(R)
        _score = func_val(func_val_type, R, graph, lamda, k)

    else:
        R, _score, Vg = predefined_k_highest_quality

    # part 2
    while len(Vg) > 0 :
        # print(len(Vg))
        # pop the one with the optimal(minimal) rep takes too long and dont
        # give benefits(maybe less swaps, but finding the vertex with minimal
        # rep take more time than it)
        # _, _node = arg_min_rep(graph,Vg, func_val_type)
        _node = Vg.pop()
        # Vg.remove(_node)
        optimal_set = None
        for r in R:
            new_set = R.difference({r}).union({_node})
            # new_set.add(_node) this line literally took me half day to debug...
            # with the above line, it would take almost half hour to run instead of few seconds.
            # new_set = R.copy()
            # new_set.add(_node)
            # new_set.remove(r)
            new_score = func_val(func_val_type, new_set, graph, lamda, k)
            if _score < new_score:
                _score = new_score
                optimal_set = new_set
        if optimal_set is not None:
            R = optimal_set

    running_time = time.time() - start_time  # in seconds

    return [frozenset(R), _score, running_time]

def algo_2_swap_v2(func_val_type, graph, k, lamda, predefined_k_highest_quality=None):
    """
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :return:
    """
    start_time = time.time()
    if predefined_k_highest_quality is None:
        Vg = set(graph.nodes)

        # part 1, k highest quality node
        R = set(node_ordered_by_quality(Vg, quality_func_node_name)[-k:])
        # _k_highest_quality = numpy_k_highest_quality(Vg, k, quality_func_node_name)
        for node in R:
            Vg.remove(node)

        # R = set(R)
        _score = func_val(func_val_type, R, graph, lamda, k)

    else:
        R, _score, Vg = predefined_k_highest_quality

    # part 2
    while len(Vg) > 0 :
        # print(len(Vg))
        # pop the one with the optimal(minimal) rep takes too long and dont
        # give benefits(maybe less swaps, but finding the vertex with minimal
        # rep take more time than it)
        # _, _node = arg_min_rep(graph,Vg, func_val_type)
        _node = Vg.pop()
        # Vg.remove(_node)
        optimal_set = None
        for r in R:
            new_set = R.difference({r})
            new_set.add(_node)
            # new_set.add(_node) this line literally took me half day to debug...
            # with the above line, it would take almost half hour to run instead of few seconds.
            # new_set = R.copy()
            # new_set.add(_node)
            # new_set.remove(r)
            new_score = func_val(func_val_type, new_set, graph, lamda, k)
            if _score < new_score:
                _score = new_score
                optimal_set = new_set
        if optimal_set is not None:
            R = optimal_set

    running_time = time.time() - start_time  # in seconds

    return [frozenset(R), _score, running_time]




def swap_same_as_pseudo(func_val_type, graph, k, lamda):
    start_time = time.time()

    Vg = set(graph.nodes)

    # part 1, k highest quality node
    ordered_by_quality = node_ordered_by_quality(Vg, quality_func_node_name)
    Vg, R = ordered_by_quality[:-k], ordered_by_quality[-k:]

    _score = func_val(func_val_type, R, graph, lamda, k)
    # R = set(R)
    # part 2
    for _node in Vg[::-1]:
        optimal_set = None
        for r in R:
            new_set = R.difference(r).union(_node)
            new_score = func_val(func_val_type, new_set, graph, lamda, k)
            if _score < new_score:
                _score = new_score
                optimal_set = new_set
        if optimal_set is not None:
            R = optimal_set

    running_time = time.time() - start_time  # in seconds

    return [frozenset(R), _score, running_time]

def algo_1_brute_force(func_val_type, graph, k, lamda):
    """
    finding the optimal set of node of size k using brute force approach
    assumption : running time we be at least tenth of second (otherwise might not be accurate)
    :param func_val_type: int that specify the type of val function we want to use
    :param graph: graph
    :param k: the size of the group we want to return
    :param lamda: trade-off variable, between quality and representativity
    :return: set of k nodes that maximize func_val and their score and
            running time of the algo
    """
    start_time = time.time()
    _score = 0.0
    optimal_set = set()
    first = True
    for R in itertools.combinations(set(graph.nodes), k):
        _score_for_R = func_val(func_val_type, set(R), graph, lamda, k)
        if first:
            first = False
            _score = _score_for_R
            optimal_set = R
            continue
        if _score_for_R > _score:
            _score = _score_for_R
            optimal_set = R

    running_time = time.time() - start_time # in seconds
    return [optimal_set, _score, running_time]

def graph_creator(info, title, text, location_to_save_graph, show,
                  i, x_label, y_label):

    plt.close()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    legends = []
    for func_name, value in info.items():
        legends.append(func_name)
        plt.plot(value[0], value[i], marker='o')
    plt.legend(legends)
    plt.title(title)
    plt.subplots_adjust(left=0.3)
    plt.gcf().text(0.001, 0.7, text, fontsize=10)
    if show:
        plt.show()
    else:
        plt.savefig(location_to_save_graph)

def graph_size_over_run_time(list_of_input_locations, k, path_to_save_data, function_val_type, lamda, save_fig):
    """

    :param list_of_input_locations: list of graphs location
    :param k: size of group we want to find
    :param path_to_save_data:
    :param function_val_type:
    :param lamda:
    :param save_fig : True = save it, False = show it
    :return:
    """
    json_name = path_to_save_data + "/info.json"
    if not os.path.exists(path_to_save_data):
        os.mkdir(path_to_save_data)

    info = {}
    info_arr = dict()
    for func in algo_func:
        info_arr[func.__name__] = np.zeros((2, len(list_of_input_locations)))

    for idx, input_location in enumerate(list_of_input_locations):
        G = nx.nx_pydot.read_dot(input_location)
        print("starting for graph of size : " + num_of_nodes_in_graph + " Nodes")

        for func in algo_func:
            _set_of_k_nodes, _score, run_time = func(function_val_type, G,
                                                     k, lamda)
            info[func.__name__ + "#G_size=" + num_of_nodes_in_graph] = {
                'input_data_source': input_location, 'algo': func.__name__,
                'run_time': run_time,
                'score': _score, 'R': [k for k in _set_of_k_nodes],
                'lamda': lamda,
                'value_function': function_val_type}
            info_arr[func.__name__][0][idx] = num_of_nodes_in_graph
            info_arr[func.__name__][1][idx] = run_time

    graph_creator(info_arr, "graph size over runtime",
                                  "\nLamda : " + str(lamda), path_to_save_data \
                  + "/graph_size_over_runtime", save_fig, 1, "Graph Size (Nodes)", "Runtime")

    # saving all the info into json
    with open(json_name, 'w') as f:
        json.dump(info, f)

def k_over_score_and_k_over_runtime_calculator(path_to_save_data, function_val_type, K_list, lamda, save_fig):
    """
    :param path_to_save_data: path to save the figures and info
    :param function_val_type: function that we want to maximize
    :param K_list: list of K to evaluate
    :param lamda: trade-off variable, between quality and representativity
    :param save_fig: True = save fig, False = show it without saving
    :return:
    """
    # X axis = size of graph (nodes), y final score of that the algo gave
    # X : K, Y : score
    # graph = read_graph(data_path)

    json_name = path_to_save_data + "/info.json"
    if not os.path.exists(path_to_save_data):
        os.mkdir(path_to_save_data)
    info = {}
    info_arr = dict()
    for func in algo_func:
        info_arr[func.__name__] = np.zeros((3, len(K_list)))
    for idx,k in enumerate(K_list):
        print("starting for k : ",k)
        for func in algo_func:
            print("Current algo : ", func.__str__())
            _set_of_k_nodes, _score, run_time = func(function_val_type, graph, k, lamda)
            info[func.__name__ + "#k=" + str(k)] = {'input_data_source' : input_path,
                                                    'algo':func.__name__,
                                                    'run_time': run_time,
                                                    'score' : _score,
                                                    'R': [k for k in _set_of_k_nodes],
                                                    'lamda': lamda,
                                                    'value_function': function_val_type,
                                                    'nodes_in_graph' : num_of_nodes_in_graph_str,
                                                    'edges_in_graph' : num_of_edges_in_graph_str}
            info_arr[func.__name__][0][idx] = k
            info_arr[func.__name__][1][idx] = _score
            info_arr[func.__name__][2][idx] = run_time

    # draw graph k over score
    graph_creator(info_arr, "group size over group score",
                                  "Graph nodes : " + num_of_nodes_in_graph_str +\
                                  "\nGraph edges : " + num_of_edges_in_graph_str +\
                                  "\nLamda : " + str(lamda) +\
                                  "\nFunction Val type : " +
                                 str(function_val_type), path_to_save_data \
                  + "/graph_k_size_over_score", save_fig, 1, "Size of R", "Score")

    # draw graph k over runtime
    graph_creator(info_arr, "group size over runetime",
                                  "Graph nodes : " + num_of_nodes_in_graph_str +\
                                  "\nGraph edges : " + num_of_edges_in_graph_str +\
                                  "\nLamda : " + str(lamda) +\
                  "\nFunction Val type : " + str(function_val_type)
                    , path_to_save_data \
                  + "/k_over_runtime", save_fig, 2, "Size of R", "Runetime")

    # saving all the info into json
    with open(json_name, 'w') as f:
        json.dump(info, f)


def plot_2d(x, y, title = None):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples
    (first dimension for x, y coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=100, marker='.')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if title is not None:
        ax.set_title(title)
    plt.show()


def read_graph(file_path):
    extension = os.path.splitext(file_path)[1]
    if extension == ".dot":
        return nx.nx_pydot.read_dot(file_path)
    else:
        return nx.read_edgelist(file_path)

def other():
    # need to add algo_1_brute_force to algo_func, but it takes a lot of time
    algo_func = [algo_2_swap, algo_3_greedy, algo_4_greedy_then_swap]
    # type_1 = X axis = size , Y axis = score
    type_1_output_size_over_score = "Output/size_of_score"


    # random.seed(123) to always get the same random numbers

    # using snap
    # TUNGraph = undirected graph
    G1 = snap.TUNGraph.New()

    G1.AddNode(1)
    G1.AddNode(2)
    G1.AddNode(3)
    G1.AddNode(4)
    G1.AddEdge(1, 2)
    G1.AddEdge(2, 3)
    G1.AddEdge(3, 1)
    # not using snap
    # snap_G = get_random_undirected_graph(100, 50)
    input_name = "input1"
    # snap.SaveGViz(snap_G, "Input/" + input_name + ".dot", "Undirected Random Graph", True,
    #               snap.TIntStrH())
    # G2 = nx.nx_pydot.read_dot("Input/" + input_name + ".dot")
    # G3 = nx.read_edgelist("Input/Facebook_social_circles/facebook_combined_input.txt")
    k_over_score_and_k_over_runtime_calculator("Output/Test5", "Input/Facebook_social_circles/facebook_combined_input.txt",
                                               1, [5, 10],
                                               1, True)

    inputs_location = ["Input/input200Nodes.dot", "Input/input1000Nodes.dot",
                       "Input/input2000Nodes.dot","Input/input3000Nodes.dot",
                       "Input/input4000Nodes.dot"]
    # graph_size_over_run_time(inputs_location, 100, "Output/Test2",1,1, True)





    # b1 = func_rep_2_networkx({4}, G2)
    # b2 = func_rep_2_networkx({1}, G2)
    # _b, aa, run_time = algo_3_greedy(2,G2,2, 1)
    # _b, aa, run_time = algo_1_brute_force(2, G2, 2, 1)
    G = nx.Graph()
    # G.add_edges_from([(1, 5), (5, 32)])

    # A = nx.nx_agraph.to_agraph(G2)
    # H = nx.nx_agraph.from_agraph(A)

    # using snap
    # a = func_rep_1_snap({5}, G1)
    # print(a)

    # using networkx
    # b = func_rep_1_networkx({5}, G2)
    # print(b)

    # draw_undirected_graph(G1, "aa.png", " ")

    # G5 = snap.LoadEdgeList(snap.PNGraph, "test.txt", 0, 1)

    # snap.SaveGViz(G1, "Input/Graph1.dot", "Undirected Random Graph", True, snap.TIntStrH())
    # UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 40)
    snap.DrawGViz(G1, snap.gvlNeato, "test1.png", "graph 2",
                  True)

    # for i in G1.Nodes():
    #     print("node id %d with out-degree %d and in-degree %d" % (
    #             i.GetId(), i.GetOutDeg(), i.GetInDeg()))

    print(G1)


    snap_G = get_random_undirected_graph(100, 50)
    snap.SaveGViz(snap_G, "Input/small_100_nodes_2000_edges", "Undirected Random Graph", True,
                  snap.TIntStrH())

if __name__ == '__main__':
    # The algorithm we want to run
    # [algo_1_brute_force]
    # [swap_same_as_pseudo]
    # [algo_2_swap, algo_3_greedy, algo_4_greedy_then_swap]
    # algo_3_greedy_for_f3_new_tw, algo_3_greedy
    # todo: from here its new
    # missing fast algo greedy for f2
    # [algo_3_greedy_for_f1_new_tw, algo_3_greedy_for_f3_new_tw]

    algo_func = [algo_3_greedy_for_f1_new_tw]
    # "Input/Facebook_social_circles/facebook_combined_input.txt"
    # Input/Small/small_50_nodes_500_edges.dot 10, 20, 40
    input_path = "Input/Facebook_social_circles/facebook_combined_input.txt"
    output_dir = "Output/Test5"
    graph = read_graph(input_path)
    num_of_edges_in_graph_str = str(graph.number_of_edges())
    num_of_nodes_in_graph = graph.number_of_nodes()
    num_of_nodes_in_graph_str = str(num_of_nodes_in_graph)
    func_type = 3
    k_over_score_and_k_over_runtime_calculator(output_dir,
                                               func_type,
                                               [10,70],
                                               1,
                                               True)
    # [5, 7, 10, 13, 17, 20, 30, 50]


    SCORE = 1
    CORRESPONDING_NODE = 2
    BRUTE_FORCE = 'algo_1_brute_force'
    # precision_graph_creator("Output/Test5/info_func3_50nodes_500_edges.json", CORRESPONDING_NODE)
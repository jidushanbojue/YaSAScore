import pandas as pd
import networkx as nx
import numpy as np
import os
# from networkx.exception import NetworkXNoPath
# from collections import defaultdict
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps
from multiprocessing import Pool
import time
import itertools as it
import argparse

def timethis(func):
    """
    Decorator that reports the execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper




def get_steps(graph_file, degree_file, reaction_to_structure):

    G_reverse = nx.read_graphml(graph_file)
    # G_reverse = G.reverse()
    relation_df = pd.read_csv(reaction_to_structure)
    product_ids = relation_df[relation_df['role']=='Product']['structure_id']### no

    degree_df = pd.read_csv(degree_file)
    zero_degree_ids = degree_df[degree_df['in_degree']==0]['structure_id']

    temp_list = []
    temp_worker = partial(worker, G_reverse)
    print(temp_worker)
    for p_id in product_ids[:12]:
        print(p_id)
        for t_id in zero_degree_ids:
            temp_list.append((str(p_id), str(t_id)))
    start = time.time()
    with ProcessPoolExecutor(8) as executor:
        result = executor.map(temp_worker, temp_list)
    end = time.time()
    print('process consume the time ', end-start)
    print(len(result))
    print(result[:10])


# @timethis
def worker(iter_obj=None, target_ids=None, product_ids=None):
    for node, dic in iter_obj:
        if node in product_ids:
            print(node)
            w = open(os.path.join(args.out, str(node)+'.csv'), 'w')
            for t_id in target_ids:
                try:
                    step = dic[t_id]
                    w.write(node+','+t_id+','+str(step)+'\n')
                except KeyError as e:
                    # print('There is no path between {} and {}'.format(node, t_id))
                    pass
            w.close()


def generate_shortest_path(graph_file, degree_file, reaction_to_structure):
    # print(type(start))
    print('Beginning read graph file')
    start = time.time()
    graph = nx.read_graphml(graph_file)
    end = time.time()
    print('Read the graph file consumed {}'.format((end-start)/60))

    relation_df = pd.read_csv(reaction_to_structure)
    product_ids = relation_df[relation_df['role']=='Product']['structure_id']
    product_ids = list(set([str(el) for el in product_ids]))

    worker_list = np.array_split(product_ids, 30)
    # print(worker_list)

    degree_df = pd.read_csv(degree_file)
    zero_degree_ids = degree_df[degree_df['in_degree']==0]['structure_id']
    zero_degree_ids = set([str(el) for el in zero_degree_ids])

    short_path_iter = nx.shortest_path_length(graph)

    # nx.single_target_shortest_path_length()

    # print('Beginning slice the iterator object')
    # iter_obj = it.islice(short_path_iter, start_idx, start_idx+10)
    # for obj in iter_obj:
    #     print(obj[0])

    process = partial(worker, iter_obj=short_path_iter, target_ids=zero_degree_ids)
    for products in worker_list:
        process(product_ids=products)


    # pool = Pool(cpu_num)
    # pool.map(process, worker_list)
    # pool.close()
    # pool.join()
    #
    #


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='split graph iterator and generate reaction steps')
    parser.add_argument('-gf', '--graph_file', type=str, default=None, help='Specify the graph file ')
    parser.add_argument('-df', '--degree_file', type=str, default=None, help='Specify the degree file')
    parser.add_argument('-rf', '--relation_file', type=str, default=None, help='Specify the reation_to_structure file')
    # parser.add_argument('-is', '--iterator_start', type=int, default=0, help='Specify the iterator start index')
    # parser.add_argument('-ie', '--iterator_end', type=int, default=0, help='Specify the iterator end index')
    parser.add_argument('-o', '--out', type=str, default=None, help='Specify the absolute path to the folder to which the result should be written')
    # parser.add_argument('-n_cpu', '--cpu_num', type=int, default=8, help='Specify the number of cpu to use')


    # args = parser.parse_args(['-gf', '../../yasascore_test/data/chemical_reaction_network/uspto_graph_reverse.graph',
    #                           '-df', '../../yasascore_test/data/chemical_reaction_network/degree.csv',
    #                           '-rf', '../../yasascore_test/data/chemical_reaction_network/reaction_to_structure_USPTO_test.csv',
    #                           '-o', '../../yasascore_test/data/chemical_reaction_network/shortest_path',
    #                           '-n_cpu', '100'])

    args = parser.parse_args()

    if os.path.exists(args.out):
        pass
    else:
        os.mkdir(args.out)

    # generate_shortest_path(args.graph_file, args.degree_file, args.relation_file, args.iterator_start)
    generate_shortest_path(args.graph_file, args.degree_file, args.relation_file)








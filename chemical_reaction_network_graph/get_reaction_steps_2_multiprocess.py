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

# @timethis
# def worker(Graph, info):
#     """
#     :param Graph:
#     :param pid: product id
#
#     : info: (pid, target_id)
#     :param degree_ids: is a list or pd.Series, its in_degree is 0
#     :return:
#     """
#     # step_dict = defaultdict(list)
#     # print(info)
#     try:
#         short_path = nx.shortest_path(Graph, str(info[0]), str(info[1]))
#         print(short_path)
#         return short_path
#     except NetworkXNoPath as e:
#         return

    # for target_id in degree_ids:
    #     print(target_id)
    #     try:
    #         short_path = nx.shortest_path(Graph, str(pid), str(target_id))
    #         step_dict[pid].append(short_path)
    #     except NetworkXNoPath as e:
    #         print(e)
    # return step_dict


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

    # p = Pool(3)
    # result = p.map(temp_worker, temp_list)
    # print(len(result))
    # print('done!!!')
    # with ThreadPoolExecutor(10) as executor:
    #     result = executor.map(temp_worker, temp_list)
    # print(result)
    # for r in result:
    #     print(r)

    # executor = ThreadPoolExecutor(max_workers=10)
    # print(executor)
    # all_tasks = [executor.submit(temp_worker, (info[0], info[1])) for info in temp_list]
    #
    # for task in as_completed(all_tasks):
    #     data = task.result()
    #     print(data)
# reaction_step_dir = '/home/baiqing/reaction_step'
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
    # for products in worker_list:
    #     process(product_ids=products)


    pool = Pool(30)
    pool.map(process, worker_list)
    #
    #
    # worker(short_path_iter, product_ids=product_ids, target_ids=zero_degree_ids)




    # p = Pool(10)
    # for iter_obj in iter_list:
    #     p.apply(worker, args=(iter_obj, product_ids, zero_degree_ids))
    #
    # p.close()
    # p.join()



    # worker(iter_list[0], product_ids=product_ids, target_ids=zero_degree_ids)

    # partial_worker = partial(worker, product_ids=product_ids, target_ids=zero_degree_ids)
    # # p = Pool(10)
    # # p.map(partial_worker, iter_list)
    # # with ProcessPoolExecutor(10) as executor:
    # #     executor.submit()
    # with ProcessPoolExecutor(10) as executor:
    #     futures = {executor.submit(partial_worker, iter_obj) for iter_obj in iter_list}



    # res_dict = dict()

    # w = open(result_file, 'a+')
    # for node, dic in short_path_iter:
    #     print(node)
    #     if node in product_ids:
    #         for t_id in zero_degree_ids:
    #             try:
    #                 step = dic[t_id]
    #                 w.write(node+','+t_id+','+str(step)+'\n')
    #             except KeyError as e:
    #                 # print('There is no path between {} and {}'.format(node, t_id))
    #                 pass
    # w.close()



    #
    # start = time.time()
    # short_path_iter = nx.shortest_path_length(graph)



    # for short_path in short_path_iter:
    #     print(short_path)
    #
    # end = time.time()
    # print(type(short_path_iter))
    # print('Generate shortest path consumed {}'.format((end-start)/60))


if __name__ == '__main__':
    # base_dir = '/home/baiqing/PycharmProjects/ReactionDB/data'
    # graph_file = os.path.join(base_dir, 'G_reverse_time.graphml')
    # degree_file = os.path.join(base_dir, 'node_degree_new.csv')
    # reaction_to_structure = os.path.join(base_dir, 'reaction_to_structure_no_dup.csv')
    # # get_steps(graph_file, degree_file, reaction_to_structure)
    # result_file = os.path.join(base_dir, 'steps.csv')
    # generate_shortest_path(graph_file, degree_file, reaction_to_structure, result_file)

    parser = argparse.ArgumentParser(description='split graph iterator and generate reaction steps')
    parser.add_argument('-gf', '--graph_file', type=str, default=None, help='Specify the graph file ')
    parser.add_argument('-df', '--degree_file', type=str, default=None, help='Specify the degree file')
    parser.add_argument('-rf', '--relation_file', type=str, default=None, help='Specify the reation_to_structure file')
    # parser.add_argument('-is', '--iterator_start', type=int, default=0, help='Specify the iterator start index')
    # parser.add_argument('-ie', '--iterator_end', type=int, default=0, help='Specify the iterator end index')
    parser.add_argument('-o', '--out', type=str, default=None, help='Specify the absolute path to the folder to which the result should be written')

    args = parser.parse_args(['-gf', '/home/baiqing/PycharmProjects/ReactionDB/data/all_graph_with_relationship_reverse_new.graphml',
                              '-df', '/home/baiqing/PycharmProjects/ReactionDB/data/node_degree_with_relationship_new.csv',
                              '-rf', '/home/baiqing/PycharmProjects/ReactionDB/data/reaction_to_structure_new.csv',
                              '-o', '/data/baiqing/src_data/reaction_step/shortest_path_multiprocess'])

    if os.path.exists(args.out):
        pass
    else:
        os.mkdir(args.out)

    # generate_shortest_path(args.graph_file, args.degree_file, args.relation_file, args.iterator_start)
    generate_shortest_path(args.graph_file, args.degree_file, args.relation_file)


    # threads = {}
    # for p_id in product_ids:
    #     for target_id in zero_degree_ids:
    #         t = MyThread(worker, (G_reverse, str(p_id), str(target_id)), worker.__name__)
    #         threads[(p_id, target_id)] = t
    #
    #     for target_id in zero_degree_ids:
    #         threads[(p_id, target_id)].start()
    #
    #     for target_id in zero_degree_ids:
    #         threads[(p_id, target_id)].join()
#
#
#
#     print('Done')

# def main():
#     print('start at: ', ctime())
#     threads = []
#     nloops = range(len(loops))
#     for i in nloops:
#         t = MyThread(loop, (loops[i], ), loop.__name__)
#         threads.append(t)
#
#     for i in nloops:
#         threads[i].start()
#     for i in nloops:
#         threads[i].join()
#
#     print(threads[i].get_result())
#     print(threads[0].get_result())
#     print('Done AT: ', ctime())

#
#
# def get_steps(graph_file, degree_file, reaction_to_structure):
#
#     G = nx.read_graphml(graph_file)
#     G_reverse = G.reverse()
#     relation_df = pd.read_csv(reaction_to_structure)
#     product_ids = relation_df[relation_df['role']=='Product']['structure_id']### no
#
#     degree_df = pd.read_csv(degree_file)
#     zero_degree_ids = degree_df[degree_df['in_degree']==0]['structure_id']
#
#     for p_id in product_ids:
#         worker(G_reverse, p_id, zero_degree_ids)
#
#     print('Done')
#
#
#
# if __name__ == '__main__':
#     base_dir = '/home/baiqing/PycharmProjects/ReactionDB/data'
#     graph_file = os.path.join(base_dir, 'all.graphml')
#     degree_file = os.path.join(base_dir, 'node_degree.csv')
#     reaction_to_structure = os.path.join(base_dir, 'reaction_to_structure_top.csv')
#     get_steps(graph_file, degree_file, reaction_to_structure)

# from time import ctime, sleep
# import threading
# import numpy as np
# import collections
#
# loops = ['GuangZhou', 'Beijing']
# t_list = ['01', '02', '03']
# cldas_sum = collections.deque()

# class MyThread(threading.Thread):
#     def __init__(self, func, args, name=''):
#         # super(self, MyThread).__init__(self)
#         super().__init__()
#         # threading.Thread.__init__(self)
#         self.name = name
#         self.func = func
#         self.args = args
#         self.result = self.func(*self.args)
#
#     def get_result(self):
#         try:
#             return self.result
#         except Exception:
#             return None

# def loop(nloop):
#     for j in t_list:
#         cldas_values = []
#         for k in range(4):
#             cldas_value = nloop + str(k)
#             cldas_values.append(cldas_value)
#         cldas_values.append(j)
#         cldas_values.append(nloop)
#         cldas_sum.append(cldas_values)
#         print(id(cldas_values))
#     return cldas_sum
#
# def main():
#     print('start at: ', ctime())
#     threads = []
#     nloops = range(len(loops))
#     for i in nloops:
#         t = MyThread(loop, (loops[i], ), loop.__name__)
#         threads.append(t)
#
#     for i in nloops:
#         threads[i].start()
#     for i in nloops:
#         threads[i].join()
#
#     print(threads[i].get_result())
#     print(threads[0].get_result())
#     print('Done AT: ', ctime())
#
# if __name__ == '__main__':
#     main()






import pandas as pd
import networkx as nx
import os
from matplotlib import pyplot as plt
from networkx import spring_layout
from networkx import draw_networkx_edges
from networkx import draw_networkx_nodes
from networkx import draw_networkx_labels
from multiprocessing import Pool
from collections import defaultdict
import itertools as it
import argparse

def get_edges(info):
    name = info[0]
    group = info[1]
    edges_list = []
    product_id = group[group['role']=='Product']['structure_id']
    reactant_id = group[group['role']=='Reactant']['structure_id']
    for r_id in reactant_id:
        for p_id in product_id:
            edges_list.append((r_id, p_id, {'relationship': name}))
    return edges_list

def get_edges_with_relationship(info):
    name = info[0]
    group = info[1]
    # edges_list = []
    edges_set = defaultdict(set)
    product_id = group[group['role']=='Product']['structure_id']
    reactant_id = group[group['role']=='Reactant']['structure_id']
    for r_id in reactant_id:
        for p_id in product_id:
            edges_set[(r_id, p_id)].add(name)
            # edges_list.append((r_id, p_id, {'relationship': name}))
    # return edges_list



def generate_net(src_file, out_graph, reverse_graph, degree_file, cpu=10):

    p = Pool(cpu)
    G = nx.DiGraph()

    df = pd.read_csv(src_file)

    edges_dic = defaultdict(list)
    for name, group in df.groupby('reaction_id'):
        print(name)
        # result_list.append((name, group))
        # get_edges_with_relationship((name, group))
        product_id = group[group['role']=='Product']['structure_id']
        reactant_id = group[group['role']=='Reactant']['structure_id']
        for r_id in reactant_id:
            for p_id in product_id:
                edges_dic[(r_id, p_id)].append(name)
    # print(edges_dic)
    # for key, value in edges_dic.items():
    #     print(key, value)

    edges_list = []

    for key, value in edges_dic.items():
        new_value = [str(el) for el in value]
        edges_list.append((key[0], key[1], {'relationship': '_'.join(new_value), 'weight': len(new_value)}))


    # edges_list = [(key[0], key[1], {'relationship': '_'.join(value), 'weight': len(value)}) for key, value in edges_dic.items()]

    G.add_edges_from(edges_list)

    get_degree(G, degree_file)
    # nx.write_graphml(G, '../data/uspto_graph_with_relationship.graphml')
    nx.write_graphml(G, out_graph)
    print('Generating chemical knowledge graph')

    G_reverse = G.reverse()
    nx.write_graphml(G_reverse, reverse_graph)
    print('Generating reverse chemical knowledge graph')


def get_degree(G, res_file):
    df = pd.DataFrame(G.degree, columns=['structure_id', 'degree'])
    df['in_degree'] = [el[1] for el in G.in_degree]
    df['out_degree'] = [el[1] for el in G.out_degree]
    df.to_csv(res_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate network multiprocessing')
    parser.add_argument('-i', '--input', type=str, default=None, help='Specify the absolute path to the reaction_to_structure.csv')
    parser.add_argument('-o', '--out_graph', type=str, default=None, help='Specify the absolute path to the network_graph')
    parser.add_argument('-ro', '--reverse_graph', type=str, default=None, help='Specify the absolute path to the reverse-network_graph')
    parser.add_argument('-d', '--degree', type=str, default=None, help='Specify the absolute path to the node degree of graph')
    parser.add_argument('-n_cpu', '--num_cpu', type=int, default=8, help='Specify the number of cpu to use')

    # args = parser.parse_args(['-i' '/data/baiqing/PycharmProjects/yasascore_test/data/chemical_reaction_network/reaction_to_structure_USPTO_test.csv', '-o', '/data/baiqing/PycharmProjects/yasascore_test/data/uspto_graph_with_relationship_test.graphml', '-ro', '/data/baiqing/PycharmProjects/yasascore_test/data/uspto_graph_with_relationship_reverse_test.graphml', '-d', '/data/baiqing/PycharmProjects/yasascore_test/data/degree_test.csv'])
    args = parser.parse_args()


    src_file = args.input
    out_graph = args.out_graph
    reverse_graph = args.reverse_graph
    degree_file = args.degree
    num_cpu = args.num_cpu
    generate_net(src_file, out_graph, reverse_graph, degree_file, num_cpu)
    print('Done')


    # graphml_file = 'uspto_graph_with_relationship.graphml'
    # node_degree_file = os.path.join(base_dir, 'node_degree_with_relationship_uspto.csv')
    # G = nx.read_graphml(graphml_file)
    # G_reverse = G.reverse()
    # nx.write_graphml(G_reverse, '../data/uspto_graph_with_relationship_reverse.graphml')
    # get_degree(G, node_degree_file)
    # print('Done')



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



def generate_net(src_file):

    p = Pool(160)
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

    # nx.write_graphml(G, '../data/all_graph_with_relationiship_new_1.graphml')
    # nx.write_gml(G, '../data/all_graph_with_relationiship_new_1.gml')
    nx.write_graphml(G, '../data/uspto_graph_with_relationship.graphml')


    # for tpl in result_list:
    #     get_edges_with_relationship(tpl)



    # result = p.map(get_edges_with_relationship, result_list)
    # for i in result:
    #     print(i)
    #     G.add_edges_from(i)
    #     # edges = generate_edges(name, group)
    #     # print(edges)
    #     # G.add_edges_from(edges)
    #
    # nx.write_graphml(G, '../data/all_1.graphml')

    # pos = spring_layout(G)
    # nodes = draw_networkx_nodes(G, pos)
    # # print(G.edges(data=True))
    # nodes.set_edgecolor('y')
    # nodes.set_color('r')
    # lines = draw_networkx_edges(G, pos)
    # draw_networkx_labels(G, pos)
    # # nx.draw(G, with_labels='True', edge_color='r')
    # # plt.axis('on')
    # # plt.xticks([])
    # # plt.yticks([])
    # plt.show()

    # nx.write_graphml('test.graphml')
    # nx.draw(G, with_labels=True, font_weight='bold')

def get_degree(G, res_file):
    df = pd.DataFrame(G.degree, columns=['structure_id', 'degree'])
    df['in_degree'] = [el[1] for el in G.in_degree]
    df['out_degree'] = [el[1] for el in G.out_degree]
    df.to_csv(res_file)


if __name__ == '__main__':
    base_dir = '/home/baiqing/PycharmProjects/ReactionDB/data'
    # src_file = os.path.join(base_dir, 'reaction_to_structure_no_dup.csv')
    # src_file = os.path.join(base_dir, 'reaction_to_structure_new_1.csv') ### USPTO+Pistachio
    src_file = os.path.join(base_dir, 'reaction_to_structure_USPTO.csv')

    generate_net(src_file)
    # print('Done')


    # graphml_file = '/home/baiqing/PycharmProjects/ReactionDB/data/all_graph_with_relationiship_new_1.graphml'
    # node_degree_file = os.path.join(base_dir, 'node_degree_with_relationship_new_1.csv')
    # G = nx.read_graphml(graphml_file)
    # G_reverse = G.reverse()
    # nx.write_graphml(G_reverse, '../data/all_graph_with_relationship_reverse_new_1.graphml')
    # get_degree(G, node_degree_file)
    # print('Done')




import networkx as nx


def load_train(name):
    # construct iteratively due to memory issues
    G = nx.DiGraph()
    with open(name) as sr:
        for line in sr:
            nodes = list(map(int, line.split()))
            for neighbor in nodes[1:]:
                G.add_edge(nodes[0], neighbor)
    return G


def main():
    data = load_train('data/train.txt')

if __name__ == '__main__':
    main()

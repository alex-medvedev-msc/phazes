import numpy


def raw_matrices(size, count):
    pass


def archive_matrices(matrices):
    return matrices


def make_graph(archived):
    pass


def clusterize(graph):
    pass


def visualize(graph, clusters):
    pass


def main():
    matrices = raw_matrices()
    archived = archive_matrices(matrices)
    graph = make_graph(archived)
    clusters = clusterize(matrices)
    visualize(graph, clusters)


if __name__ == '__main__':
    main()
import numpy
from automata import simulate
from rules import rule_180_with_signals
import matplotlib.pyplot as pyplot
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from collections import Counter
import time


class Changer(object):
    """ This class turns on and off traffic signals at the end of the road and generates new vehicles on start
    with respect to rate. Traffic signal will be always green at first step
    """
    def __init__(self, rate, red, green):
        """
        Args:
            rate (float): probability of empty square in first row turning into vehicle
            red (int): number of seconds when red traffic signal is active, must be positive
            green (int): number of seconds when green traffic signal is active, must be positive
        """
        if not red or not green:
            raise ValueError("red and green must be positive")
        self.rate = rate
        self.red = red
        self.green = green
        self.steps = 0
        self.is_green_active = True

    def _generate_vehicles(self, matrix):
        for row in range(matrix.shape[0]):
            if matrix[row, 0] == 0 and numpy.random.rand() < self.rate:
                matrix[row, 0] = 1

    def _activate_green(self, matrix):
        if self.is_green_active:
            return
        if self.steps < self.red:
            return

        matrix[:, -1] = 0
        self.is_green_active = True
        self.steps = 0

    def _activate_red(self, matrix):
        if not self.is_green_active:
            return
        if self.steps < self.green:
            return
        matrix[:, -1] = -1
        self.is_green_active = False
        self.steps = 0

    def change(self, matrix):
        """
        Generates new vehicles on road start and turns on/off traffic signal at the road end

        Args:
            matrix (numpy.ndarray): current road state
        """

        self._generate_vehicles(matrix)
        self._activate_green(matrix)
        self._activate_red(matrix)
        self.steps += 1


class Monitor(object):
    def __init__(self, animate=False):
        self.animate = animate
        self.matrices = []
        self.speeds = []
        self.densities = []
        self.images = []

    def observe(self, old_matrix, new_matrix):
        """
        Simply records each old_matrix to `self.matrices` and calculates average speed and density on road
        Args:
            old_matrix (numpy.ndarray): state of the road before iterate
            new_matrix (numpy.ndarray): state of the road after iterate
        """

        self.matrices.append(old_matrix)
        average_speed = ((new_matrix - old_matrix) == 1).sum()/new_matrix.size
        density = new_matrix[new_matrix > 0].sum()/new_matrix.size
        self.speeds.append(average_speed)
        self.densities.append(density)
        if self.animate:
            self.images.append((pyplot.pcolor(new_matrix, norm=pyplot.Normalize(-1, 1)), ))


def raw_matrices(size, count):
    lanes, length = size
    monitor = Monitor()
    changer = Changer(0.3, 25, 25)
    result = simulate(lanes=lanes, length=length, steps=count, rule=rule_180_with_signals, monitor=monitor, changer=changer)
    #pyplot.plot(monitor.densities, monitor.speeds, 'bo')
    #pyplot.show()
    return monitor


def archive_matrices(matrices):
    return matrices


def make_graph(archived):
    pass


def clusterize(matrices):
    #dbscan = DBSCAN(metric="precomputed", eps=25, min_samples=50)
    cluster = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="complete")
    distances = distance_matrix(matrices)
    print("mean of distances is {} and std of norms is {}".format(numpy.mean(distances), numpy.std([numpy.linalg.norm(m, numpy.inf) for m in matrices])))
    #pyplot.plot([numpy.linalg.norm(m, numpy.inf) for m in matrices], 'ro')
    #pyplot.show()
    #pyplot.hist(distances.flatten(), bins=20)
    #pyplot.show()
    return cluster.fit_predict(distances)


def distance_matrix(matrices):
    distances = numpy.zeros((len(matrices), len(matrices)), dtype=float)
    for i, m1 in enumerate(matrices):
        for j, m2 in enumerate(matrices):
            distances[i, j] = numpy.linalg.norm(m1 - m2, numpy.inf)
    return distances


def visualize(graph, clusters):
    pass


def animate_road(monitor):
    im_ani = animation.ArtistAnimation(pyplot.figure(), monitor.images, interval=500, repeat_delay=3000,
                                       blit=True)
    pyplot.show()


def main():
    start = 0
    monitor = raw_matrices((1, 100), 2500)

    clusters = clusterize(monitor.matrices[start:])
    counter = Counter(clusters)
    print(counter)
    densities = numpy.array(monitor.densities[start:])[clusters >= 0]
    speeds = numpy.array(monitor.speeds[start:])[clusters >= 0]
    f, ax = pyplot.subplots(2, sharex=True, sharey=True)
    ax[0].plot(monitor.densities[start:], monitor.speeds[start:], "ro")
    ax[1].scatter(densities, speeds, c=clusters[clusters >= 0], s=50, vmin=0, vmax=clusters.max(), cmap=pyplot.get_cmap("viridis"))
    pyplot.show()


if __name__ == '__main__':
    main()
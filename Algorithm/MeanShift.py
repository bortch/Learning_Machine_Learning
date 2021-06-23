import numpy as np
from numpy import linalg, mod
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt, style
import logging as log
style.use("ggplot")


class MeanShift():
    """Meanshift - learning by doing

    Keyword arguments:
    bandwidth -- the windows size (default None)
    bin -- Chunk the Bandwidth in small parts (default None)
    max_iter -- Maximum iteration Number (default 300)
    """

    def __init__(self, bandwidth=None, bin=None, max_iter=300):
        self.bandwidth = bandwidth
        self.bin = bin
        self.max_iter = max_iter

    def _get_bandwidth(self, data):
        log.info(f"\nComputing Bandwidth:")
        # get centroid for all data
        centroid_4_all_data = np.average(data, axis=0)
        log.info(f"\tGet centroid for all data: {centroid_4_all_data}")
        # compute norm of all data with the centroid
        all_data_norm = np.linalg.norm(centroid_4_all_data)
        log.info(
            f"\tCalculate vector norm of the centroid of all data: { all_data_norm }")
        if (self.bin==None):
          self.bin = 2
        self.bandwidth = all_data_norm/self.bin
        log.info(
            f"\tComputed bandwidth's bin size:\n\t({all_data_norm}/{self.bin}) = {self.bandwidth}\n")

    def _get_gaussian_kernel_weight(self, x, points, bandwidth):
        '''
        Gaussian Kernel formula: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

        note kernel at position : np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        '''
        distances = euclidean_distances(points, x.reshape(1, -1))
        log.debug("distances", distances.shape)
        weights = np.exp(-1 * (distances**2 / bandwidth**2))
        log.debug(f'point shape:{points.shape}, weights.shape:{weights.shape}')
        weighted_mean = np.sum(points * weights, axis=0) / np.sum(weights)
        return weighted_mean

    def _get_next_centroid(self, centroid, points):
        return self._get_gaussian_kernel_weight(centroid, points, self.bandwidth)

    def _get_neighbors(self, centroid, data):
        neighbors = []
        for pts in data:
            if np.linalg.norm(pts-centroid) < self.bandwidth:
                neighbors.append(pts)
        return np.array(neighbors)

    def _arr_to_dict(self, arr):
        '''returns a dictionary'''
        results = {}
        for i in range(len(arr)):
            results[i] = arr[i]
        return results

    def _filter_centroids(self, _centroids):
        ''' Remove redundant centroids
            Remove to close centroids
        '''
        # create a set to keep only distinct centroids
        distinct_centroid = sorted(list(set(_centroids)))
        log.info(f"\nAll distinct centroids found:\n{distinct_centroid}")

        # remove too close centroids
        if(len(distinct_centroid) > 1):
            to_pop = set()
            for i in distinct_centroid:
                if(not i in to_pop):
                    log.info(f"compare {i}")
                    for j in [i for i in distinct_centroid]:
                        log.info(f"\twith centroid: {j}")
                        if (i == j) or (j in to_pop):
                            # we skip as we are comparing the same entry
                            log.info(f"\tcomparison skipped")
                            pass
                        elif np.linalg.norm(np.array(i)-np.array(j)) <= self.bandwidth:
                            log.debug(
                                f"as {np.linalg.norm(np.array(i)-np.array(j))}<={self.bandwidth}")
                            log.info(f"\tadd to remove: {j}")
                            to_pop.add(j)  # we mark one of them to be removed
                            break
                else:
                    log.debug(f"{i} to be removed")
            log.debug(f"centroids to remove:{to_pop}")
            for i in to_pop:
                distinct_centroid.remove(i)
        log.info(f"filtered list of centroids: {distinct_centroid}")
        return distinct_centroid

    def _combine_solutions(self, sol_1, sol_2):
        index = len(sol_1)
        result = sol_1
        for value in sol_2.values():
            found = False
            # check if exists
            for i in range(len(sol_1)):
                if np.array_equal(sol_1[i], value):
                    found = True
                    break
            if not found:  # we add it
                result[index] = value
                index += 1
        log.debug(f"merged centroids: {result}")
        return result

    def _build_clusters(self, centroids, data):
        self.centroids = centroids
        self.clusters = {}

        for i in range(len(self.centroids)):
            self.clusters[i] = []
        log.info(f"\t{len(self.clusters)} clusters found!")

        if len(self.clusters) > 0:
            for feature in data:
                # get the euclidean distance to either centroid
                log.info(f"\tCompare feature:{feature}")
                distances = np.empty(len(self.centroids))
                for i in self.centroids:
                    log.debug(
                        f"{i} norm({feature}-{self.centroids[i]})={np.linalg.norm(feature-self.centroids[i])}")
                    distance = np.linalg.norm(feature-self.centroids[i])
                    log.debug(
                        f"\tdistance to centroid {self.centroids}: {distance}")
                    if not distance in distances:
                        distances[i] = distance
                log.debug(f"\tEvery distances computed:{distances}")
                log.info(f"\tMinimal distance is {min(distances)}")
                cluster_id = np.where(distances == (min(distances)))[0].item()

                # feature that belongs to that cluster
                log.info(f"add feature {feature} to class {cluster_id}")
                self.clusters[cluster_id].append(feature)

    def _has_converged(self, new, previous):
        log.info(f"\n{'.'*80}\nHas solution converged?")
        new = np.array(new)
        previous = np.array(previous)
        log.debug(f"{type(previous)}, {previous.shape}\n{type(new)}, {new.shape}")
        if(new.shape==previous.shape):
          res = np.allclose(new, previous)
        else:
          res = False
        log.info(f"is converged? {res}")
        return res

    def fit(self, data):

        current_iter = 0
        if self.bandwidth == None:
            self._get_bandwidth(data)
        # set all value as potential centroids
        # (or starting point) for search of centroids
        centroids = self._arr_to_dict(data)
        log.info(f"Dictionnary of initial centroids:\n{centroids}")
        prev_centroids_list = []
        while True:
            log.info(f"\n{'_'*40}{current_iter}{'_'*40}\n")
            new_centroids_list = []
            for i in centroids:
                centroid = centroids[i]
                points = self._get_neighbors(centroid, data)
                log.info(
                    f"\n{'_'*80}\nStart with Point:{centroid} as a centroid")

                log.info("Computing Distance:")
                # calculating the weight for each point
                # that are in the bandwidth
                # based on euclidean distance between
                # choosed the point and the centroid

                new_centroid = self._get_next_centroid(centroid, points)
                log.info(f"New potential centroid:{new_centroid}")
                new_centroids_list.append(tuple(new_centroid))

            new_centroids_list = self._filter_centroids(new_centroids_list)

            # ends centroids research if converged
            converged = self._has_converged(new_centroids_list, prev_centroids_list)
            log.info(
                f"Previously centroids found: {prev_centroids_list}\n \
                Current iteration centroids found: {new_centroids_list}")
            # assign new centroids found for potential next loop itereation
            centroids = {}
            for i in range(len(new_centroids_list)):
                 centroids[i] = np.array(new_centroids_list[i])

            if converged:
                break
            
            if(current_iter <= self.max_iter):
                current_iter += 1
                prev_centroids_list = new_centroids_list
            else:
                break

        # Search for centroids ended
        # we've found some centroids of clusters
        log.info(f"\n{'-°-.'*20}\n")
        self._build_clusters(centroids, data)
        return self

    def predict(self, data):
        # Todo
        pass

if __name__ == "__main__":

    from sklearn.datasets import make_blobs

    log.basicConfig(format='%(message)s', level=log.ERROR)

    # Examples of parameters values:
    # * n_points    :30   |15
    # * random_state 7    |7   |6 |5 |4    |1    |0
    # * bin          4    |3.5 |3 |5 |5.81 |2.52 |7
    n_points = 30
    random_state = 7
    bin = 4
    X, y = make_blobs(n_samples=n_points, centers=3, center_box=(0, 10),
                      n_features=2, random_state=random_state
                      )

    model = MeanShift(bin=bin)  # bandwidth=2.)
    model.fit(X)

    plt.figure(figsize=(10, 10), dpi=72)
    plt.axis('equal')
    colors = ["g", "r", "b", "c", "y"]
    theta = np.linspace(0, 2 * np.pi, 800)
    plt.scatter(X[:, 0], X[:, 1], s=10, c="k")
    for c in model.clusters:
        log.debug(
            f"classe:{c}\ncentroid: {model.centroids[c]}\n{model.clusters[c]}")
        color = colors[c % len(colors)]
        points = model.clusters[c]
        centroid = model.centroids[c]
        
        for feature in points:
            plt.scatter(feature[0], feature[1], marker="o",
                        c=color, s=50, linewidths=1, zorder=10, alpha=0.2)
        
        # define a radius to circle the cluster
        norm_cluster = linalg.norm(points-centroid, axis=1)
        radius = max(norm_cluster)
        plt.scatter(centroid[0], centroid[1], c=color, marker='+', s=150)
        x, y = (np.cos(theta) * radius) + \
            centroid[0], (np.sin(theta) * radius) + centroid[1]
        plt.plot(x, y, linewidth=1, color=color)
        plt.fill_between(x, y,facecolor=color,alpha=0.1)

    plt.show()

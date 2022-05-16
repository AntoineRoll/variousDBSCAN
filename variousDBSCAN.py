from anytree import Node
from sklearn.cluster import DBSCAN
import numpy as np


class VariousDBSCAN:
    def __init__(self, distance_matrix, original_epsilon=0.5, min_points=5, update_epsilon_func=None):
        """Run VariousDBSCAN from a distance matrix.
         VariousDBSCAN is an adapted algorithm from the Density-Based Spatial Clustering of Applications with Noise.
         It is able to create clusters with various density.

         Parameters
         ----------
        distance_matrix : ndarray of shape (n_samples, n_samples)
            The distance matrix between all sample_points. Dense Numpy array.

        original_epsilon : float, default=0.5
            The min_eps parameter that will be passed to the first DBSCAN fit.

        min_points : int, default=5
            The min_point parameter that will be passed to all DBSCAN fit.
        update_epsilon_func : function, default=None
            The function used to update the min_eps parameter everytime we want to find denser clusters.
            This function must take a float in parameter and return a smaller float.
            If 'None', the function used will be 'x->x/2'.
        """
        assert isinstance(min_points, int) and min_points > 0, "min_points must be an integer strictly greater than 0."
        self.distance_matrix = distance_matrix
        self.epsilon = original_epsilon
        self.min_points = min_points
        self.n_points = len(distance_matrix)
        self.progressive_clustering = Node('',
                                           clustered_points=[])
        self.clusters = None

        if update_epsilon_func is None:
            self.update_epsilon_func = lambda x: x / 2
        else:
            self.update_epsilon_func = update_epsilon_func
        self.dbscan_performed = 0

    def run_dbscan_on_node(self, points_idx, new_eps, parent_node):
        """Will perform a DBSCAN on a sample of the original distance_matrix.
        Returns None but updates 'parent_node' children with new clusters if any.

        Parameter
        ---------
        points_idx : list of integers
            The indices of the sample points to perform DBSCAN

        new_eps : float
            The min_eps value to perform DBSCAN

        parent_node : anytree.Node
            The parent node to which new clusters will be attached as children nodes
        """
        dbscan = DBSCAN(metric='precomputed', eps=new_eps, min_samples=self.min_points)
        dbscan.fit(self.distance_matrix[points_idx][:, points_idx])
        self.dbscan_performed += 1

        for cluster_i in range(max(dbscan.labels_) + 1):
            local_idx = set(np.where(dbscan.labels_ == cluster_i)[0])
            original_idx = [points_idx[k] for k in local_idx]

            cluster_node = Node(str(cluster_i),
                                clustered_points=original_idx,
                                parent=parent_node)

    def remove_child_cluster_points_from_parents(self):
        """Updates clustered points of each node to remove points that have been clustered in a node's descendants."""
        nodes = sorted(self.progressive_clustering.descendants, key=lambda x: -x.depth)
        for child_node in nodes:
            parent_nodes = child_node.anchestors
            for parent_node in parent_nodes:
                if parent_node is not None:
                    parent_node.clustered_points = list(
                        set(parent_node.clustered_points).difference(child_node.clustered_points))

    def fit(self, max_depth=np.inf):
        """ Performs the entire fitting of VariousDBSCAN as following:
         1. Perform DBSCAN with the original epsilon value on all the dataset.
         2. Store resulting clusters as children nodes of the root 'self.progressive_clustering'.
         3. Update the epsilon value.
         4. Repeat steps 1-3 on samples made from the clusters until no new clusters can be created.
         5. With the resulting tree, remove from each node's clusters indices that appear
         in the clusters of descendant nodes.
         6. Iterate over the tree and return the clusters that have more points than 'self.min_points'.

         Returns
         -------
         clusters : list of lists of integers
            Returns a list of the clusters ordered by decreasing depth as a list of integers"""

        new_eps = self.epsilon
        self.run_dbscan_on_node(list(range(self.n_points)), new_eps=new_eps, parent_node=self.progressive_clustering)
        depth = 1
        while True:
            if depth >= max_depth:
                break

            next_nodes = [node for node in self.progressive_clustering.descendants if node.depth == depth]
            if len(next_nodes) == 0:
                break

            depth += 1
            new_eps = self.update_epsilon_func(new_eps)
            for bottom_node in next_nodes:
                self.run_dbscan_on_node(bottom_node.clustered_points, new_eps, bottom_node)
        self.remove_child_cluster_points_from_parents()
        self.clusters = [node.clustered_points for node in sorted(self.progressive_clustering.descendants,
                                                                  key=lambda x: -(x.depth + 1))
                         if len(node.clustered_points) >= self.min_points]
        return self.clusters

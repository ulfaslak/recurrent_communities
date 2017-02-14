"""Base for Ulf's personal useful functions."""

import numpy as np
import matplotlib.pylab as plt
import networkx as nx
import math
from random import shuffle
from collections import defaultdict
import community
from scipy.interpolate import interp1d

def plt_log_hist(v, bins=10):
    """Make logplot of histogram."""
    plt.hist(v, bins=np.logspace(np.log10(min(v)), np.log10(max(v)), bins))
    plt.xscale("log")

def plt_cumulative_hist(v, bins=10):
    """Make cumulative histogram plot"""
    values, base = np.histogram(v, bins=bins)
    cumulative = np.cumsum(values)
    plt.plot(base[:-1], cumulative, c='blue')

def pareto_distribution(v, p=0.8):
    """Gets the number of entries in v which accounts for p of its sum.
    v has to be sorted in descending order."""
    thr = np.sum(v)*p
    cumsum = 0
    for i, _v in enumerate(v, 1):
        cumsum += _v
        if cumsum >= thr:
            return i * 1.0 / len(v)

def ApEn(U, m, r):
    """Compute approximate entropy of time series.

    Parameters
    ----------
    U : time-series list

        Example
        -------
        [85, 80, 89, 85, 80, 89]

    m : int
        Comparation period

    r : int/float
        Irregularity sensitivity. High sensitivity reduces entropy.

    Output
    ------
    approximate entropy : float
    
    Example
    -------
    >>> U = np.array([85, 80, 89] * 17)
    >>> ApEn(U, 2, 3)
    1.0996541105257052e-05
    """
    import numpy as np

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m) - _phi(m + 1))

def smooth(y, box_pts):
    """Sliding box smoothening of noisy data.

    Parameters
    ----------
    y : list
        Noisy y-variable. Must be sorted wrt. time.

    box_pts : int
        Convolution box size. The greater the box the smoother the plot.

    Output
    ------
    y_smooth : list
        Smooth points to replace y. Same dimensions as y.

    Example
    -------
    x = np.linspace(0,2*np.pi,100)
    y = np.sin(x) + np.random.random(100) * 0.8

    plt.plot(x, y,'o')
    plt.plot(x, smooth(y, 18), 'r-', lw=2)
    plt.plot(x, smooth(y,9), 'g-', lw=2)
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def transpose_3d():
    """This is an example to show that the transpose of a 3d array is just the 2d-array-
    wise transpose of every j'th slice.
    """
    tmp = np.random.random((10, 10, 10))

    a = tmp.T
    b = np.empty(tmp.shape)
    for j in range(tmp.shape[1]):
        b[:, j, :] = tmp[:, j, :].T

    print np.all(a == b)

def to_frac(fval):
    """Take any float with trailing decimals and reduce it to its closest exact form."""
    def simplify_fraction(numer, denom):
        def _gcd(a, b):
            """Calculate the Greatest Common Divisor of a and b.

                Unless b==0, the result will have the same sign as b (so that when
                b is divided by it, the result comes out positive).
                """
            while b:
                a, b = b, a % b
            return a

        if denom == 0:
            return "Division by 0 - result undefined"

        # Remove greatest common divisor:
        common_divisor = _gcd(numer, denom)
        (reduced_num, reduced_den) = (numer / common_divisor, denom / common_divisor)
        # Note that reduced_den > 0 as documented in the gcd function.

        if reduced_den == 1:
            return "%d / %d is simplified to %d" % (numer, denom, reduced_num)
        elif common_divisor == 1:
            return "%d / %d is most simple fraction" % (numer, denom)
        else:
            return "%d / %d is simplified to %d/%d" % (numer, denom, reduced_num, reduced_den)
        
    diff = 1
    div = 0.0
    while diff > 0.001:
        div += 1
        diff = (fval * div) % 1.0
    
    return simplify_fraction(int(fval*div), int(div))

def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.
    
    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """
    
    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)
    
    M = 0.5 * (P + Q)
    
    return 0.5 * (_kldiv(P, M) + _kldiv(Q, M))

def randomize_by_edge_swaps(G, num_iterations):
    """Randomizes the graph by swapping edges in such a way that
    preserves the degree distribution of the original graph.

    Source: https://gist.github.com/gotgenes/2770023
    """
    newgraph = G.copy()
    edge_list = newgraph.edges()
    num_edges = len(edge_list)
    total_iterations = num_edges * num_iterations

    for i in xrange(total_iterations):
        rand_index1 = int(round(random.random() * (num_edges - 1)))
        rand_index2 = int(round(random.random() * (num_edges - 1)))
        original_edge1 = edge_list[rand_index1]
        original_edge2 = edge_list[rand_index2]
        head1, tail1 = original_edge1
        head2, tail2 = original_edge2

        # Flip a coin to see if we should swap head1 and tail1 for
        # the connections
        if random.random() >= 0.5:
            head1, tail1 = tail1, head1

        if head1 == tail2 or head2 == tail1:
            continue

        if newgraph.has_edge(head1, tail2) or newgraph.has_edge(
                head2, tail1):
            continue

        # Suceeded checks, perform the swap
        original_edge1_data = newgraph[head1][tail1]
        original_edge2_data = newgraph[head2][tail2]

        newgraph.remove_edges_from((original_edge1, original_edge2))

        new_edge1 = (head1, tail2, original_edge1_data)
        new_edge2 = (head2, tail1, original_edge2_data)
        newgraph.add_edges_from((new_edge1, new_edge2))

        # Now update the entries at the indices randomly selected
        edge_list[rand_index1] = (head1, tail2)
        edge_list[rand_index2] = (head2, tail1)

    assert len(newgraph.edges()) == num_edges
    return newgraph

def display_cmap_color_range(cmap_style='rainbow'):
    """Display the range of colors offered by a cmap.
    """
    cmap = plt.get_cmap(cmap_style)
    for c in range(256):
        plt.scatter([c], [0], s=500, c=cmap(c), lw=0)
    plt.show()

class cmap_in_range:
    """Create map to range of colors inside given domain.

    Example
    -------
    >>> cmap = cmap_in_range([0, 100])
    >>> cmap(10)
    (0.30392156862745101, 0.30315267411304353, 0.98816547208125938, 1.0)
    """
    def __init__(self, cmap_domain, cmap_range=[0, 1], cmap_style='rainbow'):
        self.cmap_range = cmap_range
        self.m = interp1d(cmap_domain, cmap_range)
        self.cmap = plt.get_cmap(cmap_style)
        
    def __call__(self, value):
        if not self.cmap_domain[0] <= value <= self.cmap_domain[1]:
            raise Exception("Value must be inside cmap_domain.")
        return self.cmap(self.m(value))

def default_to_regular(d):
    """Recursively convert nested defaultdicts to nested dicts.

    Source: http://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d

def n_choose_k(n, k):
    return math.factorial(n) * 1.0 / (math.factorial(k) * math.factorial(n - k))


def plot_step(x, y, ax=None, **kwargs):
    """Plot step function from x and y coordinates."""
    x_step = [x[0]] + [_x for tup in zip(x, x)[1:] for _x in tup]
    y_step = [_y for tup in zip(y, y)[:-1] for _y in tup] + [y[-1]]
    if ax is None:
        plt.plot(x_step, y_step, **kwargs)
    else:
        ax.plot(x_step, y_step, **kwargs)

def chunks(l, n):
    """Yield successive n-sized chunks from a list l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def point_inside_polygon(x,y,poly):
    """Determine if points x and y are inside poly.
    Source: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def shuffle_arr(arr):
    """Non-inline shuffle of list. Sets and tuples are returned as lists.
    """
    _arr = list(arr)[:]
    shuffle(_arr)
    return _arr

def shuffle_list(l):
    """Non-inline list shuffle.

    Input
    -----
    l : list

    Output
    ------
    out : list
    """
    l_out = list(l)[:]
    shuffle(l_out)
    return l_out

def draw(G, partition=False, colormap='rainbow'):
    """Draw graph G in my standard style.

    Input
    -----
    G : networkx graph
    partition : bool
    """

    def shuffle_list(l):
        l_out = list(l)[:]
        shuffle(l_out)
        return l_out
    
    def _get_cols(partition):
        return dict(
            zip(
                shuffle_list(set(partition.values())),
                np.linspace(0, 256, len(set(partition.values()))).astype(int)
            )
        )

    cmap = plt.get_cmap(colormap)
    if partition == True:
        partition = community.best_partition(G)
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) is dict and len(partition) == len(G.nodes()):
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) in [list, tuple] and len(partition) == len(G.nodes()):
        colors = list(partition)
    else:
        try:
            colors = [cmap(n[1]['node_color']) for n in G.nodes(data=True)]
        except KeyError:
            # nodes do not have node_color attribute
            colors = "grey"
    
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    nx.draw_networkx_edges(G, pos=pos, width=2, alpha=.3, zorder=-10)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, alpha=1, linewidths=0, node_color=colors)
    #nx.draw_networkx_labels(G, pos=pos, font_color="red")
    plt.axis("off")

def unwrap(l):
    """Unwrap a list of lists to a single list.

    Input
    -----
    l : list of lists

    Output
    ------
    out : list
    """
    return [v for t in l for v in t]

def hinton(matrix, max_weight=None, ax=None, facecolor='#ecf0f1', color_pos='#3498db', color_neg='#d35400'):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))
    
    ax.patch.set_facecolor(facecolor)
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = color_pos if w > 0 else color_neg
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def jaccard_mutual_information(p1, p2):
    """Intersection weighted average over jaccard similarities between communities in two partitions.

    This is an alternative to NMI with a slightly simpler interpretation

    Input
    -----
    p1 : dict or list
        A partition formatted as {node: community, ...}
    p2 : dict or list

    Output
    ------
    out : float
        Value between 0 and 1
    """
    if type(p1) in [list, np.ndarray]:
        p1 = dict(zip(range(len(p1)), p1))
    if type(p2) in [list, np.ndarray]:
        p2 = dict(zip(range(len(p2)), p2))
    if set(p1.keys()) != set(p2.keys()):
        raise Exception("p1 and p2 does not have the same nodes")
    N = len(p1)
    
    # Invert partition datastructures to # {cluster: [node, node, ...], cluster: [node, node, ...]}
    p1_inv = defaultdict(list)
    p2_inv = defaultdict(list)
    for n, c in p1.items():
        p1_inv[c].append(n)
    for n, c in p2.items():
        p2_inv[c].append(n)

    # Compute average weighted jaccard similarity
    J = 0
    for ci, ni in p1_inv.items():
        for cj, nj in p2_inv.items():
            n_ij = len(set(ni) & set(nj))
            A_inter_B = len(set(ni) & set(nj))
            A_union_B = len(set(ni) | set(nj))
            J += (n_ij * 1.0 / N) * (A_inter_B * 1.0 / A_union_B)
            
    return J


def graph_list_to_pajek(G_list):
    """Convert list of graphs to multilayer pajek string
    
    Input
    -----
    G_list : list
        Networkx graphs
    
    Output
    ------
    out : str
        Pajek filestring in *Intra format
    """
    def _write_pajek(A, node_labels=None, index_from=0):
        """Return multiplex representation of multiplex network adjacency matrix A

        Providing an adjacency tensor where A[:, :, k] is adjacency matrix of temporal
        layer k, return a pajek format representation of the temporal network which weights interlayer
        edges by state node neighborhood similarity. 

        Parameters
        ----------
        A : numpy.3darray
            3d tensor where each A[:, :, k] is a layer adjacency matrix
        max_trans_prob : float/str
            Cap on interlayer edge weights. 'square' for square penalty.
        power_penalty : int/float
            Power to jaccard similarity betw. state nodes to penalize low similarity
        index_from : int
            From which number to index nodes and layers in pajek format from
        style : bool
            Either 'zigzag', 'vertical', or 'simple'. 'vertical' will give working results but is
            essentially wrong use of Infomap, 'simple' should be possible to use in Infomap but is not
            at this point, so 'zigzag' is preferred because it is an explicit representation of the way
            the network should be represented internally in Infomap.

        Returns
        -------
        out_file : string
            A network string in multiplex format
        intid_to_origid : dict
            Key-value pairs of node integer id and original id
        origid_to_intid : dict
            Reverse of intid_to_origid
        """

        def _write_outfile(A):
            """Write nodes and intra/inter-edges from A and J to string."""
            def __remove_symmetry_A(A):
                A_triu = defaultdict(int)
                for (i, j, k), w in A.items():
                    if j > i:
                        A_triu[(i, j, k)] = w
                return A_triu
            def __write_nodes(outfile):
                outfile += "*Vertices %d" % Nn
                for nid, label in enumerate(nodes):
                    outfile += '\n%d "%s" 1.0' % (nid + index_from, str(label))
                return outfile
            def __write_intra_edges(outfile):
                outfile += "\n*Intra\n# layer node node [weight]"
                for (i, j, k), w in __remove_symmetry_A(A).items():
                    outfile += '\n%d %d %d %f' % (
                        k + index_from,  # layer
                        nodemap[i] + index_from,  # node
                        nodemap[j] + index_from,  # node
                        w                # weight
                    )
                return outfile

            outfile = ""
            outfile = __write_nodes(outfile)
            outfile = __write_intra_edges(outfile)

            return outfile

        nodes = sorted(set([n for i, j, _ in A.keys() for n in [i, j]]))
        Nn = len(nodes)
        Nl = len(set([k for i, j, k in A.keys()]))

        nodemap = dict(zip(nodes, range(Nn)))

        return _write_outfile(A)

    def _create_adjacency_matrix(layer_edges):
        """Return 3d adjacency matrix of the temporal network.
        
        Input
        -----
        layer_edges : dict
        
        Output
        ------
        A : dict
        """
        A = defaultdict(int)
        for l, edges in layer_edges.items():
            for edge in edges:
                    A[(edge[0], edge[1], l)] += 1
                    A[(edge[1], edge[0], l)] += 1    
        return A
    
    return _write_pajek(
        _create_adjacency_matrix(
            dict(zip(range(len(G_list)), [G.edges() for G in G_list]))
        )
    )

def invert_partition(partition):
    """Invert a dictionary representation of a graph partition.

    Inverts a dictionary representation of a graph partition from nodes -> communities
    to communities -> lists of nodes, or the other way around.
    """
    if type(partition.items()[0][1]) is list:
        partition_inv = dict()
        for c, nodes in partition.items():
            for n in nodes:
                partition_inv[n] = c
    else:
        partition_inv = defaultdict(list)
        for n, c in partition.items():
            partition_inv[c].append(n)
    return ulf.default_to_regular(partition_inv)


# Non importable (examples)
class how_to_unittest:
    def unittest(f):    
        from functools import wraps
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Test something with the args, kwargs and the wrapped function f
            return f(*args, **kwargs)
        return wrapper

    @unittest
    def f(arg1, arg2, kwarg1=1, kwarg2=2):
        # A function that does something
        return G_copy
# Non-python stuff
"""rm build/Infomap/infomap/MultiplexNetwork.o; make; ./Infomap foursquares.net ../output/ -i multiplex --multiplex-js-relax-rate 0.05 --overlapping --expanded --pajek -z"""

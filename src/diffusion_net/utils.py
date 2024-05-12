import sys
import os
import time

import torch
import hashlib
import numpy as np
import scipy
import plyfile

import matplotlib.pyplot as plt
import math
import sys
if sys.version_info >= (3,0):
    from functools import reduce
from plyfile import PlyData, PlyElement

def save_to_ply(filename, xyz, colors):
    # Ensure that the shapes of xyz and colors match
    assert xyz.shape[0] == colors.shape[0], "Number of points must match between xyz and colors"
    
    # Create a PlyElement for vertices
    vertices = np.core.records.fromarrays([xyz[:, 0], xyz[:, 1], xyz[:, 2], colors[:, 0], colors[:, 1], colors[:, 2]], names='x, y, z, red, green, blue')
    vertex_element = PlyElement.describe(vertices, 'vertex')
    
    # Save to PLY file
    PlyData([vertex_element], text=True).write(filename)



def save_ply_attr(points, filename, colors=None, normals=None, attr = None):
    vertex = np.core.records.fromarrays(points.transpose(1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(1, 0), names='red, green, blue, alpha',
                                                      formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(1, 0), names='red, green, blue',
                                                      formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr
    
    attr = np.reshape(attr, (len(attr), 1))
    if attr is not None:
        vertex_attr = np.core.records.fromarrays(attr.transpose(1, 0), names='attr', formats='f4')
        assert len(vertex_attr) == num_vertex
        desc = desc + vertex_attr.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]
    
 
    if attr is not None:
        for prop in vertex_attr.dtype.names:
            vertex_all[prop] = vertex_attr[prop]

    ply = PlyData(
        [PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)

def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles
    Examples
   
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


from matplotlib.colors import ListedColormap

def draw_point_cloud(input_points, input_attributes, canvasSize=500, space=200, diameter=20,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 2, 1], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
            attributes: Array of attributes with shape (num_points,)
        Output:
            colored image as numpy array of size canvasSizexcanvasSize
    """
    # Initialize image with white background
    #image = np.ones((canvasSize, canvasSize, 3), dtype=np.uint8) * 255

    image = np.zeros((canvasSize, canvasSize, 3))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    attributes = input_attributes

    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    attributes = attributes[zorder]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    unique_attributes = np.unique(attributes)  # Get unique attribute values
    num_unique_attributes = len(unique_attributes)
    cmap = plt.cm.get_cmap('viridis', num_unique_attributes)  # Get a colormap with same number of colors as unique attributes

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        color = cmap(unique_attributes.tolist().index(attributes[j]))[:3]  # Get color from colormap based on attribute
        image[px, py, :] = image[px, py, :] * 0.7 + dv[:, np.newaxis] * (max_depth - points[j, 2]) * 0.3 * color

    image = image / np.max(image)
    return image

def point_cloud_three_views(points, attributes, save_path=None):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array color image of size 500x1500.
        If save_path is provided, save the image as the given path.
    """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, attributes, zrot=0, xrot=0, yrot=0)
    img2 = draw_point_cloud(points, attributes, zrot=0, xrot=90 / 180.0 * np.pi, yrot=0)
    img3 = draw_point_cloud(points, attributes, zrot=90 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0)
    image_large = np.concatenate([img1, img2, img3], 1)
    
    if save_path:
        plt.imsave(save_path, image_large)
    
    return image_large



# == read some ply files
def read_ply(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'], loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'], loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    if 'attr' in loaded['vertex'].data.dtype.names:
        labels = loaded['vertex'].data['attr']
        labels = np.reshape(labels, (1, len(labels)))
        points = np.concatenate([points, labels], axis=0)
    
    if 'scalar_attr' in loaded['vertex'].data.dtype.names:
        labels = loaded['vertex'].data['scalar_attr']
        labels = np.reshape(labels, (1, len(labels)))
        points = np.concatenate([points, labels], axis=0)

    points = points.transpose(1, 0)
    return points


# == Pytorch things

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

def label_smoothing_log_loss(pred, labels, smoothing=0.0):
    n_class = pred.shape[-1]
    one_hot = torch.zeros_like(pred)
    one_hot[labels] = 1.
    one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
    loss = -(one_hot * pred).sum(dim=-1).mean()
    return loss


# Randomly rotate points.
# Torch in, torch out
# Note fornow, builds rotation matrix on CPU. 
def random_rotate_points(pts, randgen=None):
    R = random_rotation_matrix(randgen) 
    R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
    return torch.matmul(pts, R) 

def random_rotate_points_y(pts):
    angles = torch.rand(1, device=pts.device, dtype=pts.dtype) * (2. * np.pi)
    rot_mats = torch.zeros(3, 3, device=pts.device, dtype=pts.dtype)
    rot_mats[0,0] = torch.cos(angles)
    rot_mats[0,2] = torch.sin(angles)
    rot_mats[2,0] = -torch.sin(angles)
    rot_mats[2,2] = torch.cos(angles)
    rot_mats[1,1] = 1.

    pts = torch.matmul(pts, rot_mats)
    return pts

# Numpy things

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()

def random_rotation_matrix(randgen=None):
    """
    Creates a random rotation matrix.
    randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
    """
    # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randgen is None:
        randgen = np.random.RandomState()
        
    theta, phi, z = tuple(randgen.rand(3).tolist())
    
    theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0 # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# Python string/file utilities
def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

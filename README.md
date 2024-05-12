**DiffusionNet** is a general-purpose method for deep learning on surfaces such as 3D triangle meshes and point clouds. It is well-suited for tasks like segmentation, classification, feature extraction, etc.

DiffusionNet 

## Outline

  - `diffusion_net/src` implementation of the method, including preprocessing, layers, etc
  - `rna_mesh_segmentation` examples and scripts to reproduce experiments 


## Prerequisites

DiffusionNet depends on pytorch, as well as a handful of other fairly typical numerical packages. 

The code assumes a GPU with CUDA support. DiffusionNet has minimal memory requirements; >4GB GPU memory should be sufficient. 

## Applying the network to your task

```python
import diffusion_net

# Here we use Nx3 positions as features. Any other features you might have will work!
# See our experiments for the use of of HKS features, which are naturally 
# invariant to (isometric) deformations.
C_in = 3

# Output dimension (e.g., for a 10-class segmentation problem)
C_out = 10 

# Create the model
model = diffusion_net.layers.DeltaDiffusionNet(C_in=C_in,
                            C_out=n_class,
                            C_width=128, 
                            N_block=4, 
                            last_activation=lambda x : torch.nn.functional.log_softmax(x,dim=-1),
                            #last_activation = None,
                            outputs_at='vertices', 
                            dropout=True)

# An example epoch loop.
# For a dataloader example see experiments/human_segmentation_original/human_segmentation_original_dataset.py
for sample in your_dataset:
    
    verts = sample.vertices  # (Vx3 array of vertices)
    faces = sample.faces     # (Fx3 array of faces, None for point cloud) 
    
    # center and unit scale
    verts = diffusion_net.geometry.normalize_positions(verts)
    
  
    outputs = model(features, verts, mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
    
    # Now do whatever you want! Apply your favorite loss function, 
    # backpropgate with loss.backward() to train the DiffusionNet, etc. 
```

### Tips and Tricks

By default, DeltaDiffNet uses _spectral acceleration_ for fast performance, which requires some CPU-based precomputation to compute operators & eigendecompositions for each input, which can take a few seconds for moderately sized inputs. DeltaDiffNet will be fastest if this precomputation only needs to be performed once for the dataset, rather than for each input. 

### Thanks
The dataset loaders mimic code from [HSN](https://github.com/rubenwiersma/hsn), [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric), and probably indirectly from [Deltaconv](https://github.com/rubenwiersma/deltaconv). Thank you!

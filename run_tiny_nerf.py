import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import math, sys
from time import sleep

from torch import nn, optim

matplotlib.rc('text', usetex=True) #use latex for text
# add amsmath to the preamble
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"


def get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os):
    u_is_c = torch.rand(*list(ds.shape[:2]) + [N_c]).to(ds) #   random noise to intervals inside frustum 
    #print(f'u_is_c.shape : {u_is_c.shape}');  exit(0)   #   [100, 100, 32]
    #print(f'u_is_c[0, 0] : {u_is_c[0, 0]}');  exit(0)   #   [0.16, 0.88, 0.34, ... 0.92, 0.59, 0.10]
    t_is_c = t_i_c_bin_edges + u_is_c * t_i_c_gap       #   intervals with noise inside frustum for each ray thru pixel
    #print(f't_is_c.shape : {t_is_c.shape}');  exit(0)   #   [100, 100, 32]
    #t0 = t_is_c[..., :, None];  print(f't0.shape : {t0.shape}');    #      [100, 100, 32, 1]
    #t1 = os[..., None, :];  print(f't1.shape : {t1.shape}');    exit(0); #  [100, 100, 1, 3]
    #t2 = ds[..., None, :];  print(f't2.shape : {t2.shape}');    exit(0); #  [100, 100, 1, 3]
    r_ts_c = os[..., None, :] + t_is_c[..., :, None] * ds[..., None, :] #   positions of points randomly sampled along the ray thru each pixel in object coordinate systems.  
    #print(f'r_ts_c.shape : {r_ts_c.shape}');  exit(0)   #   [100, 100, 32, 3]
    return (r_ts_c, t_is_c)


def render_radiance_volume(r_ts, ds, chunk_size, F, t_is):
    #print(f'r_ts.shape : {r_ts.shape}');  #exit(0)   #   [100, 100, 32, 3]         #   position of points
    r_ts_flat = r_ts.reshape((-1, 3))
    #print(f'r_ts_flat.shape : {r_ts_flat.shape}');  #exit(0)   #   [320000, 3]     #   flattened position of points
    #print(f'ds.shape : {ds.shape}');  #exit(0)   #   [100, 100, 3]
    ds_rep = ds.unsqueeze(2).repeat(1, 1, r_ts.shape[-2], 1)
    #print(f'ds_rep.shape : {ds_rep.shape}');  exit(0)   #   [100, 100, 32, 3]       #   direction of points
    ds_flat = ds_rep.reshape((-1, 3))                   #   [320000, 3]             #   flattened direction of points
    c_is = []
    sigma_is = []
    for chunk_start in range(0, r_ts_flat.shape[0], chunk_size):
        r_ts_batch = r_ts_flat[chunk_start : chunk_start + chunk_size]
        ds_batch = ds_flat[chunk_start : chunk_start + chunk_size]
        #print(f'r_ts_batch.shape : {r_ts_batch.shape}');  #exit(0)  #   [16384, 3]    
        #print(f'ds_batch.shape : {ds_batch.shape}');  exit(0)       #   [16384, 3]    
        preds = F(r_ts_batch, ds_batch)
        '''
        for key, val in preds.items():
            print(f'key : {key}')   
            #   c_is : color
            #   sigma_is : density
        exit(0);  
        '''
        #print(f'preds["c_is"].shape : {preds["c_is"].shape}')                   #   [16383, 3]
        #print(f'preds["sigma_is"].shape : {preds["sigma_is"].shape}');  exit(0);#   [16384]
        c_is.append(preds["c_is"])
        sigma_is.append(preds["sigma_is"])

    c_is = torch.cat(c_is).reshape(r_ts.shape)
    sigma_is = torch.cat(sigma_is).reshape(r_ts.shape[:-1])

    #print(f'c_is.shape : {c_is.shape}')                         #   [100, 100, 32, 3]
    #print(f'sigma_is.shape : {sigma_is.shape}');    exit(0);    #   [100, 100, 32]
    delta_is = t_is[..., 1:] - t_is[..., :-1]
    #print(f'delta_is.shape : {delta_is.shape}');    exit(0);    #   [100, 100, 31]
    #print(f'delta_is[0, 0] : {delta_is[0, 0]}');    #exit(0);    #   [0.16, 0.04, 0.08, ... 0.14, 0.06, 0.04]

    one_e_10 = torch.Tensor([1e10]).expand(delta_is[..., :1].shape)
    #print(f'one_e_10.shape : {one_e_10.shape}');    exit(0);    #   [100, 100, 1]
    delta_is = torch.cat([delta_is, one_e_10.to(delta_is)], dim=-1)
    #print(f'delta_is.shape : {delta_is.shape}');    exit(0);    #   [100, 100, 32]
    #print(f'delta_is[0, 0] : {delta_is[0, 0]}');    exit(0);     #   [0.16, 0.04, 0.08, ... 0.14, 0.06, 0.04, 1.0e10]
    delta_is = delta_is * ds.norm(dim=-1).unsqueeze(-1)
    #print(f'delta_is.shape : {delta_is.shape}');    #exit(0);     #   [100, 100, 32]
    #print(f'delta_is[0, 0] : {delta_is[0, 0]}');    exit(0);     #   [0.19, 0.05, 0.1, ... 0.17, 0.07, 0.05, 1.2e10]

    alpha_is = 1.0 - torch.exp(-sigma_is * delta_is)
    #print(f'alpha_is.shape : {alpha_is.shape}');    exit(0);     #   [100, 100, 32]

    T_is = torch.cumprod(1.0 - alpha_is + 1e-10, -1)
    #print(f'T_is.shape : {T_is.shape}');    #exit(0);     #   [100, 100, 32]
    T_is = torch.roll(T_is, 1, -1)
    #print(f'T_is.shape : {T_is.shape}');    #exit(0);     #   [100, 100, 32]
    T_is[..., 0] = 1.0

    w_is = T_is * alpha_is

    t0 = w_is[..., None] * c_is;    print(f't0.shape : {t0.shape}');    #exit(0);     #   [100, 100, 32]
    C_rs = (w_is[..., None] * c_is).sum(dim=-2)
    print(f'C_rs.shape : {C_rs.shape}');    exit(0);     #   [100, 100, 32]

    return C_rs


def run_one_iter_of_tiny_nerf(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, chunk_size, F_c):
    (r_ts_c, t_is_c) = get_coarse_query_points(ds, N_c, t_i_c_bin_edges, t_i_c_gap, os)
    C_rs_c = render_radiance_volume(r_ts_c, ds, chunk_size, F_c, t_is_c)
    return C_rs_c


class VeryTinyNeRFMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L_pos = 6
        self.L_dir = 4
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        net_width = 256
        self.early_mlp = nn.Sequential(
            nn.Linear(pos_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width + 1),
            nn.ReLU(),
        )
        self.late_mlp = nn.Sequential(
            nn.Linear(net_width + dir_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3),
            nn.Sigmoid(),
        )

    def forward(self, xs, ds):
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2**l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2**l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2**l_dir * torch.pi * ds))
            ds_encoded.append(torch.cos(2**l_dir * torch.pi * ds))

        ds_encoded = torch.cat(ds_encoded, dim=-1)

        outputs = self.early_mlp(xs_encoded)
        sigma_is = outputs[:, 0]
        c_is = self.late_mlp(torch.cat([outputs[:, 1:], ds_encoded], dim=-1))
        return {"c_is": c_is, "sigma_is": sigma_is}


def main():
    seed = 9458
    torch.manual_seed(seed)
    np.random.seed(seed)
    '''
    num_iters = 20000
    display_every = 100
    '''
    num_iters = 200000
    display_every = 1000


    #num_iters = 20
    #display_every = 3

    i_last_display = int(num_iters / display_every) * display_every
    #print(f'num_iters : {num_iters}, display_every : {display_every}, i_last_display : {i_last_display}');  exit(0)



    device = "cuda:0"
    F_c = VeryTinyNeRFMLP().to(device)
    chunk_size = 16384

    lr = 5e-3
    optimizer = optim.Adam(F_c.parameters(), lr=lr)
    criterion = nn.MSELoss()

    data_f = "66bdbc812bd0a196e194052f3f12cb2e.npz"
    data = np.load(data_f)

    #for key, val in data.items():
    #    print(f'key : {key}')       #   images, poses, focal, camera_distance
    #exit(0)                  #   98.209

    images = data["images"] / 255
    n_im_total = len(images)
    #print(f'n_im_total : {n_im_total}'); exit(0)       #   800
    #print(f'images.shape : {images.shape}'); exit(0)   #   (800, 100, 100, 3)
    img_size = images.shape[1]                          #   100
    xs = torch.arange(img_size) - (img_size / 2 - 0.5)  
    #print(f'xs : {xs}');    exit(0)                     #   -49.5 -48.5 ... -0.5, 0.5, ... 48.5, 49.5
    ys = torch.arange(img_size) - (img_size / 2 - 0.5)
    #print(f'ys : {ys}');    exit(0)                     #   -49.5 -48.5 ... -0.5, 0.5, ... 48.5, 49.5
    (xs, ys) = torch.meshgrid(xs, -ys, indexing="xy")   #   this is because y is positive at the top and negative at the bottom of a image.
    #print(f'xs.shape : {xs.shape}');    #exit(0)        #   [100, 100]
    #print(f'xs : {xs}');    exit(0)                     
    '''
    [[-49.5, -48.5, ... 48.5, 49.5],  
     [-49.5, -48.5, ... 48.5, 49.5],  
     ...
     [-49.5, -48.5, ... 48.5, 49.5],  
     [-49.5, -48.5, ... 48.5, 49.5]]
    '''
    #print(f'ys : {ys}');    exit(0)         
    '''
    [[49.5, 49.5, ... 49.5, 49.5],  
     [48.5, 48.5, ... 48.5, 48.5],  
     ...
     [-48.5, -48.5, ... -48.5, -48.5],  
     [-49.5, -49.5, ... -49.5, -49.5]]
    '''
    focal = float(data["focal"])
#   print(f'focal : {focal}'); exit(0)                  #   98.209
    #print(f"data['camera_distance'] : {data['camera_distance']}"); exit(0)          #2.25    
    pixel_coords = torch.stack([xs, ys, torch.full_like(xs, -focal)], dim=-1)
    #print(f"pixel_coords.shape : {pixel_coords.shape}"); exit(0)    #   [100, 100, 3]
    #print(f"pixel_coords[:, :, -1] : \n{pixel_coords[:, :, -1]}"); exit(0)    #   This is because the coordinate is right-handed and x axis is toward left, y axis is toward up, z axis is toward out of screen.
    '''
    [[-98.2, -98.2, ... -98.2, -98.2],  
     [-98.2, -98.2, ... -98.2, -98.2],  
     ...
     [-98.2, -98.2, ... -98.2, -98.2],  
     [-98.2, -98.2, ... -98.2, -98.2]]
    '''
    camera_coords = pixel_coords / focal    #   This makes the focal length as the unit of distance.  Now the 3D coordinate of pixel is in the form of (x, y, -1) since it is on the plane far from camera by the focal length in the negative z direction.
    #print(f"camera_coords[:, :, 0] : \n{camera_coords[:, :, 0]}"); exit(0) 
    #print(f"camera_coords[:, :, 1] : \n{camera_coords[:, :, 1]}"); exit(0) 
    #print(f"camera_coords[:, :, 2] : \n{camera_coords[:, :, 2]}"); exit(0) 
    init_ds = camera_coords.to(device)
    #print(f"init_ds.shape : {init_ds.shape}"); exit(0)  #   [100, 100, 3] 
    cam_dist = float(data["camera_distance"])
    init_o = torch.Tensor(np.array([0, 0, cam_dist])).to(device)
    #init_o = torch.Tensor(np.array([0, 0, float(data["camera_distance"])])).to(device)

    test_idx = 150
    '''
    plt.imshow(images[test_idx]);    plt.show()
    '''

    test_img = torch.Tensor(images[test_idx]).to(device)
    poses = data["poses"]
    test_R = torch.Tensor(poses[test_idx, :3, :3]).to(device)
    #print(f'poses[test_idx, :3, :3] : \n{poses[test_idx, :3, :3]}');    exit(0)
    '''
    [[-0.134, 0.549, -0.824]
     [0     , 0.832, 0.554]
     [0.99  , 0.074, -0.11]]
    '''
    
    #print(f"init_ds[:, :, 0] : \n{init_ds[:, :, 0]}"); exit(0) 
    '''
    [[-0.50, -0.49, ... 0.49, 0.50],  
     [-0.50, -0.49, ... 0.49, 0.50],  
     ...
     [-0.50, -0.49, ... 0.49, 0.50],  
     [-0.50, -0.49, ... 0.49, 0.50]]
    '''
 
    #print(f"init_ds[:, :, 1] : \n{init_ds[:, :, 1]}"); exit(0) 
    '''
    [[0.50, 0.50, ... 0.50, 0.50],  
     [0.49, 0.49, ... 0.49, 0.49],  
     ...
     [-0.49, -0.49, ... -0.49, -0.49],  
     [-0.50, -0.50, ... -0.50, -0.50]]
    '''
    #print(f"init_ds[:, :, 2] : \n{init_ds[:, :, 2]}"); exit(0) 
    '''
    [[-1, -1, ... -1, -1],  
     [-1, -1, ... -1, -1],  
     ...
     [-1, -1, ... -1, -1],  
     [-1, -1, ... -1, -1]]
    '''
    test_ds = torch.einsum("ij,hwj->hwi", test_R, init_ds)      #   Rotate point in camera coordinate system to get the postion in object coordinate system.  
    #print(f"test_ds.shape : {test_ds.shape}"); exit(0)         #   [100, 100, 3] 
    #print(f"test_R.shape : {test_R.shape}"); #exit(0)           #   [3, 3]    
    #print(f"init_ds[0, 0].shape : {init_ds[0, 0].shape}");      #   [3] 
    #print(f"torch.matmul(test_R, init_ds[0, 0]) : \n{torch.matmul(test_R, init_ds[0, 0])}"); # [1.169, -0.134, -0.350] 
    #print(f"init_ds[0, 0] : \n{init_ds[0, 0]}"); #exit(0)       #   [-0.50, 0.50, -1] 
    #print(f"test_ds[0, 0] : \n{test_ds[0, 0]}"); exit(0)        #   [1.169, -0.134, -0.350]

    #print(f"init_o.shape : {init_o.shape}"); exit(0)                #   [3] 
    #print(f"init_o : {init_o}"); exit(0)                            #   [0, 0, 2.25] 
    test_os = (test_R @ init_o).expand(test_ds.shape)
    #print(f"test_os.shape : {test_os.shape}"); exit(0)               #  [100, 100, 3] 
    #print(f"test_os[0, 0] : {test_os[0, 0]}"); exit(0)               #  [-1.85, 1.24, -0.25] 

    #print(f'poses.shape : {poses.shape}');  exit(0)     #   (800, 4, 4)
    #'''
    #n_im = min(20, n_im_total)
    n_im = min(1, n_im_total)
    for iI in range(n_im):
        #print(f'iI : {iI} / {n_im_total}')
        #print(f'poses[iI] : {poses[iI]}')
        translation_norm = math.sqrt(poses[iI, 0, 3] * poses[iI, 0, 3] + poses[iI, 1, 3] * poses[iI, 1, 3] + poses[iI, 2, 3] * poses[iI, 2, 3]) 
        plt.title(f'iI : {iI} / {n_im}')
        plt.imshow(images[iI]); 
        plt.text(1, 31,
        r'\['
        r'\begin{bmatrix}' 
        r'' + '{:.2f}'.format(poses[iI, 0, 0]) + '&' + '{:.2f}'.format(poses[iI, 0, 1]) + '&' + '{:.2f}'.format(poses[iI, 0, 2]) + '&' + '{:.2f}'.format(poses[iI, 0, 3]) + r'\\'
        r'' + '{:.2f}'.format(poses[iI, 1, 0]) + '&' + '{:.2f}'.format(poses[iI, 1, 1]) + '&' + '{:.2f}'.format(poses[iI, 1, 2]) + '&' + '{:.2f}'.format(poses[iI, 1, 3]) + r'\\'
        r'' + '{:.2f}'.format(poses[iI, 2, 0]) + '&' + '{:.2f}'.format(poses[iI, 2, 1]) + '&' + '{:.2f}'.format(poses[iI, 2, 2]) + '&' + '{:.2f}'.format(poses[iI, 2, 3]) + r'\\'
        r'' + '{:.2f}'.format(poses[iI, 3, 0]) + '&' + '{:.2f}'.format(poses[iI, 3, 1]) + '&' + '{:.2f}'.format(poses[iI, 3, 2]) + '&' + '{:.2f}'.format(poses[iI, 3, 3]) + 
        r'\end{bmatrix}' 
        r'\]' + '\ncamera_distance : {}'.format(cam_dist) + '\ntranslation_norm : {:.2f}'.format(translation_norm)        
        , size = 12)
        plt.pause(0.05)
        sys.stdout.write('{} / {}\r'.format(iI, n_im));  sys.stdout.flush();
        if n_im - 1 != iI:
            plt.clf()
    #plt.show()
    #'''

    t_n = 1.0
    t_f = 4.0
    N_c = 32
    t_i_c_gap = (t_f - t_n) / N_c
    #print(f't_i_c_gap : {t_i_c_gap}');  exit(0)   #   0.09375
    t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)
    #print(f't_i_c_bin_edges : \n{t_i_c_bin_edges}');  exit(0)   #   [1.0, 1.09, 1.19, ... 3.71, 3.81, 3.90]
    #train_idxs = np.arange(len(images)) != test_idx
    train_idxs = np.arange(n_im_total) != test_idx
    images = torch.Tensor(images[train_idxs])
    poses = torch.Tensor(poses[train_idxs])
    psnrs = []
    losses = []
    iternums = []


    F_c.train()
    plt.figure(figsize=(10, 4))
    for i in range(num_iters):
        target_img_idx = np.random.randint(images.shape[0])
        target_pose = poses[target_img_idx].to(device)
        R = target_pose[:3, :3]

        ds = torch.einsum("ij,hwj->hwi", R, init_ds)
        os = (R @ init_o).expand(ds.shape)

        C_rs_c = run_one_iter_of_tiny_nerf(
            ds, N_c, t_i_c_bin_edges, t_i_c_gap, os, chunk_size, F_c
        )
        loss = criterion(C_rs_c, images[target_img_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % display_every == 0:
            F_c.eval()
            with torch.no_grad():
                C_rs_c = run_one_iter_of_tiny_nerf(
                    test_ds, N_c, t_i_c_bin_edges, t_i_c_gap, test_os, chunk_size, F_c
                )

            loss = criterion(C_rs_c, test_img)
            print(f"Loss: {loss.item()}")
            psnr = -10.0 * torch.log10(loss)

            psnrs.append(psnr.item())
            losses.append(loss.item())
            iternums.append(i)

            #plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(C_rs_c.detach().cpu().numpy())
            plt.title(f"Iteration {i} / {num_iters}")
            plt.subplot(122)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.pause(0.05)
            if i_last_display != i:
                plt.clf()
            #plt.show()

            F_c.train()

    print("Done!")
    plt.show()


if __name__ == "__main__":
    main()

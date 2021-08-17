# -*- coding: utf-8 -*-

import numpy as np

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes(data, route, ax1, depot_num = 1, markersize=5, visualize_demands=False, demand_scale=1, round_demand=False, is_numpy = False, plot_legend = True,epoch=0):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    if not is_numpy:
        data = data.cpu().numpy()
        route = route.cpu().numpy()
    
    route = route.astype(int)
    # route is one sequence, separating different routes with 0 (depot)
    if depot_num == 1:
        routes = [r[r!=0] for r in np.split(route, np.where(route==0)[0]) if (r != 0).any()]
        route_depots = np.zeros(len(routes))
    else:
        route = route
        route_ends = np.where(route < depot_num)[0][1::2] 
        route_depots = route[route_ends]
        idx = []
        for e,r in enumerate(np.split(route, route_ends + 1)):
            if r.shape[0] >2:
                idx.append(e)
        route_depots = route_depots[idx]       
        routes = [r[r>depot_num-1] for r in np.split(route, route_ends + 1) if (r > depot_num-1).any()]
    
    
    
    depot = data[:depot_num,0:2]
    locs = data[depot_num:,0:2]
    demands = data[depot_num:,2] * demand_scale
    capacity = demand_scale # Capacity is always 1

    for d in range(depot_num):
        x_dep = np.atleast_1d(depot[d][0])
        y_dep = np.atleast_1d(depot[d][1])
        ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
        
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    #legend = ax1.legend(loc='upper center')
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    for veh_number, r in enumerate(routes):
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        current_depot = int(route_depots[veh_number])
        route_demands = demands[r - depot_num]
        coords = locs[r - depot_num, :]
        xs, ys = coords.transpose()
        total_route_demand = sum(route_demands)
        #assert total_route_demand <= capacity

        if not visualize_demands:
            ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_dep, y_dep = np.atleast_1d(depot[current_depot,0]), np.atleast_1d(depot[current_depot,1])
        x_prev, y_prev = x_dep, y_dep
        cum_demand = 0
        for (x, y), d in zip(coords, route_demands):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_demand / capacity))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_demand / capacity), 0.01, 0.1 * d / capacity))
            
            x_prev, y_prev = x, y
            cum_demand += d
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        
        xs = np.concatenate((x_dep,xs,x_dep))
        ys = np.concatenate((y_dep,ys,y_dep))
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='R{}, # {}, c {} / {}, d {:.2f}'.format(
                veh_number, 
                len(r), 
                int(total_route_demand) if round_demand else total_route_demand, 
                int(capacity) if round_demand else capacity,
                dist.item()
            )
        )
        
        qvs.append(qv)
        
        

        
    ax1.set_title('Episode {}. Distance {:.2f}'.format(epoch,len(routes), total_dist.item()),fontdict={'fontsize':'x-large'})
    if plot_legend:
        ax1.legend(handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')
    
    if visualize_demands:
        ax1.add_collection(pc_cap)
        ax1.add_collection(pc_used)
        ax1.add_collection(pc_dem)


import argparse
from tqdm import tqdm
import numpy as np
from cylinder import Cylinder
from torus import Torus
import open3d as o3d
import globals
from Tee import Tee
from tools_elbow import *
import queue
TOP = 0
BOTTOM = 1


# x_top=[-2.,0,0]
# x_bottom=[2,0,0]
#
# z_top=[0,0,0]
# z_bottom=[0,0,2]
# test=Tee(x_top,x_bottom,1.,z_top,z_bottom,1.)
# test.to_o3d_mesh()

def custom_sort(obj):
    return obj.get_height()


def update_cylinder_para(cylinder_idx,cylinder_side, neighbor_cylinder_index,neighbor_side, vertical_cylinder_index,vertical_side):
    used_cylinder_sides[cylinder_idx][cylinder_side] = True
    used_cylinder_sides[neighbor_cylinder_index][neighbor_side] = True
    used_cylinder_sides[vertical_cylinder_index][vertical_side] = True


def extend_intersection(cylinder_idx, cylinder_side, neighbor_cylinder_index, neighbor_side):
    # 首先判断两个圆柱是否相交
    cylinder = globals.desc_load_cylinders[cylinder_idx]
    neighbor_cylinder = globals.desc_load_cylinders[neighbor_cylinder_index]
    neighbor_cylinder_center = globals.para_top_bottom[neighbor_cylinder_index][neighbor_side]
    if cylinder.is_point_inside(neighbor_cylinder_center):
        return False
    if neighbor_cylinder.is_point_inside(globals.para_top_bottom[cylinder_idx][cylinder_side]):
        return False
    cylinder_nor = get_correct_dir(cylinder, cylinder_side)
    for step in np.arange(0, 1, 0.005):  # 这里的参数很重要，不一定要从0开始，因为会有异线交叉的情况
        extend_nor = globals.para_top_bottom[cylinder_idx][cylinder_side] + cylinder_nor * step
        if np.linalg.norm(extend_nor - neighbor_cylinder_center) < 3*args.threshold_radius:
            return True
    return False

# 判断圆柱另一端是否连接过拐弯
def in_elbow(cy_idx, side):
    center = globals.para_top_bottom[cy_idx][side]
    for torus in globals.toruses:
        if neighbor_elbow(center, torus.center_coord, torus.torus_radius + torus.radius):
            return torus
    return None

def del_part(torus):
    globals.toruses=[x for x in globals.toruses if x!=torus]

def del_elbow(idx,side):
    coord=globals.para_top_bottom[idx][side]
    for torus in globals.toruses:
        if neighbor_elbow(coord, torus.center_coord, torus.torus_radius + torus.radius):
            del_part(torus)

'''如果没有交点，在主函数中已经判断'''
def cast_intersection(top1, bottom1, vertical_cylinder_index, vertical_side):
    vertical_dir = get_correct_dir(globals.desc_load_cylinders[vertical_cylinder_index], vertical_side)
    level_dir = (bottom1 - top1)
    intersection_pos = []
    length=[]
    res = 1e9
    for t in np.arange(0, 1, 0.05):
        level_extend = top1 + level_dir * t
        for k in np.arange(0, 0.5, 0.01):
            vertical_extend = globals.para_top_bottom[vertical_cylinder_index][vertical_side] + vertical_dir * k
            if res > np.linalg.norm(level_extend - vertical_extend):
                res = min(res, np.linalg.norm(level_extend - vertical_extend))
                intersection_pos = level_extend
                length= k
    if res<args.threshold_radius:
        return intersection_pos,length
    else:
        return [],[]

def change_elbow_recursive(rotate,coord,new_neighbor_cylinder,move_torus):
    '''找出所有连接部件，经过旋转、平移得到新的部件'''
    pass


def find_tee(neighbors_index, cylinder_idx, cylinder_side, anchor_center):
    cylinder = globals.desc_load_cylinders[cylinder_idx]
    for neighbor_cylinder_index, neighbor_side in neighbors_index:
        args.threshold_radius = globals.desc_load_cylinders[cylinder_idx].radius + \
                                globals.desc_load_cylinders[neighbor_cylinder_index].radius
        if neighbor_cylinder_index == cylinder_idx:
            continue
        neighbor_cylinder = globals.desc_load_cylinders[neighbor_cylinder_index]
        # o3d.visualization.draw_geometries([cylinder.to_o3d_mesh(),neighbor_cylinder.to_o3d_mesh()])
        if used_cylinder_sides[neighbor_cylinder_index][neighbor_side]:
            continue
        # if np.abs(cylinder.radius - neighbor_cylinder.radius) > args.threshold_radius:
        #     continue
        '''假定三通的优先级高于拐弯，要拼接的部件，如果存在拐弯，则把拐弯删掉'''
        # if in_elbow(neighbor_cylinder_index, neighbor_side) is not None:
        #     continue
        # 检测点云数据，条件设置极为宽松
        neighbor_points_index = np.argmin(np.linalg.norm(tees_center - anchor_center, axis=1))
        if (np.min(np.linalg.norm(tees_points[neighbor_points_index]-anchor_center,axis=1))>3*cylinder.radius
            and np.min(np.linalg.norm(tees_points[neighbor_points_index]-globals.para_top_bottom[neighbor_cylinder_index][neighbor_side],axis=1))
            >3*neighbor_cylinder.radius):
            continue
        # 保证在异侧
        connect_line=globals.para_top_bottom[neighbor_cylinder_index][neighbor_side]-globals.para_top_bottom[cylinder_idx][cylinder_side]
        if np.dot(get_correct_dir(cylinder,cylinder_side),connect_line)<0:
            continue
        # 检测方向条件是否满足
        residual = np.dot(get_correct_dir(cylinder,cylinder_side), get_correct_dir(neighbor_cylinder,neighbor_side))
        if residual > 0:
            continue
        '''平行判断，如果平行判断不满足，再考虑斜交判断'''
        residual = np.abs(residual)
        if residual > args.threshold_parallel:
            # 检查共线条件是否满足
            can_extend = extend_intersection(cylinder_idx, cylinder_side, neighbor_cylinder_index, neighbor_side)
            if can_extend:
                # 接下来判断垂直管道
                for vertical_cylinder_index, vertical_side in neighbors_index:
                    if vertical_cylinder_index == cylinder_idx or vertical_cylinder_index==neighbor_cylinder_index:
                        continue
                    vertical_cylinder = globals.desc_load_cylinders[vertical_cylinder_index]
                    # o3d.visualization.draw_geometries([cylinder.to_o3d_mesh(),neighbor_cylinder.to_o3d_mesh(),vertical_cylinder.to_o3d_mesh()])
                    if used_cylinder_sides[vertical_cylinder_index][vertical_side]:
                        continue
                    vertical_cos = np.abs(np.dot(cylinder.get_direction(), vertical_cylinder.get_direction()))
                    # cos(pi/2-theta)=sin(theta) ，在这里角度限制为55以上
                    if vertical_cos > np.sqrt(1 - args.threshold_parallel*args.threshold_parallel):
                        continue
                    if np.linalg.norm(anchor_center - globals.para_top_bottom[vertical_cylinder_index][
                        vertical_side]) > 6 * args.threshold_radius:
                        continue

                    # 开始处理三通，保持cylinder不动，进行延长
                    '''不考虑原本连接的是拐弯，即假定三通的地方不会提前拟合成拐弯'''
                    # 可视化查看
                    # o3d.visualization.draw_geometries([cylinder.to_o3d_mesh(),neighbor_cylinder.to_o3d_mesh(),vertical_cylinder.to_o3d_mesh()])
                    top_tee = globals.para_top_bottom[cylinder_idx][cylinder_side]
                           # +args.gap_radius*get_correct_dir(cylinder,1-cylinder_side)
                    cylinder_dir=get_correct_dir(cylinder,cylinder_side)
                    tee_height=np.dot(cylinder_dir,(globals.para_top_bottom[neighbor_cylinder_index][neighbor_side]-globals.para_top_bottom[cylinder_idx][cylinder_side]))
                    bottom_tee=top_tee+cylinder_dir*tee_height
                            # +args.gap_radius * get_correct_dir(neighbor_cylinder, 1 - neighbor_side)
                    new_cy_anchor1=bottom_tee
                    new_cy_anchor2=bottom_tee+cylinder_dir*neighbor_cylinder.get_height()
                    new_neighbor_cylinder=Cylinder(new_cy_anchor1,new_cy_anchor2,cylinder.radius)

                    # 计算垂直圆柱延长线的交点
                    vertical_tee_top,vertical_tee_length = cast_intersection(top_tee, bottom_tee, vertical_cylinder_index, vertical_side)
                    if len(vertical_tee_top)==0 or vertical_tee_length==0:
                        continue
                    globals.desc_load_cylinders[neighbor_cylinder_index]=new_neighbor_cylinder
                    globals.para_top_bottom[neighbor_cylinder_index]=(new_cy_anchor1,new_cy_anchor2)
                    # 重新计算垂直圆柱
                    vertical_tee_bottom=vertical_tee_top+get_correct_dir(vertical_cylinder,1-vertical_side)*vertical_tee_length
                    vertical_another_center=vertical_tee_bottom+get_correct_dir(vertical_cylinder,1-vertical_side)*vertical_cylinder.get_height()
                    new_vertical_cy=Cylinder(vertical_another_center,vertical_tee_bottom,vertical_cylinder.radius)
                    globals.desc_load_cylinders[vertical_cylinder_index] = new_vertical_cy
                    globals.para_top_bottom[vertical_cylinder_index] = (vertical_another_center,vertical_tee_bottom)



                    '''找到第二根垂直管道，形成四通'''
                    for vertical_cylinder_index2, vertical_side2 in neighbors_index:
                        if vertical_cylinder_index2 == cylinder_idx or vertical_cylinder_index2 == neighbor_cylinder_index or vertical_cylinder_index2==vertical_cylinder_index:
                            continue
                        # 与前两圆柱的判断关系
                        vertical_cylinder2 = globals.desc_load_cylinders[vertical_cylinder_index2]
                        if used_cylinder_sides[vertical_cylinder_index2][vertical_side2]:
                            continue
                        vertical_cos2 = np.abs(np.dot(cylinder.get_direction(), vertical_cylinder2.get_direction()))
                        if vertical_cos2 > np.sqrt(1 - args.threshold_parallel * args.threshold_parallel):
                            continue
                        if np.linalg.norm(anchor_center - globals.para_top_bottom[vertical_cylinder_index2][
                            vertical_side2]) > 6 * args.threshold_radius:
                            continue
                        # 与平行圆柱的判断关系
                        line=globals.para_top_bottom[vertical_cylinder_index2][vertical_side2]-globals.para_top_bottom[vertical_cylinder_index][vertical_side]
                        if np.dot(get_correct_dir(vertical_cylinder,vertical_side),line)<0:
                            continue
                        residual=np.dot(get_correct_dir(vertical_cylinder,vertical_side),get_correct_dir(vertical_cylinder2,vertical_side2))
                        if residual>0:
                            continue
                        if np.abs(residual)>args.threshold_parallel:
                            vertical_tee_length2=np.linalg.norm(vertical_tee_top-globals.para_top_bottom[vertical_cylinder_index2][vertical_side2])
                            vertical_tee_top=vertical_tee_top+get_correct_dir(vertical_cylinder2,1-vertical_side2)*vertical_tee_length2
                            vertical_another_center2=vertical_tee_top+get_correct_dir(vertical_cylinder2,1-vertical_side2)*vertical_cylinder2.get_height()
                            new_vertical_cy2 = Cylinder(vertical_another_center2, vertical_tee_top,
                                                       vertical_cylinder2.radius)
                            globals.desc_load_cylinders[vertical_cylinder_index2] = new_vertical_cy2
                            globals.para_top_bottom[vertical_cylinder_index2] = (
                            vertical_another_center2, vertical_tee_top)
                            break



                    mytee=Tee(top_tee, bottom_tee, cylinder.radius,vertical_tee_top, vertical_tee_bottom, vertical_cylinder.radius)
                    # 可视化查看
                    pcd_points=o3d.geometry.PointCloud()
                    pcd_points.points=o3d.utility.Vector3dVector(tees_points[neighbor_points_index])
                    o3d.visualization.draw_geometries([mytee.to_o3d_mesh(),globals.desc_load_cylinders[cylinder_idx].to_o3d_mesh(),\
                                                       globals.desc_load_cylinders[neighbor_cylinder_index].to_o3d_mesh(),\
                                                       globals.desc_load_cylinders[vertical_cylinder_index].to_o3d_mesh(),globals.desc_load_cylinders[vertical_cylinder_index2].to_o3d_mesh(),pcd_points])
                    # 更新状态
                    '''还有最有一个圆柱可能也需要更新'''
                    update_cylinder_para(cylinder_idx,cylinder_side, neighbor_cylinder_index,neighbor_side, vertical_cylinder_index,vertical_side)
                    '''因为没有连接关系的数据结构，打组目前搁置'''
                    # move_torus=in_elbow(neighbor_cylinder_index,1-neighbor_side)
                    # rotate=get_rotation_from_cy1_to_cy2(neighbor_cylinder,neighbor_side,new_neighbor_cylinder)
                    # coord=get_coord_error(new_neighbor_cylinder,neighbor_cylinder)
                    # if move_torus is not None:
                    #     change_elbow_recursive(rotate,coord,neighbor_cylinder_index,move_torus)
                    # move_torus=in_elbow(vertical_cylinder_index,1-vertical_side)
                    # if move_torus is not None:
                    #     change_elbow_recursive(vertical_cylinder_index,1-vertical_side,move_torus)
                    if used_cylinder_sides[neighbor_cylinder_index][1-neighbor_side]==False:
                        if used_cylinder_sides[vertical_cylinder_index][1-vertical_side]==False:
                            return mytee,(neighbor_cylinder_index,1-neighbor_side),(vertical_cylinder_index,1-vertical_side)
                        else:
                            return mytee,(neighbor_cylinder_index,1-neighbor_side)
                    if used_cylinder_sides[vertical_cylinder_index][1-vertical_side]==False:
                        return mytee,(vertical_cylinder_index,1-vertical_side)
                    return mytee

    return None


def fit_side(cylinder_idx, cylinder_side):
    anchor_center = globals.para_top_bottom[cylinder_idx][cylinder_side]
    # 得到按照距离排序的side下标
    neighbors_index = get_neighbors(globals.para_top_bottom, anchor_center)[1:args.num_consider_cylinders]
    return find_tee(neighbors_index, cylinder_idx, cylinder_side, anchor_center)

def fit_side_queue(i):
    tees=[]
    pipes=queue.Queue()
    pipes.put((i,TOP))
    pipes.put((i,BOTTOM))
    while not pipes.empty():
        cylinder_idx,cylinder_side=pipes.get()
        # 优先级的更改，有拐弯继续做三通，删除拐弯
        # if in_elbow(cylinder_idx,cylinder_side) is not None:
        #     continue
        # 与elbow一样进行递归处理，但是tee只包含三通，不包含圆柱
        tee=fit_side(cylinder_idx,cylinder_side)
        if tee is not None:
            if isinstance(tee,tuple):
                if len(tee)==2:
                    pipes.put(tee[1])
                    tee=tee[0]
                elif len(tee)==3:
                    pipes.put(tee[1])
                    pipes.put(tee[2])
                    tee=tee[0]
        tees.append(tee)
    return tees

# 一个圆柱斜插入另一个圆柱，不需要再建立三通，直接将圆柱延长即可
def fit_slope_side(cylinder_idx):
    neighbors_index1 = get_neighbors(globals.para_top_bottom, globals.para_top_bottom[cylinder_idx][TOP])[1:args.num_consider_cylinders//2]
    neighbors_index2=get_neighbors(globals.para_top_bottom,globals.para_top_bottom[cylinder_idx][BOTTOM])[1:args.num_consider_cylinders//2]
    neighbors_index=neighbors_index1+neighbors_index2
    cylinder = globals.desc_load_cylinders[cylinder_idx]
    for neighbor_cylinder_index, neighbor_side in neighbors_index:
        args.threshold_radius = (globals.desc_load_cylinders[cylinder_idx].radius + \
                                globals.desc_load_cylinders[neighbor_cylinder_index].radius)/1.5
        if neighbor_cylinder_index == cylinder_idx:
            continue
        neighbor_cylinder = globals.desc_load_cylinders[neighbor_cylinder_index]
        if np.abs(cylinder.radius - neighbor_cylinder.radius) > args.threshold_radius: # 计算得知，如果半径相差5倍关系，则跳过
            continue

        # 检测点云数据
        neighbor_points_index = np.argmin(np.linalg.norm(tees_center - globals.para_top_bottom[neighbor_cylinder_index][neighbor_side], axis=1))
        if np.min(np.linalg.norm(tees_points[neighbor_points_index] - globals.para_top_bottom[neighbor_cylinder_index][neighbor_side], axis=1)) > 3 * args.threshold_radius:
            continue

        # 检测方向条件是否满足
        residual = np.abs(np.dot(cylinder.get_direction(), get_correct_dir(neighbor_cylinder, neighbor_side)))
        if residual < args.threshold_parallel: # 角度至少要大于30度
            # 计算圆柱延长线的交点
            tee_top, tee_length = cast_intersection(cylinder.top_center, cylinder.bottom_center,
                                                                      neighbor_cylinder_index, neighbor_side)
            if len(tee_top)==0 or tee_length ==0:
                continue
            del_elbow(neighbor_cylinder_index, neighbor_side)
            # 重新计算圆柱
            neighbor_another_center = tee_top + get_correct_dir(neighbor_cylinder,
                                                                            1 - neighbor_side) * (tee_length+neighbor_cylinder.get_height())
            new_neighbor_cy = Cylinder(tee_top, neighbor_another_center, neighbor_cylinder.radius)
            globals.desc_load_cylinders[neighbor_cylinder_index] = new_neighbor_cy
            globals.para_top_bottom[neighbor_cylinder_index] = (tee_top, neighbor_another_center)
            # 可视化查看
            pcd_points=o3d.geometry.PointCloud()
            pcd_points.points=o3d.utility.Vector3dVector(tees_points[neighbor_points_index])
            # o3d.visualization.draw_geometries([globals.desc_load_cylinders[cylinder_idx].to_o3d_mesh(),\
            #                                    globals.desc_load_cylinders[neighbor_cylinder_index].to_o3d_mesh()])
            '''因为没有连接关系的数据结构，打组目前搁置'''
            # move_torus=in_elbow(neighbor_cylinder_index,1-neighbor_side)
            # rotate=get_rotation_from_cy1_to_cy2(neighbor_cylinder,neighbor_side,new_neighbor_cylinder)
            # coord=get_coord_error(new_neighbor_cylinder,neighbor_cylinder)
            # if move_torus is not None:
            #     change_elbow_recursive(rotate,coord,neighbor_cylinder_index,move_torus)
            # move_torus=in_elbow(vertical_cylinder_index,1-vertical_side)
            # if move_torus is not None:
            #     change_elbow_recursive(vertical_cylinder_index,1-vertical_side,move_torus)
    return


# 如果圆柱的另一端存在拐弯，则不对该圆柱进行挪动，而是挪动其他的圆柱
def main():
    all_tees=[]
    for i, myCylinder in enumerate(tqdm(globals.desc_load_cylinders)):
        if used_cylinder_sides[i][TOP] == False:
            tees = fit_side_queue(i)
            if tees is not None:
                for tee in tees:
                    all_tees.append(tee)
    # 处理两根管道斜插入的三通，遍历每个圆柱，判断最近圆柱的延长线是否插入
    for i,myCylinder in enumerate(tqdm(globals.desc_load_cylinders)):
        fit_slope_side(i)
    mesh_all=o3d.geometry.TriangleMesh()
    for cy in globals.desc_load_cylinders:
        mesh_all+=cy.to_o3d_mesh()
    for tee in all_tees:
        if tee is not None:
            mesh_all+=tee.to_o3d_mesh()
    o3d.io.write_triangle_mesh('./cross_tee_cylinder/mesh_all.obj', mesh_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tees', help="number of elbows", default=8)
    parser.add_argument('--num_cross_tees', help="number of elbows", default=8)
    parser.add_argument('--num_consider_cylinders', help="neighbor cylinders taken into account", default=11)
    parser.add_argument('--threshold_parallel', help="threshold of two parallel cylinders", default=0.9) # 限制为30度以内
    parser.add_argument('--threshold_radius', help="threshold of two cylinders in distance detection", default=0)
    parser.add_argument('--gap_radius', help="extend the anchor to make the mesh watertight", default=0.1)

    args = parser.parse_args()
    all_points = np.load("cross_tee_cylinder/cross_tees.npy")
    path = "./cross_tee_cylinder/fit_cylinders_for_cross_tee"
    # 保存三通点云
    tees_points = []
    tees_center=[]
    for tee_i in tqdm(range(args.num_tees)):
        data=all_points[all_points[:, 6] == tee_i][:, :3]
        tees_points.append(data)
        tees_center.append(np.mean(data,axis=0))
    # 导入圆柱和拐弯
    load_cylinders = Cylinder.load_cylinders(path)
    globals.desc_load_cylinders = sorted(load_cylinders, key=custom_sort,reverse=True)
    para_numpy = np.asarray([cylinder.numpy_get() for cylinder in globals.desc_load_cylinders])
    globals.para_top_bottom = list((row1, row2) for row1, row2 in zip(para_numpy[:, :3], para_numpy[:, 3:6]))
    cylinder_nums = len(globals.para_top_bottom)
    # 判断圆柱是否使用过的标记
    used_cylinder_sides = [[False, False] for _ in range(cylinder_nums)]
    main()
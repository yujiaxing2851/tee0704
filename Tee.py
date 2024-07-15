import numpy as np
import open3d as o3d
import trimesh
from cylinder import Cylinder

class Tee:
    def __init__(self, top1,bottom1,radius1,top2,bottom2,radius2):
        self.top1 = top1
        self.bottom1 = bottom1
        self.top2 = top2
        self.bottom2 = bottom2
        self.radius1 = radius1
        self.radius2 = radius2

    def get_direction(self) -> 'np.array[float,float,float]':
        pass

    def get_rotation_matrix(self) -> 'np.array[[float,float,float],[float,float,float],[float,float,float]]':
        pass

    def to_o3d_mesh(self)->'o3d.geometry.TriangleMesh':
        cylinderx=Cylinder(self.top1,self.bottom1,self.radius1)
        cylinderz=Cylinder(self.top2,self.bottom2,self.radius2)
        o3d_cylinderx=cylinderx.to_o3d_mesh()
        o3d_cylinderz=cylinderz.to_o3d_mesh()
        # 转换数据格式，求交后再变换回去
        tri_cylinderx=trimesh.Trimesh(vertices=np.asarray(o3d_cylinderx.vertices),
                                      faces=np.asarray(o3d_cylinderx.triangles),
                                      process=False)
        tri_cylinderz=trimesh.Trimesh(vertices=np.asarray(o3d_cylinderz.vertices),
                                      faces=np.asarray(o3d_cylinderz.triangles),
                                      process=False)
        tee=trimesh.boolean.union([tri_cylinderx,tri_cylinderz])
        tee.export("tee.obj")
        mesh=o3d.geometry.TriangleMesh()
        mesh.vertices=o3d.utility.Vector3dVector(tee.vertices)
        mesh.triangles=o3d.utility.Vector3iVector(tee.faces)
        mesh.compute_vertex_normals()
        return mesh

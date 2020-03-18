# Example demonstrating the use of the full-resolution dataset to display
# the ground truth locations of the pelvis, femurs, and anatomical landmarks
# with respect to the projective coordinate frame.
#
# Copyright (C) 2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import sys
import os.path

import numpy as np

import h5py as h5

from vtk import VTK_FLOAT
from vtk import VTK_UNSIGNED_CHAR

from vtk import vtkImageData
from vtk import vtkImageImport
from vtk import vtkImageFlip
from vtk import vtkTransformPolyDataFilter
from vtk import vtkMatrixToHomogeneousTransform
from vtk import vtkMatrix4x4

from vtk import vtkDiscreteMarchingCubes
from vtk import vtkWindowedSincPolyDataFilter
from vtk import vtkQuadricDecimation

from vtk import vtkRenderer
from vtk import vtkRenderWindow
from vtk import vtkRenderWindowInteractor
from vtk import vtkInteractorStyleTrackballCamera
from vtk import vtkCubeAxesActor
from vtk import vtkPolyDataMapper
from vtk import vtkActor
from vtk import vtkSphereSource
from vtk import vtkLineSource
from vtk import vtkPoints
from vtk import vtkQuad
from vtk import vtkCellArray
from vtk import vtkPolyData
from vtk import vtkTexture
from vtk import vtkFloatArray

# Helper function for transforming a surface mesh using
# a 4x4 homogeneous transformation matrix
def xform_mesh(mesh, xform_4x4):
    xform_4x4_vtk = vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            xform_4x4_vtk.SetElement(i,j,xform_4x4[i,j])

    mat_to_xform = vtkMatrixToHomogeneousTransform()
    mat_to_xform.SetInput(xform_4x4_vtk)
    
    xform_polydata = vtkTransformPolyDataFilter()
    xform_polydata.SetInputData(mesh)
    xform_polydata.SetTransform(mat_to_xform)
    xform_polydata.Update()

    return xform_polydata.GetOutput()

# Helper function for creating a mesh from a label volume
def create_mesh(label_pix, labels_to_use, inds_to_phys=None):
    # convert the numpy representation to VTK, so we can create a mesh
    # using marching cubes for display later
    vtk_import = vtkImageImport()
    vtk_import.SetImportVoidPointer(label_pix, True)
    vtk_import.SetDataScalarType(VTK_UNSIGNED_CHAR)
    vtk_import.SetNumberOfScalarComponents(1)

    vtk_import.SetDataExtent(0, label_pix.shape[2] - 1,
                             0, label_pix.shape[1] - 1,
                             0, label_pix.shape[0] - 1)

    vtk_import.SetWholeExtent(0, label_pix.shape[2] - 1,
                              0, label_pix.shape[1] - 1,
                              0, label_pix.shape[0] - 1)

    vtk_import.Update()

    flipper = vtkImageFlip()
    flipper.SetInputData(vtk_import.GetOutput())
    flipper.SetFilteredAxis(1)
    flipper.FlipAboutOriginOff()
    flipper.Update()

    vtk_img = flipper.GetOutput()

    marching_cubes = vtkDiscreteMarchingCubes()
    marching_cubes.SetInputData(vtk_img)
    
    num_labels_to_use = len(labels_to_use)
    
    marching_cubes.SetNumberOfContours(num_labels_to_use)

    for i in range(num_labels_to_use):
        marching_cubes.SetValue(i,labels_to_use[i])
    
    marching_cubes.Update()
    
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(marching_cubes.GetOutput())
    smoother.SetNumberOfIterations(25)
    smoother.SetPassBand(0.1)
    smoother.SetBoundarySmoothing(False)
    smoother.SetFeatureEdgeSmoothing(False)
    smoother.SetFeatureAngle(120.0)
    smoother.SetNonManifoldSmoothing(True)
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    
    mesh_reduce = vtkQuadricDecimation()
    mesh_reduce.SetInputData(smoother.GetOutput())
    mesh_reduce.SetTargetReduction(0.25)
    mesh_reduce.Update()
    
    vertex_xform = np.mat(np.eye(4))
    vertex_xform[1,1] = -1
    vertex_xform[1,3] = label_pix.shape[1] + 1

    vertex_xform = inds_to_phys * vertex_xform

    return xform_mesh(mesh_reduce.GetOutput(), vertex_xform)

# Helper function to invert a rigid transformation stored in a
# 4x4 homogeneous matrix
def invert_rigid(H):
    H_inv = np.mat(np.eye(4))
    
    R_inv = H[0:3,0:3].T
    
    H_inv[0:3,0:3] = R_inv
    H_inv[0:3,3]   = R_inv * -H[0:3,3]

    return H_inv

# Main routine
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: {} <HDF5 full-res data file> <specimen ID> <projection index>'.format(os.path.basename(sys.argv[0])))
        sys.exit(1)
 
    spec_id = sys.argv[2]
    
    proj_idx = int(sys.argv[3])

    # open dataset file for reading
    f = h5.File(sys.argv[1], 'r')

    print('reading projection parameters...')
    proj_params_g = f['proj-params']
    extrinsic = np.mat(proj_params_g['extrinsic'][:]) # world to camera
    extrins_inv = invert_rigid(extrinsic)
    intrinsic = np.mat(proj_params_g['intrinsic'][:]) # project camera to image plane
    intrinsic_inv = intrinsic.I

    proj_num_cols    = proj_params_g['num-cols'][()]
    proj_num_rows    = proj_params_g['num-rows'][()]
    proj_col_spacing = proj_params_g['pixel-col-spacing'][()]
    proj_row_spacing = proj_params_g['pixel-row-spacing'][()]

    focal_len = abs((intrinsic[0,0] * proj_col_spacing) + (intrinsic[1,1] * proj_row_spacing)) / 2.0

    # convert 2D indices to 3D points on the detector plane
    # with respect to the camera projective frame
    def index_2d_to_3d_det(x):
        return intrinsic_inv * -focal_len * x

    det_r0c0 = index_2d_to_3d_det(np.mat([0,0,1]).T)
    det_r0cN = index_2d_to_3d_det(np.mat([proj_num_cols-1,0,1]).T)
    det_rMc0 = index_2d_to_3d_det(np.mat([0,proj_num_rows-1,1]).T)
    det_rMcN = index_2d_to_3d_det(np.mat([proj_num_cols-1,proj_num_rows-1,1]).T)

    # open the group for the specimen we want to deal with
    spec_g = f[spec_id]

    proj_g = spec_g['projections/{:03d}'.format(proj_idx)]

    print('reading projection...')
    proj_pix = proj_g['image/pixels'][:]
    # convert to [0,255] uint8
    proj_pix_min = proj_pix.min()
    proj_pix_max = proj_pix.max()
    proj_pix = 255 * ((proj_pix - proj_pix_min) / (proj_pix_max - proj_pix_min))
    proj_pix = proj_pix.astype('uint8')

    # convert projection into vtk image for display
    vtk_import = vtkImageImport()
    vtk_import.SetImportVoidPointer(proj_pix, True)
    vtk_import.SetDataScalarType(VTK_UNSIGNED_CHAR)
    vtk_import.SetNumberOfScalarComponents(1)

    vtk_import.SetDataExtent(0, proj_pix.shape[1] - 1,
                             0, proj_pix.shape[0] - 1,
                             0, 0)

    vtk_import.SetWholeExtent(0, proj_pix.shape[1] - 1,
                              0, proj_pix.shape[0] - 1,
                              0, 0)

    vtk_import.Update()

    proj_vtk_img = vtk_import.GetOutput()

    print('reading GT poses...')
    gt_poses_g = proj_g['gt-poses']

    cam_to_pelvis_vol      = np.mat(gt_poses_g['cam-to-pelvis-vol'][:])
    cam_to_left_femur_vol  = np.mat(gt_poses_g['cam-to-left-femur-vol'][:])
    cam_to_right_femur_vol = np.mat(gt_poses_g['cam-to-right-femur-vol'][:])

    pelvis_vol_to_cam_proj      = extrinsic * invert_rigid(cam_to_pelvis_vol)
    left_femur_vol_to_cam_proj  = extrinsic * invert_rigid(cam_to_left_femur_vol)
    right_femur_vol_to_cam_proj = extrinsic * invert_rigid(cam_to_right_femur_vol)

    print('reading GT 2D landmarks...')
    lands_2d = {}

    gt_lands_g = proj_g['gt-landmarks']
    for land_name in gt_lands_g:
        land_2d = gt_lands_g[land_name][:]

        if (land_2d[0] >= 0) and (land_2d[1] >= 0) and \
                (land_2d[0] < (proj_num_cols - 1)) and (land_2d[1] < (proj_num_rows - 1)):
            # landmark is visible, now convert the 2D pixel location to a physical point on the detector plane
            lands_2d[land_name] = index_2d_to_3d_det(np.mat(np.append(land_2d, 1)).T)

    print('reading 3D landmarks...')

    lands_3d = {}
    
    lands_3d_g = spec_g['vol-landmarks']
    for land_name in lands_3d_g:
        lands_3d[land_name] = pelvis_vol_to_cam_proj * np.mat(np.append(lands_3d_g[land_name][:],1)).T

    print('reading 3D segmentation...')

    # read in the 3D volume segmentation
    vol_seg_g = spec_g['vol-seg']

    vol_seg_img_g = vol_seg_g['image']

    vol_seg_pix = vol_seg_img_g['pixels'][:]

    vol_seg_spacing = vol_seg_img_g['spacing'][:]
    vol_seg_dir_mat = vol_seg_img_g['dir-mat'][:]
    vol_seg_origin  = vol_seg_img_g['origin'][:]

    vol_seg_idx_to_phys_pt = np.mat(np.eye(4))
    for r in range(3):
        for c in range(3):
            vol_seg_idx_to_phys_pt[r,c] = vol_seg_dir_mat[r,c] * vol_seg_spacing[c]
        vol_seg_idx_to_phys_pt[r,3] = vol_seg_origin[r]

    #print('creating pelvis mesh (both hemipelves)...')
    #pelvis_sur = create_mesh(vol_seg_pix, [1,2])
    
    print('creating left hemipelvis mesh...')
    left_hemipelvis_sur = xform_mesh(create_mesh(vol_seg_pix, [1], vol_seg_idx_to_phys_pt), pelvis_vol_to_cam_proj)
    
    print('creating right hemipelvis mesh...')
    right_hemipelvis_sur = xform_mesh(create_mesh(vol_seg_pix, [2], vol_seg_idx_to_phys_pt), pelvis_vol_to_cam_proj)

    print('creating left femur mesh...')
    left_femur_sur = xform_mesh(create_mesh(vol_seg_pix, [5], vol_seg_idx_to_phys_pt), left_femur_vol_to_cam_proj)
    
    print('creating right femur mesh...')
    right_femur_sur = xform_mesh(create_mesh(vol_seg_pix, [6], vol_seg_idx_to_phys_pt), right_femur_vol_to_cam_proj)

    # Draw everything

    ren     = vtkRenderer()
    ren_win = vtkRenderWindow()
    iren    = vtkRenderWindowInteractor()
    
    ren_win.AddRenderer(ren)
    
    cube_axes_actor = vtkCubeAxesActor()
    ren.AddViewProp(cube_axes_actor)
   
    def add_sur(s, r, g, b):
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(s)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        
        actor.GetProperty().SetColor(r, g, b)

        ren.AddViewProp(actor)

    add_sur(left_hemipelvis_sur, 0.0, 1.0, 0.0)
    add_sur(right_hemipelvis_sur, 1.0, 0.0, 0.0)
    add_sur(left_femur_sur, 0.0, 1.0, 1.0)
    add_sur(right_femur_sur, 1.0, 0.5, 0.0)

    def add_sphere(pt, r, g, b, radius):
        sphere_src = vtkSphereSource()
        sphere_src.SetCenter(pt[0], pt[1], pt[2])
        sphere_src.SetThetaResolution(20)
        sphere_src.SetPhiResolution(20)
        sphere_src.SetRadius(radius)
        sphere_src.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(sphere_src.GetOutput())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(r,g,b)
        
        ren.AddViewProp(actor)

    def add_line(pt1, pt2, r, g, b):
        line_src = vtkLineSource()
        line_src.SetPoint1(pt1[0], pt1[1], pt1[2])
        line_src.SetPoint2(pt2[0], pt2[1], pt2[2])
        line_src.SetResolution(200)
        line_src.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(line_src.GetOutput())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(r,g,b)
        actor.GetProperty().SetLineWidth(2)

        ren.AddViewProp(actor)

    # render the 3D landmarks
    for cur_land in lands_3d.values():
        add_sphere(cur_land, 0.5, 0.0, 0.5, 5)

    # X-ray source
    add_sphere([0,0,0], 0.0, 1.0, 0.0, 10)

    # render the 2D landmarks on the detector plane
    for (land_name, land_2d) in lands_2d.items():
        add_sphere(land_2d, 0.0, 1.0, 0.0, 2.5)
        
        # draw a line between the X-ray source and where the 3D landmark
        # projects to on the detector - it should align with the 2D sphere
        land_3d = lands_3d[land_name]
        
        proj_pt = intrinsic * land_3d[0:3,:]
        proj_pt /= proj_pt[2]

        add_line([0,0,0], index_2d_to_3d_det(proj_pt), 0.0, 1.0, 0.0)

    # Draw the detector plane
    
    #add_sphere(det_r0c0, 1, 1, 0, 5.0)
    #add_sphere(det_r0cN, 1, 146.0/255.0, 20.0/255.0, 5.0)
    #add_sphere(det_rMc0, 116.0/255.0, 0, 189.0/255.0, 5.0)
    #add_sphere(det_rMcN, 46.0/255.0, 238.0/255.0, 1, 5.0)

    det_corners = vtkPoints()
    det_corners.InsertNextPoint(det_r0c0[0], det_r0c0[1], det_r0c0[2])
    det_corners.InsertNextPoint(det_rMc0[0], det_rMc0[1], det_rMc0[2])
    det_corners.InsertNextPoint(det_rMcN[0], det_rMcN[1], det_rMcN[2])
    det_corners.InsertNextPoint(det_r0cN[0], det_r0cN[1], det_r0cN[2])
    
    det_quad = vtkQuad()
    det_quad.GetPointIds().SetId(0,0)
    det_quad.GetPointIds().SetId(1,1)
    det_quad.GetPointIds().SetId(2,2)
    det_quad.GetPointIds().SetId(3,3)

    quads = vtkCellArray()
    quads.InsertNextCell(det_quad)

    det_poly_data = vtkPolyData()
    det_poly_data.SetPoints(det_corners)
    det_poly_data.SetPolys(quads)
    
    tex_coords = vtkFloatArray()
    tex_coords.SetNumberOfComponents(2)
    tex_coords.SetName('TextureCoordinates')
    tex_coords.InsertNextTuple([0,0])
    tex_coords.InsertNextTuple([0,1])
    tex_coords.InsertNextTuple([1,1])
    tex_coords.InsertNextTuple([1,0])

    det_poly_data.GetPointData().SetTCoords(tex_coords)

    tex = vtkTexture()
    tex.SetInputData(proj_vtk_img)
    
    plane_mapper = vtkPolyDataMapper()
    plane_mapper.SetInputData(det_poly_data)
    plane_mapper.Update()
    plane_actor = vtkActor()
    plane_actor.SetMapper(plane_mapper)
    plane_actor.SetTexture(tex)

    plane_actor.GetProperty().BackfaceCullingOff()
    plane_actor.GetProperty().FrontfaceCullingOff()

    ren.AddViewProp(plane_actor)

    ren.ResetCamera()
    ren.ResetCameraClippingRange()

    ren_win.SetSize(1024, 1024)
    
    ren.SetBackground(0.7, 0.8, 1.0)
    
    # setup the axes display parameters

    cube_axes_actor.VisibilityOn()
   
    cube_axes_actor.SetBounds(ren.ComputeVisiblePropBounds())
    
    cube_axes_actor.SetCamera(ren.GetActiveCamera())

    cube_axes_actor.GetTitleTextProperty(0).SetColor(1.0, 0.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(0).SetColor(1.0, 0.0, 0.0)

    cube_axes_actor.GetTitleTextProperty(1).SetColor(0.0, 1.0, 0.0)
    cube_axes_actor.GetLabelTextProperty(1).SetColor(0.0, 1.0, 0.0)

    cube_axes_actor.GetTitleTextProperty(2).SetColor(0.0, 0.0, 1.0)
    cube_axes_actor.GetLabelTextProperty(2).SetColor(0.0, 0.0, 1.0)

    cube_axes_actor.DrawXGridlinesOn()
    cube_axes_actor.DrawYGridlinesOn()
    cube_axes_actor.DrawZGridlinesOn()

    cube_axes_actor.SetGridLineLocation(vtkCubeAxesActor.VTK_GRID_LINES_FURTHEST)

    cube_axes_actor.XAxisMinorTickVisibilityOff()
    cube_axes_actor.YAxisMinorTickVisibilityOff()
    cube_axes_actor.ZAxisMinorTickVisibilityOff()

    # kick off the interactive renderer

    ren_win.Render()
    
    iren.SetRenderWindow(ren_win)
    
    interactor_style = vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(interactor_style)

    iren.Start()


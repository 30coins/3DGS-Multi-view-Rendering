"""
3D Gaussian Splatting 水平多视点宫格图采样脚本
从正前方采样12*6的多视点宫格图，水平方向40°范围，垂直方向不变
"""

import numpy as np
from plyfile import PlyData
from PIL import Image
import os
import argparse
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    HAS_RASTERIZER = True
except ImportError:
    HAS_RASTERIZER = False

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: open3d not found. Please install it for better rendering: pip install open3d")


def load_ply(path):
    """加载3DGS的.ply文件"""
    plydata = PlyData.read(path)
    
    # 提取顶点数据（高斯参数）
    vertices = plydata['vertex']
    
    # 获取字段名列表（安全的方式）
    def has_field(field_name):
        """检查字段是否存在"""
        try:
            _ = vertices[field_name]
            return True
        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
            try:
                if hasattr(vertices, 'dtype'):
                    dtype_obj = vertices.dtype
                    if callable(dtype_obj):
                        dtype_obj = dtype_obj()
                    if hasattr(dtype_obj, 'names') and dtype_obj.names is not None:
                        return field_name in dtype_obj.names
                    elif hasattr(dtype_obj, 'fields') and dtype_obj.fields is not None:
                        return field_name in dtype_obj.fields
                return False
            except:
                return False
    
    # 提取位置
    positions = np.stack([
        np.array(vertices['x']),
        np.array(vertices['y']),
        np.array(vertices['z'])
    ], axis=1).astype(np.float32)
    
    # 提取颜色（3DGS通常使用SH系数，需要转换为RGB）
    try:
        if hasattr(vertices.dtype, 'names') and vertices.dtype.names:
            available_fields = list(vertices.dtype.names)
            print(f"PLY文件中的字段: {', '.join(available_fields)}")
    except Exception as e:
        print(f"无法读取字段列表: {e}")
    
    # 优先检查SH系数（f_dc_0, f_dc_1, f_dc_2）- 这是3DGS标准格式
    if has_field('f_dc_0') and has_field('f_dc_1') and has_field('f_dc_2'):
        f_dc = np.stack([
            np.array(vertices['f_dc_0']),
            np.array(vertices['f_dc_1']),
            np.array(vertices['f_dc_2'])
        ], axis=1).astype(np.float32)
        
        C0 = 0.28209479177387814
        colors = 1 / (1 + np.exp(-(f_dc + C0)))
        
        # 降低亮度并增强饱和度
        brightness_reduction = 0.85
        colors = colors * brightness_reduction
        
        gray = np.mean(colors, axis=1, keepdims=True)
        saturation_factor = 1.3
        colors = gray + (colors - gray) * saturation_factor
        colors = np.clip(colors, 0, 1)
        
        gamma = 0.9
        colors = np.power(colors, gamma)
        
        print(f"从SH系数提取颜色，颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    elif has_field('SH0') and has_field('SH1') and has_field('SH2'):
        sh_dc = np.stack([
            np.array(vertices['SH0']),
            np.array(vertices['SH1']),
            np.array(vertices['SH2'])
        ], axis=1).astype(np.float32)
        colors = 1 / (1 + np.exp(-sh_dc))
        print(f"从SH系数（SH0/1/2）提取颜色，颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    elif has_field('red') and has_field('green') and has_field('blue'):
        colors = np.stack([
            np.array(vertices['red']),
            np.array(vertices['green']),
            np.array(vertices['blue'])
        ], axis=1).astype(np.float32) / 255.0
        print(f"从RGB值提取颜色，颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    else:
        print("警告: 未找到颜色信息（f_dc_0/1/2 或 red/green/blue），使用默认灰色")
        colors = np.ones_like(positions) * 0.5
    
    # 提取不透明度
    if has_field('opacity'):
        opacities = np.array(vertices['opacity']).astype(np.float32)
    else:
        opacities = np.ones(len(positions), dtype=np.float32)
    
    # 提取缩放
    if has_field('scale_0') and has_field('scale_1') and has_field('scale_2'):
        scales = np.stack([
            np.array(vertices['scale_0']),
            np.array(vertices['scale_1']),
            np.array(vertices['scale_2'])
        ], axis=1).astype(np.float32)
    else:
        scales = np.ones_like(positions) * 0.01
    
    # 提取旋转（四元数）
    if has_field('rot_0') and has_field('rot_1') and has_field('rot_2') and has_field('rot_3'):
        rotations = np.stack([
            np.array(vertices['rot_0']),
            np.array(vertices['rot_1']),
            np.array(vertices['rot_2']),
            np.array(vertices['rot_3'])
        ], axis=1).astype(np.float32)
    else:
        rotations = np.zeros((len(positions), 4), dtype=np.float32)
        rotations[:, 0] = 1.0
    
    return {
        'positions': positions,
        'colors': colors,
        'opacities': opacities,
        'scales': scales,
        'rotations': rotations
    }


def create_camera_matrices(width, height, fov, eye, target, up):
    """创建相机矩阵"""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    view_matrix = np.array([
        [right[0], up[0], -forward[0], 0],
        [right[1], up[1], -forward[1], 0],
        [right[2], up[2], -forward[2], 0],
        [-np.dot(right, eye), -np.dot(up, eye), np.dot(forward, eye), 1]
    ])
    
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    aspect = width / height
    near, far = 0.01, 100.0
    
    proj_matrix = np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, 2 * far * near / (near - far), 0]
    ])
    
    return view_matrix, proj_matrix


def calculate_optimal_distance(positions, width, height, fov, fill_ratio=0.8):
    """根据物体边界框自动计算合适的相机距离"""
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    center = (min_pos + max_pos) / 2.0
    
    size = max_pos - min_pos
    max_size = np.max(size)
    
    image_min_dim = min(width, height)
    
    fov_rad = np.radians(fov)
    tan_half_fov = np.tan(fov_rad / 2.0)
    
    aspect_ratio = width / height
    if aspect_ratio > 1:
        view_height = max_size / fill_ratio
        view_width = view_height * aspect_ratio
    else:
        view_width = max_size / fill_ratio
        view_height = view_width / aspect_ratio
    
    view_size = max(view_width, view_height)
    distance = (view_size / 2.0) / tan_half_fov
    
    return distance


def generate_camera_poses(num_cols=12, num_rows=6, horizontal_angle=40.0, distance=3.0):
    """生成相机位姿（整个12*6宫格从左上的-20°到右下的+20°，垂直方向不变）
    
    Args:
        num_cols: 列数（横向视角数，默认12）
        num_rows: 行数（纵向视角数，默认6，但垂直角度相同）
        horizontal_angle: 水平总视角范围（度，默认40°），整个宫格覆盖从-angle/2到+angle/2
        distance: 相机距离
    """
    poses = []
    
    # 计算物体中心（假设在原点）
    center = np.array([0.0, 0.0, 0.0])
    
    # 水平角度范围：从 -horizontal_angle/2 到 +horizontal_angle/2
    start_h_angle = -horizontal_angle / 2.0  # 起始角度（例如-20°）
    end_h_angle = horizontal_angle / 2.0      # 结束角度（例如+20°）
    
    # 计算总视角数
    total_views = num_cols * num_rows
    
    # 计算角度间隔：整个宫格从左上的-20°到右下的+20°
    if total_views > 1:
        angle_step = horizontal_angle / (total_views - 1)
    else:
        angle_step = 0.0
    
    # 垂直角度：所有行都使用相同的角度（稍微向下看，能看到椅子腿）
    vertical_angle = -10.0  # 向下倾斜10度
    
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前视角在整个宫格中的索引（从左到右、从上到下）
            view_index = row * num_cols + col
            
            # 计算水平角度：从左上角的-20°连续递增到右下角的+20°
            if total_views == 1:
                horizontal_angle_curr = 0.0
            else:
                horizontal_angle_curr = start_h_angle + view_index * angle_step
            
            # 垂直角度对所有行都相同
            vertical_angle_curr = vertical_angle
            
            # 转换为弧度
            h_rad = np.radians(horizontal_angle_curr)
            v_rad = np.radians(vertical_angle_curr)
            
            # 计算相机位置
            # 从Y轴正方向（上方）看，相机应该在XZ平面上绕Y轴旋转
            # 相机在Y轴正方向（上方），高度为distance，在XZ平面上水平旋转
            # 为了能看到明显的水平视角变化，相机应该在XZ平面上形成一个明显的圆
            cos_h = np.cos(h_rad)
            sin_h = np.sin(h_rad)
            cos_v = np.cos(v_rad)  # v_rad是-10度，cos(-10°) ≈ 0.985
            sin_v = np.sin(v_rad)  # sin(-10°) ≈ -0.174
            
            # 相机位置：在XZ平面上，高度在Y轴正方向
            # 为了让水平旋转更明显，相机应该在XZ平面上形成一个圆
            # 相机高度在Y轴正方向，在XZ平面上水平旋转
            # 使用一个固定的半径在XZ平面上旋转，高度保持在Y轴正方向
            
            # 方法：相机在XZ平面上，高度在Y轴正方向
            # 在XZ平面上的旋转半径应该足够大，以便能看到明显的视角变化
            # 如果垂直角度是-10度（向下倾斜10度），相机应该在Y轴正方向，但稍微向下
            # 在XZ平面上的旋转半径 = distance * sin(10°) ≈ distance * 0.174，这太小了
            
            # 更好的方法：相机在XZ平面上，高度在Y轴正方向，但旋转半径更大
            # 或者，让相机在XZ平面上形成一个圆，高度保持在Y轴正方向
            # 相机到原点的距离为distance，在XZ平面上水平旋转
            
            # 使用球坐标系，但让水平旋转更明显：
            # 相机在XZ平面上，高度在Y轴正方向
            # 在XZ平面上的旋转半径应该足够大
            # 如果垂直角度是-10度，相机应该在Y轴正方向，但稍微向下
            # 在XZ平面上的旋转半径 = distance * sin(10°) ≈ distance * 0.174
            
            # 为了让水平旋转更明显，我们可以增加在XZ平面上的旋转半径
            # 或者，让相机在XZ平面上形成一个圆，高度保持在Y轴正方向
            # 相机到原点的距离为distance，在XZ平面上水平旋转
            
            # 重新设计：相机在XZ平面上，高度在Y轴正方向
            # 在XZ平面上的旋转半径 = distance * sin(从Y轴正方向的角度)
            # 如果垂直角度是-10度（向下倾斜10度），从Y轴正方向的角度是10度
            angle_from_y = -v_rad  # 从Y轴正方向的角度，-10度变成10度（向下倾斜10度）
            angle_from_y_rad = np.radians(angle_from_y)
            
            # 在XZ平面上的旋转半径
            # 如果使用 sin(10°)，旋转半径太小，看不到明显的视角变化
            # 为了让水平旋转更明显，我们增加旋转半径
            # 使用一个更大的旋转半径，比如 distance 的一部分
            # 这样相机在XZ平面上形成一个明显的圆，高度保持在Y轴正方向
            
            # 方法1：使用固定的旋转半径（比如 distance 的 0.5 倍）
            # radius_xz = distance * 0.5
            
            # 方法2：使用基于垂直角度的旋转半径，但放大
            # 如果垂直角度是-10度，从Y轴正方向的角度是10度
            # 但为了让水平旋转更明显，我们可以使用一个更大的角度
            # 比如使用30度或45度来计算旋转半径
            
            # 方法3：直接使用 distance 作为旋转半径的一部分
            # 让相机在XZ平面上形成一个圆，高度保持在Y轴正方向
            # 旋转半径 = distance * 0.7（这样能看到明显的视角变化）
            radius_xz = distance * 0.7  # 使用较大的旋转半径，让水平旋转更明显
            
            # 相机高度保持在Y轴正方向
            height_y = distance * np.cos(angle_from_y_rad)  # ≈ distance * 0.985
            
            eye = np.array([
                radius_xz * sin_h,   # X轴：左右（在XZ平面上）
                height_y,            # Y轴：高度（在Y轴正方向）
                radius_xz * cos_h    # Z轴：前后（在XZ平面上）
            ])
            
            # 目标点：物体中心（稍微向下，能看到椅子背和椅子腿）
            target = center + np.array([0.0, -0.2, 0.0])
            # 从Y轴正方向看时，up向量应该是Z轴正方向
            # 但需要根据相机位置调整，确保up向量垂直于视线方向
            forward = target - eye
            forward = forward / np.linalg.norm(forward)
            # 计算right向量
            right = np.cross(forward, np.array([0.0, 0.0, 1.0]))  # 使用Z轴正方向作为参考
            if np.linalg.norm(right) < 1e-6:
                # 如果forward和Z轴正方向平行，使用X轴正方向作为参考
                right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
            right = right / np.linalg.norm(right)
            # 重新计算up向量，确保与forward和right垂直
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            poses.append({
                'eye': eye,
                'target': target,
                'up': up,
                'row': row,
                'col': col,
                'horizontal_angle': horizontal_angle_curr,
                'vertical_angle': vertical_angle_curr
            })
    
    return poses


def render_with_rasterizer(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """使用diff-gaussian-rasterization渲染"""
    if not HAS_RASTERIZER:
        return None
    
    means3D = torch.from_numpy(gaussians['positions']).cuda().float()
    colors = torch.from_numpy(gaussians['colors']).cuda().float()
    opacity = torch.from_numpy(gaussians['opacities']).cuda().float()
    scales = torch.from_numpy(gaussians['scales']).cuda().float()
    rotations = torch.from_numpy(gaussians['rotations']).cuda().float()
    
    view_matrix, proj_matrix = create_camera_matrices(
        width, height, fov,
        camera_pose['eye'],
        camera_pose['target'],
        camera_pose['up']
    )
    
    view_matrix = torch.from_numpy(view_matrix).cuda().float()
    proj_matrix = torch.from_numpy(proj_matrix).cuda().float()
    
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=np.tan(np.radians(fov / 2)) * (width / height),
        tanfovy=np.tan(np.radians(fov / 2)),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=proj_matrix,
        sh_degree=0,
        campos=torch.from_numpy(camera_pose['eye']).cuda().float(),
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=None,
        sh=None,
        colors_precomp=colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    
    image = rendered_image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    
    return image


def render_with_open3d(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """使用Open3D渲染点云"""
    if not HAS_OPEN3D:
        return None
    
    pcd = o3d.geometry.PointCloud()
    positions = gaussians['positions']
    colors = gaussians['colors']
    
    colors = np.clip(colors, 0.0, 1.0)
    
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        material.shader = "defaultUnlit"
        
        renderer.scene.add_geometry("pointcloud", pcd, material)
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        
        eye = camera_pose['eye']
        target = camera_pose['target']
        up = camera_pose['up']
        
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        
        center = target
        
        fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        
        renderer.scene.camera.look_at(center, eye, up_corrected)
        
        try:
            renderer.scene.camera.set_projection(
                intrinsic_matrix=np.array([
                    [fx, 0, width / 2.0],
                    [0, fy, height / 2.0],
                    [0, 0, 1]
                ], dtype=np.float64),
                fov_deg=fov,
                near=0.01,
                far=100.0,
                width=width,
                height=height
            )
        except (TypeError, AttributeError):
            try:
                renderer.scene.camera.set_projection(
                    np.array([
                        [fx, 0, width / 2.0],
                        [0, fy, height / 2.0],
                        [0, 0, 1]
                    ], dtype=np.float64),
                    fov, 0.01, 100.0, width, height
                )
            except:
                pass
        
        image = renderer.render_to_image()
        image = np.asarray(image)
        
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
        
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
        
    except (AttributeError, RuntimeError, Exception) as e:
        has_display = os.environ.get('DISPLAY') is not None
        
        if has_display:
            try:
                vis = o3d.visualization.Visualizer()
                success = vis.create_window(width=width, height=height, visible=False)
                
                if not success:
                    raise RuntimeError("Failed to create window")
                
                vis.add_geometry(pcd)
                
                render_option = vis.get_render_option()
                if render_option is None:
                    raise RuntimeError("Failed to get render option")
                
                render_option.background_color = np.asarray([0.0, 0.0, 0.0])
                render_option.point_size = 2.0
                
                ctr = vis.get_view_control()
                
                eye = camera_pose['eye']
                target = camera_pose['target']
                up = camera_pose['up']
                
                forward = target - eye
                forward = forward / np.linalg.norm(forward)
                
                param = ctr.convert_to_pinhole_camera_parameters()
                
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up_corrected = np.cross(right, forward)
                
                extrinsic = np.eye(4)
                extrinsic[0:3, 0] = right
                extrinsic[0:3, 1] = -up_corrected
                extrinsic[0:3, 2] = -forward
                extrinsic[0:3, 3] = -np.dot(extrinsic[0:3, 0:3], eye)
                
                param.extrinsic = extrinsic
                
                fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
                cx = width / 2.0
                cy = height / 2.0
                
                param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
                ctr.convert_from_pinhole_camera_parameters(param)
                
                vis.poll_events()
                vis.update_renderer()
                
                image = vis.capture_screen_float_buffer(do_render=True)
                image = np.asarray(image)
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                
                vis.destroy_window()
                
                return image
                
            except Exception as e2:
                print(f"Warning: Open3D GUI rendering also failed: {e2}")
                return None
        else:
            print(f"Warning: Open3D offscreen rendering failed (no DISPLAY): {e}")
            return None


def render_simple(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """简单的渲染方法（当没有其他渲染器时使用）"""
    eye = camera_pose['eye']
    target = camera_pose['target']
    up = camera_pose['up']
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_corrected = np.cross(right, forward)
    
    positions = gaussians['positions']
    colors = gaussians['colors']
    opacities = gaussians['opacities']
    scales = gaussians['scales']
    
    colors = np.clip(colors, 0.0, 1.0)
    
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    focal_length = width / (2.0 * np.tan(np.radians(fov) / 2.0))
    
    rel_positions = positions - eye
    depths = np.dot(rel_positions, forward)
    
    valid_mask = depths > 0.01
    
    if not np.any(valid_mask):
        return (image * 255).astype(np.uint8)
    
    valid_positions = rel_positions[valid_mask]
    valid_depths = depths[valid_mask]
    valid_colors = colors[valid_mask]
    valid_opacities = opacities[valid_mask]
    valid_scales = scales[valid_mask]
    
    x_coords = np.dot(valid_positions, right) * focal_length / valid_depths
    y_coords = np.dot(valid_positions, up_corrected) * focal_length / valid_depths
    
    pixel_x = (x_coords + width / 2).astype(int)
    pixel_y = (-y_coords + height / 2).astype(int)
    
    point_sizes = np.max(valid_scales, axis=1) * focal_length / valid_depths
    point_sizes = np.clip(point_sizes, 1, 10).astype(int)
    
    sort_indices = np.argsort(-valid_depths)
    
    for idx in sort_indices:
        px, py = pixel_x[idx], pixel_y[idx]
        size = point_sizes[idx]
        color = valid_colors[idx]
        alpha = min(valid_opacities[idx], 1.0)
        depth = valid_depths[idx]
        
        half_size = size // 2
        y_start = max(0, py - half_size)
        y_end = min(height, py + half_size + 1)
        x_start = max(0, px - half_size)
        x_end = min(width, px + half_size + 1)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                dx, dy = x - px, y - py
                if dx*dx + dy*dy <= half_size*half_size:
                    if depth < depth_buffer[y, x]:
                        image[y, x] = image[y, x] * (1 - alpha) + color * alpha
                        depth_buffer[y, x] = depth
    
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    return image


def create_grid_image(images, num_cols=12, num_rows=6, img_width=800, img_height=600):
    """将多个图像组合成宫格图"""
    grid_width = num_cols * img_width
    grid_height = num_rows * img_height
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for img_data in images:
        row = img_data['row']
        col = img_data['col']
        img = img_data['image']
        
        if img.shape[0] != img_height or img.shape[1] != img_width:
            img = Image.fromarray(img)
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            img = np.array(img)
        
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        grid_image[y_start:y_end, x_start:x_end] = img
    
    return grid_image


def main():
    parser = argparse.ArgumentParser(description='3DGS水平多视点宫格图采样（水平40°，垂直不变）')
    parser.add_argument('--input', type=str, required=True, help='输入的.ply文件路径')
    parser.add_argument('--output', type=str, default='grid_horizontal_output.png', help='输出的宫格图路径')
    parser.add_argument('--cols', type=int, default=12, help='列数（横向视角数，默认12）')
    parser.add_argument('--rows', type=int, default=6, help='行数（纵向视角数，默认6，但垂直角度相同）')
    parser.add_argument('--angle', type=float, default=40.0, help='水平总视角范围（度，默认40°）')
    parser.add_argument('--distance', type=float, default=None, help='相机距离（如果为None则自动计算，默认None）')
    parser.add_argument('--fill_ratio', '--fill-ratio', type=float, default=0.8, help='物体占据图像的比例（0-1，默认0.8表示80%%，仅在distance为None时有效）')
    parser.add_argument('--width', type=int, default=800, help='单个图像宽度（默认800）')
    parser.add_argument('--height', type=int, default=600, help='单个图像高度（默认600）')
    parser.add_argument('--fov', type=float, default=60.0, help='视野角度（度，默认60°）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"加载3DGS模型: {args.input}")
    gaussians = load_ply(args.input)
    print(f"加载了 {len(gaussians['positions'])} 个高斯点")
    
    if args.distance is None:
        print(f"自动计算相机距离（物体占比: {args.fill_ratio*100:.0f}%）...")
        distance = calculate_optimal_distance(
            gaussians['positions'],
            args.width,
            args.height,
            args.fov,
            args.fill_ratio
        )
        print(f"计算得到的相机距离: {distance:.2f}")
    else:
        distance = args.distance
        print(f"使用指定的相机距离: {distance:.2f}")
    
    print(f"生成 {args.cols}x{args.rows} 的相机位姿...")
    print(f"水平视角范围：从 -{args.angle/2:.1f}° 到 +{args.angle/2:.1f}°（总共 {args.angle}°）")
    print(f"垂直视角：所有行都使用相同的角度（稍微向下看）")
    camera_poses = generate_camera_poses(
        num_cols=args.cols,
        num_rows=args.rows,
        horizontal_angle=args.angle,
        distance=distance
    )
    
    # 打印第一个和最后一个视角的角度，用于调试
    if len(camera_poses) > 0:
        first_pose = camera_poses[0]
        last_pose = camera_poses[-1]
        print(f"第一个视角（左上角）：行{first_pose['row']+1}, 列{first_pose['col']+1}, 水平角度{first_pose['horizontal_angle']:.2f}°")
        print(f"最后一个视角（右下角）：行{last_pose['row']+1}, 列{last_pose['col']+1}, 水平角度{last_pose['horizontal_angle']:.2f}°")
        print(f"角度差：{last_pose['horizontal_angle'] - first_pose['horizontal_angle']:.2f}°")
    
    print("开始渲染...")
    use_simple_render = False
    
    if HAS_RASTERIZER and HAS_TORCH and torch.cuda.is_available():
        print("使用 diff-gaussian-rasterization 渲染器")
    elif HAS_OPEN3D:
        print("使用 Open3D 渲染器（如果颜色显示有问题，将自动切换到numpy渲染）")
    else:
        print("使用 numpy 渲染方法")
        use_simple_render = True
    
    images = []
    total = len(camera_poses)
    open3d_failed = False
    
    for i, pose in enumerate(camera_poses):
        print(f"渲染 {i+1}/{total} (行{pose['row']+1}, 列{pose['col']+1}, 水平角度{pose['horizontal_angle']:.1f}°)...")
        
        img = None
        
        if not use_simple_render and HAS_RASTERIZER and HAS_TORCH and torch.cuda.is_available():
            img = render_with_rasterizer(
                gaussians, pose,
                width=args.width,
                height=args.height,
                fov=args.fov
            )
        elif not use_simple_render and HAS_OPEN3D and not open3d_failed:
            img = render_with_open3d(
                gaussians, pose,
                width=args.width,
                height=args.height,
                fov=args.fov
            )
            
            # 如果Open3D渲染失败（返回None）或第一张图检测到黑白，切换到numpy渲染
            if img is None:
                if i == 0:
                    print("检测到Open3D渲染失败，切换到numpy渲染方法...")
                open3d_failed = True
                use_simple_render = True
                img = render_simple(
                    gaussians, pose,
                    width=args.width,
                    height=args.height,
                    fov=args.fov
                )
            elif img is not None and i == 0:
                img_check = img.astype(np.float32) / 255.0
                if img_check.max() < 0.1 or img_check.std() < 0.01:
                    print("检测到Open3D渲染结果为黑白，切换到numpy渲染方法...")
                    open3d_failed = True
                    use_simple_render = True
                    img = render_simple(
                        gaussians, pose,
                        width=args.width,
                        height=args.height,
                        fov=args.fov
                    )
        
        if img is None or use_simple_render:
            img = render_simple(
                gaussians, pose,
                width=args.width,
                height=args.height,
                fov=args.fov
            )
        
        if img is not None:
            images.append({
                'image': img,
                'row': pose['row'],
                'col': pose['col']
            })
        else:
            print(f"警告: 渲染失败 (行{pose['row']+1}, 列{pose['col']+1})")
    
    print("组合宫格图...")
    grid_image = create_grid_image(
        images,
        num_cols=args.cols,
        num_rows=args.rows,
        img_width=args.width,
        img_height=args.height
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    grid_pil = Image.fromarray(grid_image)
    grid_pil.save(output_path)
    print(f"宫格图已保存到: {output_path}")
    print(f"图像尺寸: {grid_image.shape[1]}x{grid_image.shape[0]}")


if __name__ == '__main__':
    main()

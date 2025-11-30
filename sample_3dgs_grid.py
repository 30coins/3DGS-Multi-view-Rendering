"""
3D Gaussian Splatting 多视点宫格图采样脚本
从正前方采样12*6的多视点宫格图，视角间隔约15°
"""

import numpy as np
import torch
from plyfile import PlyData
from PIL import Image
import os
import argparse
from pathlib import Path

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
            # 直接尝试访问字段，如果成功则存在
            _ = vertices[field_name]
            return True
        except (KeyError, ValueError, IndexError, TypeError, AttributeError):
            # 如果直接访问失败，尝试其他方法
            try:
                # 尝试获取dtype信息
                if hasattr(vertices, 'dtype'):
                    dtype_obj = vertices.dtype
                    # 检查是否是函数
                    if callable(dtype_obj):
                        # 如果是函数，尝试调用
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
    # 首先列出所有可用的字段以便调试
    try:
        if hasattr(vertices.dtype, 'names') and vertices.dtype.names:
            available_fields = list(vertices.dtype.names)
            print(f"PLY文件中的字段: {', '.join(available_fields)}")
            # 检查颜色相关字段
            color_fields = [f for f in available_fields if 'color' in f.lower() or 'red' in f.lower() or 'green' in f.lower() or 'blue' in f.lower() or 'f_dc' in f.lower() or 'sh' in f.lower()]
            if color_fields:
                print(f"找到颜色相关字段: {', '.join(color_fields)}")
    except Exception as e:
        print(f"无法读取字段列表: {e}")
    
    # 优先检查SH系数（f_dc_0, f_dc_1, f_dc_2）- 这是3DGS标准格式
    if has_field('f_dc_0') and has_field('f_dc_1') and has_field('f_dc_2'):
        # 3DGS使用SH系数，DC项（f_dc）需要经过sigmoid转换为RGB
        f_dc = np.stack([
            np.array(vertices['f_dc_0']),
            np.array(vertices['f_dc_1']),
            np.array(vertices['f_dc_2'])
        ], axis=1).astype(np.float32)
        
        # 将SH系数转换为RGB（使用sigmoid函数）
        # 3DGS的SH系数需要先加上C0系数（0.28209479177387814），然后再sigmoid
        print(f"SH系数范围: f_dc_0[{f_dc[:, 0].min():.3f}, {f_dc[:, 0].max():.3f}], "
              f"f_dc_1[{f_dc[:, 1].min():.3f}, {f_dc[:, 1].max():.3f}], "
              f"f_dc_2[{f_dc[:, 2].min():.3f}, {f_dc[:, 2].max():.3f}]")
        
        # 标准3DGS方式：先加上C0系数再sigmoid
        C0 = 0.28209479177387814
        colors = 1 / (1 + np.exp(-(f_dc + C0)))
        
        # 降低亮度并增强饱和度（减少发白，使颜色更鲜艳）
        # 方法1：降低整体亮度（减少发白）
        brightness_reduction = 0.85  # 降低15%的亮度
        colors = colors * brightness_reduction
        
        # 方法2：增强饱和度（使颜色更鲜艳，减少发白）
        # 通过增加颜色通道之间的差异来增强饱和度
        gray = np.mean(colors, axis=1, keepdims=True)  # 计算灰度值
        saturation_factor = 1.3  # 饱和度增强因子
        colors = gray + (colors - gray) * saturation_factor
        colors = np.clip(colors, 0, 1)
        
        # 方法3：使用gamma校正增强对比度（但不要过度）
        gamma = 0.9  # 稍微增强对比度
        colors = np.power(colors, gamma)
        
        print(f"转换后颜色范围: R[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}], "
              f"G[{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}], "
              f"B[{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
        # 显示前几个样本
        print(f"前5个点的颜色样本: {colors[:5]}")
    # 检查其他可能的SH格式（SH0, SH1, SH2等）
    elif has_field('SH0') and has_field('SH1') and has_field('SH2'):
        # 另一种SH格式
        sh_dc = np.stack([
            np.array(vertices['SH0']),
            np.array(vertices['SH1']),
            np.array(vertices['SH2'])
        ], axis=1).astype(np.float32)
        colors = 1 / (1 + np.exp(-sh_dc))
        print(f"从SH系数（SH0/1/2）提取颜色，颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    elif has_field('red') and has_field('green') and has_field('blue'):
        # 直接使用RGB值
        colors = np.stack([
            np.array(vertices['red']),
            np.array(vertices['green']),
            np.array(vertices['blue'])
        ], axis=1).astype(np.float32) / 255.0
        print(f"从RGB值提取颜色，颜色范围: [{colors.min():.3f}, {colors.max():.3f}]")
    else:
        # 如果没有颜色信息，使用默认颜色
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
        # 默认四元数 [1, 0, 0, 0]
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
    # 计算视图矩阵
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
    
    # 投影矩阵
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
    """根据物体边界框自动计算合适的相机距离，使物体占据图像的一定比例
    
    Args:
        positions: 物体位置数组 (N, 3)
        width: 图像宽度
        height: 图像高度
        fov: 视野角度（度）
        fill_ratio: 物体占据图像的比例（0-1，默认0.8表示80%）
    
    Returns:
        合适的相机距离
    """
    # 计算物体的边界框
    min_pos = np.min(positions, axis=0)
    max_pos = np.max(positions, axis=0)
    center = (min_pos + max_pos) / 2.0
    
    # 计算物体的尺寸
    size = max_pos - min_pos
    max_size = np.max(size)  # 物体的最大尺寸
    
    # 计算图像的对角线尺寸（在3D空间中的对应尺寸）
    # 使用较小的维度来计算，确保物体完全可见
    image_min_dim = min(width, height)
    
    # 根据FOV计算在指定距离处的视野尺寸
    # tan(fov/2) = (视野尺寸/2) / 距离
    # 距离 = (视野尺寸/2) / tan(fov/2)
    # 我们希望：max_size * fill_ratio = 视野尺寸
    # 所以：视野尺寸 = max_size / fill_ratio
    fov_rad = np.radians(fov)
    tan_half_fov = np.tan(fov_rad / 2.0)
    
    # 计算需要的视野尺寸（考虑图像宽高比）
    aspect_ratio = width / height
    if aspect_ratio > 1:
        # 宽图像，以高度为准
        view_height = max_size / fill_ratio
        view_width = view_height * aspect_ratio
    else:
        # 高图像，以宽度为准
        view_width = max_size / fill_ratio
        view_height = view_width / aspect_ratio
    
    # 使用较大的视野尺寸来计算距离
    view_size = max(view_width, view_height)
    
    # 计算距离
    distance = (view_size / 2.0) / tan_half_fov
    
    return distance


def generate_camera_poses(num_cols=12, num_rows=6, total_angle=15.0, distance=3.0):
    """生成相机位姿（在横向和纵向各15°范围内均匀分布，用于3D显示屏）
    
    Args:
        num_cols: 列数（横向视角数）
        num_rows: 行数（纵向视角数）
        total_angle: 总视角范围（度），横向和纵向都在这个范围内
        distance: 相机距离
    """
    poses = []
    
    # 计算物体中心（假设在原点）
    center = np.array([0.0, 0.0, 0.0])
    
    # 计算角度范围：从 -total_angle/2 到 +total_angle/2
    angle_range = total_angle  # 总范围（例如15°）
    start_angle = -angle_range / 2.0  # 起始角度（例如-7.5°）
    end_angle = angle_range / 2.0     # 结束角度（例如+7.5°）
    
    # 计算每列/行的角度间隔
    if num_cols > 1:
        horizontal_step = angle_range / (num_cols - 1)
    else:
        horizontal_step = 0.0
    
    if num_rows > 1:
        vertical_step = angle_range / (num_rows - 1)
    else:
        vertical_step = 0.0
    
    # 默认视角：正视图（从正前方看，能看到椅子腿和椅子背）
    # 相机高度在物体中间位置，稍微向下看
    camera_height = 0.3  # 相机高度（相对于物体中心，正值表示在物体上方）
    default_pitch = -10.0  # 向下倾斜10度（能看到椅子腿）
    
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算水平角度（左右旋转，绕Y轴）：从 -total_angle/2 到 +total_angle/2
            if num_cols == 1:
                horizontal_angle = 0.0
            else:
                horizontal_angle = start_angle + col * horizontal_step
            
            # 计算垂直角度（上下旋转，绕X轴）
            # 基础角度是向下倾斜（能看到椅子腿），然后在此基础上微调
            if num_rows == 1:
                vertical_angle = default_pitch
            else:
                # 在默认向下倾斜的基础上，在±total_angle/2范围内变化
                vertical_angle = default_pitch + (end_angle - row * vertical_step)
            
            # 转换为弧度
            h_rad = np.radians(horizontal_angle)  # 水平旋转角度（绕Y轴）
            v_rad = np.radians(vertical_angle)    # 垂直旋转角度（绕X轴）
            
            # 计算相机位置
            # 相机沿着Y轴正方向（从上方看），但稍微向前偏移以看到正面
            # 基础位置：在Y轴正方向，距离为distance，稍微向前（Z轴正方向）
            base_pos = np.array([0.0, distance, distance * 0.3])  # Y轴正方向，稍微向前
            
            # 绕Y轴旋转（水平旋转，左右看）
            cos_h = np.cos(h_rad)
            sin_h = np.sin(h_rad)
            rot_y = np.array([
                [cos_h, 0, sin_h],
                [0, 1, 0],
                [-sin_h, 0, cos_h]
            ])
            pos_after_h = rot_y @ base_pos
            
            # 绕X轴旋转（垂直旋转，上下看）
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)
            rot_x = np.array([
                [1, 0, 0],
                [0, cos_v, -sin_v],
                [0, sin_v, cos_v]
            ])
            eye = rot_x @ pos_after_h
            
            # 目标点稍微向下（能看到椅子背和椅子腿）
            target = center + np.array([0.0, -0.2, 0.0])  # 稍微向下看
            # 从Y轴正方向看时，up向量应该是Z轴负方向，但需要翻转以修正上下颠倒
            up = np.array([0.0, 0.0, 1.0])  # Z轴正方向作为up（修正上下颠倒）
            
            poses.append({
                'eye': eye,
                'target': target,
                'up': up,
                'row': row,
                'col': col,
                'horizontal_angle': horizontal_angle,
                'vertical_angle': vertical_angle
            })
    
    return poses


def render_with_rasterizer(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """使用diff-gaussian-rasterization渲染"""
    if not HAS_RASTERIZER:
        return None
    
    # 转换为torch tensor
    means3D = torch.from_numpy(gaussians['positions']).cuda().float()
    colors = torch.from_numpy(gaussians['colors']).cuda().float()
    opacity = torch.from_numpy(gaussians['opacities']).cuda().float()
    scales = torch.from_numpy(gaussians['scales']).cuda().float()
    rotations = torch.from_numpy(gaussians['rotations']).cuda().float()
    
    # 创建视图和投影矩阵
    view_matrix, proj_matrix = create_camera_matrices(
        width, height, fov,
        camera_pose['eye'],
        camera_pose['target'],
        camera_pose['up']
    )
    
    view_matrix = torch.from_numpy(view_matrix).cuda().float()
    proj_matrix = torch.from_numpy(proj_matrix).cuda().float()
    
    # 创建光栅化设置
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
    
    # 创建光栅化器
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # 渲染
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
    
    # 转换为numpy数组
    image = rendered_image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    
    # 从Y轴正方向看时，需要垂直翻转图像以修正上下颠倒（让椅子腿朝下）
    image = np.flipud(image)
    
    return image


def render_with_open3d(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """使用Open3D渲染点云（将高斯点作为点云处理）"""
    if not HAS_OPEN3D:
        return None
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    positions = gaussians['positions']
    colors = gaussians['colors']
    
    # 确保颜色值在[0, 1]范围内
    colors = np.clip(colors, 0.0, 1.0)
    
    # 验证颜色数据
    if not hasattr(render_with_open3d, '_color_info_printed'):
        if len(colors) > 0:
            print(f"Open3D渲染: 使用 {len(colors)} 个点的颜色")
            print(f"  颜色范围: R[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}], "
                  f"G[{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}], "
                  f"B[{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
            # 检查是否有颜色变化
            color_std = colors.std(axis=0)
            print(f"  颜色标准差: R={color_std[0]:.3f}, G={color_std[1]:.3f}, B={color_std[2]:.3f}")
            if color_std.max() < 0.01:
                print("  警告: 颜色变化很小，可能所有点都是相似颜色")
        render_with_open3d._color_info_printed = True
    
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 验证点云是否有颜色
    if not pcd.has_colors():
        print("警告: 点云没有颜色属性！")
    
    # 尝试使用离屏渲染器（适用于无头服务器）
    try:
        # 使用新的渲染API（Open3D >= 0.13.0）
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # 设置材质 - 使用支持颜色的shader
        material = o3d.visualization.rendering.MaterialRecord()
        material.point_size = 2.0
        # 使用defaultUnlit shader，它应该使用顶点颜色
        # 注意：某些Open3D版本可能需要使用"defaultLit"或"unlitSolidColor"
        material.shader = "defaultUnlit"
        
        # 确保点云有颜色
        if not pcd.has_colors():
            print("错误: 点云没有颜色属性，无法渲染颜色！")
            # 尝试重新设置颜色
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 添加几何体
        renderer.scene.add_geometry("pointcloud", pcd, material)
        
        # 确保渲染器使用颜色
        # 某些Open3D版本可能需要显式启用颜色渲染
        try:
            # 尝试设置渲染选项
            render_option = renderer.scene.get_render_option()
            if hasattr(render_option, 'point_size'):
                render_option.point_size = 2.0
        except:
            pass
        
        # 设置背景色
        renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
        
        # 计算相机参数
        eye = camera_pose['eye']
        target = camera_pose['target']
        up = camera_pose['up']
        
        # 计算视图方向
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_corrected = np.cross(right, forward)
        
        # 设置相机
        center = target
        
        # 计算焦距
        fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
        
        # 设置相机（使用 look_at 方法）
        # Open3D 的 look_at 参数顺序：center, eye, up
        renderer.scene.camera.look_at(center, eye, up_corrected)
        
        # 设置投影参数（尝试不同的API版本）
        try:
            # 方法1: 使用 set_projection (新版本)
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
            # 方法2: 使用 set_projection (旧版本，只接受内参矩阵)
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
                # 如果都失败，使用默认投影
                pass
        
        # 渲染
        image = renderer.render_to_image()
        image = np.asarray(image)
        
        # Open3D的render_to_image可能返回RGBA格式，需要转换为RGB
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA格式，提取RGB
                image = image[:, :, :3]
            elif image.shape[2] == 3:
                # 已经是RGB格式
                pass
            else:
                print(f"警告: 意外的图像通道数: {image.shape[2]}")
        
        # 确保图像值在[0, 1]范围内，然后转换为[0, 255]
        if image.max() <= 1.0:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 验证渲染后的图像是否有颜色
        if not hasattr(render_with_open3d, '_render_check_done'):
            if len(image.shape) == 3 and image.shape[2] >= 3:
                img_rgb = image[:, :, :3].astype(np.float32) / 255.0
                print(f"渲染后图像颜色范围: R[{img_rgb[:, :, 0].min():.3f}, {img_rgb[:, :, 0].max():.3f}], "
                      f"G[{img_rgb[:, :, 1].min():.3f}, {img_rgb[:, :, 1].max():.3f}], "
                      f"B[{img_rgb[:, :, 2].min():.3f}, {img_rgb[:, :, 2].max():.3f}]")
                # 检查图像是否主要是黑色
                if img_rgb.max() < 0.1:
                    print("警告: 渲染后的图像几乎完全是黑色！可能是Open3D离屏渲染器不支持点云颜色")
                # 检查是否有颜色变化
                color_std = img_rgb.std(axis=(0, 1))
                print(f"渲染图像颜色标准差: R={color_std[0]:.3f}, G={color_std[1]:.3f}, B={color_std[2]:.3f}")
            render_with_open3d._render_check_done = True
        
        # 从Y轴正方向看时，可能需要翻转图像
        # 如果椅子腿朝上，尝试不翻转或反向翻转
        # image = np.flipud(image)  # 暂时注释掉，看效果
        
        return image
        
    except (AttributeError, RuntimeError, Exception) as e:
        # 如果离屏渲染失败，检查是否有DISPLAY环境变量
        has_display = os.environ.get('DISPLAY') is not None
        
        if has_display:
            # 如果有DISPLAY，尝试使用传统方法
            try:
                # 创建可视化器
                vis = o3d.visualization.Visualizer()
                success = vis.create_window(width=width, height=height, visible=False)
                
                if not success:
                    raise RuntimeError("Failed to create window")
                
                vis.add_geometry(pcd)
                
                # 设置渲染选项
                render_option = vis.get_render_option()
                if render_option is None:
                    raise RuntimeError("Failed to get render option")
                
                render_option.background_color = np.asarray([0.0, 0.0, 0.0])
                render_option.point_size = 2.0
                
                # 设置相机参数
                ctr = vis.get_view_control()
                
                # 计算相机参数
                eye = camera_pose['eye']
                target = camera_pose['target']
                up = camera_pose['up']
                
                # 计算视图方向
                forward = target - eye
                forward = forward / np.linalg.norm(forward)
                
                # 使用Open3D的相机参数设置
                param = ctr.convert_to_pinhole_camera_parameters()
                
                # 计算视图矩阵（Open3D使用列主序）
                right = np.cross(forward, up)
                right = right / np.linalg.norm(right)
                up_corrected = np.cross(right, forward)
                
                # 构建外参矩阵（从世界到相机）
                extrinsic = np.eye(4)
                extrinsic[0:3, 0] = right
                extrinsic[0:3, 1] = -up_corrected
                extrinsic[0:3, 2] = -forward
                extrinsic[0:3, 3] = -np.dot(extrinsic[0:3, 0:3], eye)
                
                param.extrinsic = extrinsic
                
                # 设置内参（基于FOV）
                fx = fy = width / (2.0 * np.tan(np.radians(fov) / 2.0))
                cx = width / 2.0
                cy = height / 2.0
                
                param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
                ctr.convert_from_pinhole_camera_parameters(param)
                
                # 渲染
                vis.poll_events()
                vis.update_renderer()
                
                # 捕获图像
                image = vis.capture_screen_float_buffer(do_render=True)
                image = np.asarray(image)
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                
                vis.destroy_window()
                
                # 从Y轴正方向看时，需要垂直翻转图像以修正上下颠倒
                image = np.flipud(image)
                
                return image
                
            except Exception as e2:
                print(f"Warning: Open3D GUI rendering also failed: {e2}")
                return None
        else:
            # 无头服务器，离屏渲染失败，返回None让代码使用简单渲染方法
            print(f"Warning: Open3D offscreen rendering failed (no DISPLAY): {e}")
            return None


def render_simple(gaussians, camera_pose, width=800, height=600, fov=60.0):
    """简单的渲染方法（当没有其他渲染器时使用）- 使用numpy直接渲染点云颜色"""
    # 计算相机参数
    eye = camera_pose['eye']
    target = camera_pose['target']
    up = camera_pose['up']
    
    # 计算视图方向
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up_corrected = np.cross(right, forward)
    
    # 获取数据
    positions = gaussians['positions']
    colors = gaussians['colors']
    opacities = gaussians['opacities']
    scales = gaussians['scales']
    
    # 确保颜色在[0, 1]范围内，但不压缩颜色范围
    # 如果颜色值超出范围，只裁剪，不要重新归一化（保持原始鲜艳度）
    colors = np.clip(colors, 0.0, 1.0)
    
    # 创建图像和深度缓冲区
    image = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    # 计算焦距
    focal_length = width / (2.0 * np.tan(np.radians(fov) / 2.0))
    
    # 计算所有点相对于相机的坐标
    rel_positions = positions - eye
    depths = np.dot(rel_positions, forward)
    
    # 只处理在相机前方的点
    valid_mask = depths > 0.01
    
    if not np.any(valid_mask):
        return (image * 255).astype(np.uint8)
    
    valid_positions = rel_positions[valid_mask]
    valid_depths = depths[valid_mask]
    valid_colors = colors[valid_mask]
    valid_opacities = opacities[valid_mask]
    valid_scales = scales[valid_mask]
    
    # 投影到屏幕空间（向量化计算）
    x_coords = np.dot(valid_positions, right) * focal_length / valid_depths
    y_coords = np.dot(valid_positions, up_corrected) * focal_length / valid_depths
    
    # 转换为像素坐标
    # 注意：y坐标需要翻转，因为图像坐标系Y轴向下
    pixel_x = (x_coords + width / 2).astype(int)
    pixel_y = (-y_coords + height / 2).astype(int)  # 取负号以正确映射
    
    # 根据点的尺寸计算渲染半径（简化处理）
    point_sizes = np.max(valid_scales, axis=1) * focal_length / valid_depths
    point_sizes = np.clip(point_sizes, 1, 10).astype(int)
    
    # 按深度排序（从远到近）
    sort_indices = np.argsort(-valid_depths)
    
    # 渲染每个点
    for idx in sort_indices:
        px, py = pixel_x[idx], pixel_y[idx]
        size = point_sizes[idx]
        color = valid_colors[idx]
        alpha = min(valid_opacities[idx], 1.0)
        depth = valid_depths[idx]
        
        # 在点周围绘制一个小的圆形区域
        half_size = size // 2
        y_start = max(0, py - half_size)
        y_end = min(height, py + half_size + 1)
        x_start = max(0, px - half_size)
        x_end = min(width, px + half_size + 1)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 简单的圆形检查
                dx, dy = x - px, y - py
                if dx*dx + dy*dy <= half_size*half_size:
                    if depth < depth_buffer[y, x]:
                        # Alpha混合
                        image[y, x] = image[y, x] * (1 - alpha) + color * alpha
                        depth_buffer[y, x] = depth
    
    # 转换为uint8
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # 由于我们修改了y坐标的计算方式，不需要再翻转
    # 如果椅子腿还是朝上，可能需要移除所有翻转或调整up向量
    
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
        
        # 调整图像大小
        if img.shape[0] != img_height or img.shape[1] != img_width:
            img = Image.fromarray(img)
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            img = np.array(img)
        
        # 放置到网格中
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        
        grid_image[y_start:y_end, x_start:x_end] = img
    
    return grid_image


def main():
    parser = argparse.ArgumentParser(description='3DGS多视点宫格图采样')
    parser.add_argument('--input', type=str, required=True, help='输入的.ply文件路径')
    parser.add_argument('--output', type=str, default='grid_output.png', help='输出的宫格图路径')
    parser.add_argument('--cols', type=int, default=12, help='列数（横向视角数，默认12）')
    parser.add_argument('--rows', type=int, default=6, help='行数（纵向视角数，默认6）')
    parser.add_argument('--angle', type=float, default=15.0, help='总视角范围（度），横向和纵向都在这个范围内（默认15°）')
    parser.add_argument('--distance', type=float, default=None, help='相机距离（如果为None则自动计算，默认None）')
    parser.add_argument('--fill_ratio', '--fill-ratio', type=float, default=0.8, help='物体占据图像的比例（0-1，默认0.8表示80%%，仅在distance为None时有效）')
    parser.add_argument('--width', type=int, default=800, help='单个图像宽度（默认800）')
    parser.add_argument('--height', type=int, default=600, help='单个图像高度（默认600）')
    parser.add_argument('--fov', type=float, default=60.0, help='视野角度（度，默认60°）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"加载3DGS模型: {args.input}")
    gaussians = load_ply(args.input)
    print(f"加载了 {len(gaussians['positions'])} 个高斯点")
    
    # 计算或使用指定的相机距离
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
    
    # 生成相机位姿
    print(f"生成 {args.cols}x{args.rows} 的相机位姿...")
    print(f"视角范围：横向和纵向各在 ±{args.angle/2:.1f}° 范围内（总共 {args.angle}°）")
    camera_poses = generate_camera_poses(
        num_cols=args.cols,
        num_rows=args.rows,
        total_angle=args.angle,
        distance=distance
    )
    
    # 渲染每个视角
    print("开始渲染...")
    use_simple_render = False
    
    if HAS_RASTERIZER and torch.cuda.is_available():
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
        print(f"渲染 {i+1}/{total} (行{pose['row']+1}, 列{pose['col']+1})...")
        
        img = None
        
        if not use_simple_render and HAS_RASTERIZER and torch.cuda.is_available():
            img = render_with_rasterizer(
                gaussians, pose,
                width=args.width,
                height=args.height,
                fov=args.fov
            )
        elif not use_simple_render and HAS_OPEN3D and not open3d_failed:
            # 尝试使用Open3D
            img = render_with_open3d(
                gaussians, pose,
                width=args.width,
                height=args.height,
                fov=args.fov
            )
            
            # 检查Open3D渲染结果是否有颜色
            if img is not None and i == 0:  # 只在第一张图时检查
                img_check = img.astype(np.float32) / 255.0
                if img_check.max() < 0.1 or img_check.std() < 0.01:
                    print("检测到Open3D渲染结果为黑白，切换到numpy渲染方法...")
                    open3d_failed = True
                    use_simple_render = True
                    # 重新渲染第一张图
                    img = render_simple(
                        gaussians, pose,
                        width=args.width,
                        height=args.height,
                        fov=args.fov
                    )
        
        # 如果Open3D失败或没有使用Open3D，使用简单渲染
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
    
    # 创建宫格图
    print("组合宫格图...")
    grid_image = create_grid_image(
        images,
        num_cols=args.cols,
        num_rows=args.rows,
        img_width=args.width,
        img_height=args.height
    )
    
    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    grid_pil = Image.fromarray(grid_image)
    grid_pil.save(output_path)
    print(f"宫格图已保存到: {output_path}")
    print(f"图像尺寸: {grid_image.shape[1]}x{grid_image.shape[0]}")


if __name__ == '__main__':
    main()


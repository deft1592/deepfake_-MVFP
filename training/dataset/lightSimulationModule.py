import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from DECA.decalib.deca import DECA
from DECA.decalib.utils import util
from DECA.decalib.utils.config import cfg as deca_cfg
from skimage.io import imsave
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import os
import multiprocessing as mp
import torch.nn.functional as F


# 常见的光照方向示例
LIGHT_DIRECTIONS = {
    "正面光": [0, 0, 1],
    "右侧光": [0.7, 0.2, 0.5],
    "左侧光": [-0.7, 0.2, 0.5],
    "顶部光": [0, 0.9, 0.1],
    "底部光": [0, -0.9, 0.1],
    "右上前方光": [0.5, 0.5, 0.7],
    "左上前方光": [-0.5, 0.5, 0.7],
}



class DECALightParameterGenerator:
    """
    DECA光照参数生成器，用于生成合理范围内的随机光照参数。
    基于球谐光照模型，特别针对DECA模型的光照参数结构进行了简化。
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def generate_random_sh_direction(self, batch_size=1):
        """
        生成随机的球谐光照方向系数（前9个系数）
        策略：在合理的球谐系数范围内随机采样，模拟不同方向的光照
        """
        # 基础方向系数范围（根据经验值设定）
        sh_basis_range = 0.3
        sh_basis_center = 0.0
        
        # 生成前9个球谐系数（3阶球谐，3个颜色通道）
        sh_coeffs = torch.randn(batch_size, 9, device=self.device) * sh_basis_range + sh_basis_center
        
        # 对主要方向系数进行适当加强，确保光照有明确方向性
        sh_coeffs[:, 0] = 0.8  # L0系数，基础光照
        sh_coeffs[:, 1:4] = sh_coeffs[:, 1:4] * 0.5  # L1系数，主要方向
        
        return sh_coeffs
    
    def generate_random_light_color(self, batch_size=1):
        """
        生成随机的光照颜色（RGB系数）
        返回：环境光颜色和主要光照颜色
        """
        # 环境光颜色 - 偏向中性色，强度较低
        ambient_color = torch.rand(batch_size, 3, device=self.device) * 0.3 + 0.1
        
        # 主要光照颜色 - 可以有一些随机变化，但避免极端颜色
        light_color = torch.rand(batch_size, 3, device=self.device) * 0.4 + 0.6  # 偏向白色
        
        return ambient_color, light_color
    
    def generate_random_light_parameters(self, batch_size=1, reference_light=None):
        """
        生成完整的随机光照参数
        """
        if reference_light is not None:
            # 基于参考光照的参数形状创建新参数
            light_params = reference_light.clone()
        else:
            # 默认创建27维的光照参数（3阶球谐 x 3个颜色通道 x 3组？）
            # 具体维度可能需要根据您的DECA版本调整
            light_params = torch.zeros(batch_size, 27, device=self.device)
        
        # 生成随机球谐方向系数
        sh_direction = self.generate_random_sh_direction(batch_size)
        light_params[:, :9] = sh_direction
        
        # 生成随机颜色
        ambient_color, main_light_color = self.generate_random_light_color(batch_size)
        
        # 设置环境光颜色（通常在第9-12个参数）
        if light_params.shape[1] > 11:
            light_params[:, 9:12] = ambient_color
        
        # 设置主要光照颜色（如果参数维度允许）
        if light_params.shape[1] > 14:
            light_params[:, 12:15] = main_light_color
        
        return light_params
    
    def generate_structured_lighting(self, light_type="random", intensity=1.0):
        """
        生成特定类型的结构化光照
        light_type: 光照类型，可选 "random", "frontal", "side", "top", "bottom", "dramatic"
        intensity: 光照强度乘子
        """
        batch_size = 1
        
        # 初始化基础参数
        light_params = torch.zeros(batch_size, 27, device=self.device)
        
        # 基础环境光
        light_params[:, 9:12] = torch.tensor([0.3, 0.3, 0.3])  # 灰色环境光
        
        if light_type == "frontal":
            # 正面光 - 较小的L1系数
            light_params[:, 1:4] = torch.tensor([0.0, 0.0, 0.5]) * intensity
            
        elif light_type == "side":
            # 侧光 - X方向有较强的系数
            side_intensity = 0.7 * intensity
            light_params[:, 1:4] = torch.tensor([side_intensity, 0.1, 0.1])
            
        elif light_type == "top":
            # 顶光 - Y方向有较强的系数
            light_params[:, 1:4] = torch.tensor([0.0, 0.7 * intensity, 0.0])
            
        elif light_type == "bottom":
            # 底光 - Y负方向
            light_params[:, 1:4] = torch.tensor([0.0, -0.5 * intensity, 0.0])
            
        elif light_type == "dramatic":
            # 戏剧性光照 - 较强的方向性光
            dramatic_dir = torch.randn(3)
            dramatic_dir = F.normalize(dramatic_dir, dim=0)
            light_params[:, 1:4] = dramatic_dir * 0.8 * intensity
            
        else:  # random
            return self.generate_random_light_parameters(batch_size)
        
        return light_params * intensity
    
class DECALightEditor:

    def __init__(self, device=None):
       
        self.device = self._get_current_device()
        self.deca = None  # 延迟初始化
        self._initialized = False
        print(f"进程 {os.getpid()} 使用设备: {self.device}")


        # 添加清理方法
    def cleanup(self):
        """清理DECA模型和相关资源"""
        if self.deca is not None:
            # 删除DECA模型
            del self.deca
            self.deca = None
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        self._initialized = False
        print(f"进程 {os.getpid()} 的DECA资源已清理")
        
    def _ensure_initialized(self):
        if not self._initialized:
            if self.deca is None:
                self.deca = DECA(config=deca_cfg, device=self.device)
                self.deca = self.deca.to(self.device)
            self._initialized = True
    def _get_current_device(self):
        """获取当前进程的设备"""
        # 方法1: 检查是否在分布式环境中
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # 在分布式训练中，使用当前进程的设备
            return f'cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}'
        
        # 方法2: 检查当前CUDA设备
        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                return f'cuda:{current_device}'
            except:
                # 如果无法获取当前设备，尝试设置一个
                for i in range(torch.cuda.device_count()):
                    try:
                        # 检查设备是否可用
                        torch.cuda.get_device_properties(i)
                        return f'cuda:{i}'
                    except:
                        continue
        
        # 方法3: 回退到CPU
        return 'cpu'
    

    def load_image(self, image_path):
        """加载并预处理图像"""
        # 读取图像并转换为RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def process_image(self, image):
        """使用DECA处理图像"""
        # 转换为torch张量并进行预处理
        image_tensor = torch.tensor(image, device=self.device).float() / 255.
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        print(image_tensor.shape)
        # 使用DECA进行3D重建
        with torch.no_grad():
            codedict = self.deca.encode(image_tensor)
            opdict, visdict = self.deca.decode(codedict)
        
        return codedict, opdict, visdict
    

    def modify_lighting(self, codedict, light_direction=None, ambient_color=None):
        """修改光照参数"""
        # 创建光照参数的副本
        new_light_code = codedict['light'].clone()
        
        # 随机生成光照方向（如果没有提供）
        if light_direction is None:
            azimuth = np.random.uniform(0, 2*np.pi)  # 方位角
            elevation = np.random.uniform(0, np.pi/2)  # 仰角
            x = np.cos(elevation) * np.sin(azimuth)
            y = np.sin(elevation)
            z = np.cos(elevation) * np.cos(azimuth)
            light_direction = [x, y, z]
        
        # 将方向向量转换为球谐系数（简化版）
        sh_coeffs = self.direction_to_sh(light_direction)
        
        # 修复维度问题 - 确保维度匹配
        # 将sh_coeffs转换为正确的形状 [1, 9, 1] 然后扩展为 [1, 9, 3]
        sh_tensor = torch.tensor(sh_coeffs, device=self.device).float().view(1, 9, 1)
        sh_tensor = sh_tensor.expand(1, 9, 3)  # 扩展到3个通道
        
        # 更新光照参数
        new_light_code[:, :9] = sh_tensor
        
        # 修改环境光颜色
        if ambient_color is not None:
            # 确保环境光颜色有正确的维度
            ambient_tensor = torch.tensor(ambient_color, device=self.device).float().view(1, 3)
            new_light_code[:, 9:12] = ambient_tensor
        
        return new_light_code
    
    def direction_to_sh(self, direction, order=2):
        """将方向向量转换为球谐系数（简化版）"""
        x, y, z = direction
        r = np.sqrt(x*x + y*y + z*z)
        x, y, z = x/r, y/r, z/r
        
        sh_coeffs = []
        if order >= 0:
            sh_coeffs.extend([1.0])  # L0
        if order >= 1:
            sh_coeffs.extend([y, z, x])  # L1
        if order >= 2:
            sh_coeffs.extend([x*y, y*z, 3*z*z-1, x*z, x*x-y*y])  # L2
        
        return np.array(sh_coeffs[:9])  # 取前9个系数
    

    
    def save_image(self, image, output_path):
        """保存图像到本地"""
        # 确保输出目录存在
        
        # 保存图像
        print(image.shape)
        imsave(output_path, image)
        print(f"图像已保存至: {output_path}")


    def compute_normals_from_vertices(self, vertices, faces):
        """从3D顶点计算法线图"""
        # 确保顶点和面片在正确的设备上
        vertices = vertices.to(self.device)
        faces = faces.to(self.device)
        
        # 创建网格对象
        mesh = Meshes(verts=[vertices], faces=[faces])
        
        # 计算顶点法线
        vertex_normals = mesh.verts_normals_packed().unsqueeze(0)
        
        return vertex_normals

    def render_normal_map(self, vertices, faces, image_size=224):
        """渲染法线图"""
        # 计算顶点法线
        vertex_normals = self.compute_normals_from_vertices(vertices, faces)
        
        # 使用DECA的渲染器渲染法线图
        # 这里需要根据DECA的具体实现来调整
        # 以下是一个简化的实现
        
        # 创建一个伪纹理（使用法线作为颜色）
        normals_rgb = (vertex_normals + 1) / 2  # 将法线从[-1,1]映射到[0,1]
        textures = TexturesVertex(verts_features=normals_rgb)
        
        # 创建网格
        mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
        
        # 使用DECA的相机参数进行渲染
        # 这里需要获取DECA的相机参数
        # 以下是一个简化的实现，您可能需要根据DECA的具体实现进行调整
        
        # 使用PyTorch3D的渲染器
        from pytorch3d.renderer import (
            FoVPerspectiveCameras,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader
        )
        
        # 创建相机（使用DECA的相机参数）
        # 这里需要从DECA的codedict中获取相机参数
        cameras = FoVPerspectiveCameras(device=self.device)
        
        # 创建渲染器
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras
            )
        )
        
        # 渲染法线图
        with torch.no_grad():
            normal_map = renderer(mesh)
        
        # 提取法线图（去除alpha通道）
        normal_map = normal_map[0, ..., :3].cpu().numpy()
        
        return normal_map

    def render_normal_map_optimized(self, vertices, faces, image_size=224):
        """优化版本的法线图渲染，减少显存消耗"""
        try:
            # 计算顶点法线
            with torch.no_grad():
                vertices_tensor = vertices.unsqueeze(0).to(self.device)
                faces_tensor = faces.to(self.device)
                
                # 创建网格对象
                mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor])
                
                # 计算法线
                vertex_normals = mesh.verts_normals_packed().unsqueeze(0)
                
                # 将法线转换为纹理
                normals_rgb = (vertex_normals + 1) / 2
                textures = TexturesVertex(verts_features=normals_rgb)
                
                # 更新网格纹理
                mesh.textures = textures
                
                # 简化渲染设置
                from pytorch3d.renderer import (
                    FoVPerspectiveCameras,
                    RasterizationSettings,
                    MeshRenderer,
                    MeshRasterizer,
                    SoftPhongShader
                )
                
                # 使用简化的相机和渲染设置
                cameras = FoVPerspectiveCameras(device=self.device)
                raster_settings = RasterizationSettings(
                    image_size=image_size,
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    max_faces_per_bin=10000,  # 限制每bin的面数
                )
                
                renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings
                    ),
                    shader=SoftPhongShader(
                        device=self.device,
                        cameras=cameras
                    )
                )
                
                # 渲染
                normal_map = renderer(mesh)
                normal_map = normal_map[0, ..., :3].cpu().numpy()
                
                # 清理显存
                del mesh, vertices_tensor, faces_tensor, vertex_normals, normals_rgb, textures
                del cameras, raster_settings, renderer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return normal_map
                
        except Exception as e:
            print(f"法线图渲染失败: {e}")
            # 返回默认法线图（全正面）
            return np.ones((image_size, image_size, 3), dtype=np.float32) * 0.5
   
    def apply_3d_informed_lighting_with_computed_normals(self, original_image, light_direction, mask,intensity=1.0):
        
        self._ensure_initialized()
        
        image_tensor = torch.tensor(original_image, device=self.device).float() / 255.
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        print(image_tensor.device)
        print(self.deca.device)


        with torch.no_grad():
            print("start deca encode")
            codedict = self.deca.encode(image_tensor)
            print("start deca decode")
            
            opdict, visdict = self.deca.decode(codedict)
            print("deca decode done ")

        # # 获取法线图  
        if 'normal_images' in opdict:
            # 如果DECA直接提供了法线图，使用它
            normal_map = opdict['normal_images'].squeeze().cpu().numpy()
            normal_map = normal_map.transpose(1, 2, 0)
            print("使用DECA提供的法线图")
        else:
            # 否则尝试从顶点计算法线图
            
            # 获取顶点
            vertices = opdict['verts']  # 3D顶点
             
            faces = self.deca.flame.faces_tensor

            # 计算并渲染法线图
            normal_map = self.render_normal_map(vertices.squeeze(0), faces)
            print("使用计算的法线图")
            
 
         # 清理显存
        del image_tensor, codedict, visdict

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        
        #调整法线图大小以匹配原始图像
        h, w = original_image.shape[:2]
        normal_map_resized = cv2.resize(normal_map, (w, h))
        # 拿到albedo 图
        
        # 创建与图片相同尺寸的结果数组
        result = np.zeros_like(original_image)
        # 将mask区域内的像素值复制到结果中
        mask_bool = mask > 0  # 创建布尔掩码
        result[mask_bool] = original_image[mask_bool]

        img_float = result.astype(np.float32) / 255.0
        
       
        # 使用法线图应用光照
        lit_image = self.apply_lighting_with_normals(img_float, normal_map_resized, light_direction, intensity)
        

        print("apply light done ")

        del normal_map, normal_map_resized

        binary_mask= np.expand_dims(mask, axis=2)
 
        # 将结果转换回uint8类型
        result_image = (lit_image * 255).astype(np.uint8)
        binary_mask=binary_mask.astype(np.uint8)

        final_image=result_image* binary_mask +  (1-binary_mask)*original_image
        
        return final_image.astype(np.uint8)

    def apply_lighting_with_normals(self, img_float, normal_map, light_direction, intensity=1.0):
        """使用法线图应用光照"""
        # 确保法线图是单位向量
        # 法线图的范围是[0,1]，需要映射回[-1,1]
        normal_map = normal_map * 2 - 1
        
        # 归一化法线
        norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norm = np.where(norm == 0, 1, norm)  # 避免除以零
        unit_normals = normal_map / norm
        
        # 归一化光照方向
        light_dir = np.array(light_direction)
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # 计算每个像素的亮度（朗伯反射模型）
        # 亮度 = max(0, 法线 · 光照方向)
        brightness = np.sum(unit_normals * light_dir, axis=2)
        brightness = np.clip(brightness, 0.2, 1.5)  # 限制亮度范围
        
        # 添加环境光
        ambient = 0.3
        brightness = ambient + (1 - ambient) * brightness
        
        # 应用亮度调整
        lit_image = img_float * brightness[..., np.newaxis] * intensity
        
        return np.clip(lit_image, 0, 1)


# 使用示例
if __name__ == "__main__":
    # 初始化光照编辑器
    editor = DECALightEditor(device='cuda:1')
    
    # 输入图像路径
    input_image_path = "augmented_image.png"
    output_image_path = "augImg_lightModified.png"
    
    original_image = editor.load_image(input_image_path)

    light_direction = [0.5, 0.3, 0.8]  # 右侧上方光   

    modified_image = editor.apply_3d_informed_lighting_with_computed_normals(original_image, light_direction, intensity=1.2)

    
    editor.save_image(modified_image, output_image_path)

    # # 2. 处理图像（3D重建）
    # codedict, opdict, visdict = editor.process_image(original_image)
    # # 示例3: 随机光照
    # new_light = editor.modify_lighting(codedict)
    
    # # 4. 使用新光照渲染图像
    # modified_image = editor.render_with_new_light(codedict, new_light,original_image)
    
    # # 打印图像形状以确认
    # print(f"修改后图像形状: {modified_image.shape}")
    
    # # 5. 保存结果
    # editor.save_image(modified_image, output_image_path)
    
  
#!/usr/bin/env python3
"""
五子棋游戏PNG图片序列转视频工具 (FFmpeg版本)

使用FFmpeg将每个文件夹中的图片序列（move_000.png, move_001.png, ...）转换为对应的视频文件。
相比OpenCV版本，此实现具有更高的效率和更好的视频质量。
"""

import os
import argparse
import subprocess
import re
import shutil
from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import sys
import tempfile

def check_ffmpeg():
    """检查FFmpeg是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def natural_sort_key(s):
    """用于自然排序的键函数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_image_folders(root_dir):
    """获取包含图片序列的文件夹"""
    folders = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            # 检查文件夹中是否包含图片序列
            png_files = [f for f in os.listdir(entry.path) if f.endswith('.png') and f.startswith('move_')]
            if png_files:
                folders.append(entry.path)
    return folders

def convert_images_to_video_ffmpeg(folder_path, output_dir=None, fps=2, video_format='mp4', quality=23, ffmpeg_path=None):
    """使用FFmpeg将一个文件夹中的图片序列转换为视频
    
    Args:
        folder_path: 图片文件夹路径
        output_dir: 输出视频目录，如果为None则与图片文件夹相同
        fps: 视频帧率
        video_format: 视频格式，支持'mp4'和'avi'
        quality: 视频质量 (FFmpeg的CRF值, 0-51, 数值越小质量越高)
        ffmpeg_path: FFmpeg可执行文件路径
        
    Returns:
        str: 输出视频的路径
    """
    try:
        folder_name = os.path.basename(folder_path)
        if output_dir is None:
            output_dir = os.path.dirname(folder_path)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定FFmpeg路径
        ffmpeg_cmd = ffmpeg_path if ffmpeg_path else 'ffmpeg'
        
        # 确定输出文件名
        if video_format.lower() == 'mp4':
            output_file = os.path.join(output_dir, f"{folder_name}.mp4")
            codec = 'libx264'
            pix_fmt = 'yuv420p'
        elif video_format.lower() == 'avi':
            output_file = os.path.join(output_dir, f"{folder_name}.avi")
            codec = 'mpeg4'
            pix_fmt = 'yuv420p'
        elif video_format.lower() == 'webm':
            output_file = os.path.join(output_dir, f"{folder_name}.webm")
            codec = 'libvpx-vp9'
            pix_fmt = 'yuv420p'
        elif video_format.lower() == 'gif':
            output_file = os.path.join(output_dir, f"{folder_name}.gif")
            # GIF特殊处理
            return convert_to_gif(folder_path, output_file, fps, ffmpeg_cmd)
        else:
            raise ValueError(f"不支持的格式: {video_format}")
        
        # 获取PNG文件列表并排序
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.startswith('move_')]
        png_files.sort(key=natural_sort_key)
        
        if not png_files:
            print(f"警告: 文件夹 {folder_path} 中没有找到PNG图片")
            return None
        
        # 创建输入文件列表
        list_file_path = os.path.join(tempfile.gettempdir(), f"ffmpeg_list_{os.getpid()}.txt")
        with open(list_file_path, 'w') as list_file:
            for png_file in png_files:
                image_path = os.path.join(folder_path, png_file)
                # 转换为绝对路径并确保路径格式正确
                abs_path = os.path.abspath(image_path).replace('\\', '/')
                list_file.write(f"file '{abs_path}'\n")
                list_file.write(f"duration {1/fps}\n")
            
            # 对最后一帧重复写入，但不设置duration
            if png_files:
                last_image = os.path.join(folder_path, png_files[-1])
                abs_path = os.path.abspath(last_image).replace('\\', '/')
                list_file.write(f"file '{abs_path}'\n")
        
        # 构建FFmpeg命令
        command = [
            ffmpeg_cmd,
            '-y',  # 覆盖输出文件
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file_path,
            '-c:v', codec,
            '-pix_fmt', pix_fmt,
            '-crf', str(quality),
            output_file
        ]
        
        # 执行FFmpeg命令
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 删除临时文件
        try:
            os.remove(list_file_path)
        except OSError:
            pass
        
        if process.returncode != 0:
            print(f"FFmpeg错误: {process.stderr}")
            return None
        
        print(f"视频已生成: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"转换文件夹 {folder_path} 时出错: {str(e)}")
        return None

def convert_to_gif(folder_path, output_file, fps, ffmpeg_cmd):
    """特殊处理GIF格式，使用调色板优化GIF质量
    """
    try:
        # 获取PNG文件列表并排序
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and f.startswith('move_')]
        png_files.sort(key=natural_sort_key)
        
        if not png_files:
            print(f"警告: 文件夹 {folder_path} 中没有找到PNG图片")
            return None
        
        # 临时文件
        temp_dir = tempfile.mkdtemp()
        palette_file = os.path.join(temp_dir, "palette.png")
        list_file_path = os.path.join(temp_dir, "filelist.txt")
        
        # 创建输入文件列表
        with open(list_file_path, 'w') as list_file:
            for png_file in png_files:
                image_path = os.path.join(folder_path, png_file)
                abs_path = os.path.abspath(image_path).replace('\\', '/')
                list_file.write(f"file '{abs_path}'\n")
                list_file.write(f"duration {1/fps}\n")
            # 对最后一帧重复写入，但不设置duration
            if png_files:
                last_image = os.path.join(folder_path, png_files[-1])
                abs_path = os.path.abspath(last_image).replace('\\', '/')
                list_file.write(f"file '{abs_path}'\n")
        
        # 第一步：生成调色板
        palette_cmd = [
            ffmpeg_cmd,
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file_path,
            '-vf', 'palettegen=stats_mode=diff',
            palette_file
        ]
        
        subprocess.run(
            palette_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        
        # 第二步：使用调色板生成GIF
        gif_cmd = [
            ffmpeg_cmd,
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file_path,
            '-i', palette_file,
            '-lavfi', 'paletteuse=dither=sierra2_4a',
            output_file
        ]
        
        subprocess.run(
            gif_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        print(f"GIF已生成: {output_file}")
        return output_file
    
    except Exception as e:
        print(f"生成GIF时出错: {str(e)}")
        return None

def create_html_index(folders, output_files, output_dir):
    """创建HTML索引页面，用于浏览所有生成的视频
    
    Args:
        folders: 文件夹路径列表
        output_files: 对应的视频文件路径列表
        output_dir: 输出目录
    """
    html_path = os.path.join(output_dir, "index.html")
    
    # 创建相对路径
    rel_video_paths = []
    for video_file in output_files:
        if video_file:
            rel_path = os.path.relpath(video_file, output_dir).replace("\\", "/")
            rel_video_paths.append(rel_path)
        else:
            rel_video_paths.append(None)
    
    # 生成HTML内容
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>五子棋游戏视频集</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            h1 { 
                text-align: center; 
                color: #333;
                margin-bottom: 30px;
            }
            .video-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 20px; 
                margin-top: 20px; 
            }
            .video-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                background-color: white;
                transition: transform 0.2s;
            }
            .video-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            }
            .video-card h3 { 
                margin-top: 0; 
                color: #444;
                border-bottom: 1px solid #eee;
                padding-bottom: 8px;
            }
            video, img { 
                width: 100%; 
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            .stats {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .timestamp {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .filter-controls {
                margin-bottom: 20px;
                text-align: center;
            }
            .filter-controls input {
                padding: 8px 15px;
                width: 300px;
                border-radius: 20px;
                border: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <h1>五子棋游戏视频集</h1>
        
        <div class="stats">
            <p>生成时间: <span id="timestamp"></span></p>
    """
    
    # 计算成功生成的视频数量
    successful_videos = sum(1 for v in output_files if v is not None)
    html_content += f"""
            <p>共生成<strong>{successful_videos}</strong>个视频，来自<strong>{len(folders)}</strong>个图片序列文件夹</p>
        </div>
        
        <div class="filter-controls">
            <input type="text" id="searchInput" placeholder="搜索游戏ID..." oninput="filterVideos()">
        </div>
    """
    
    html_content += """
        <div class="video-grid" id="videoGrid">
    """
    
    # 添加每个视频卡片
    for i, (folder, video_path) in enumerate(zip(folders, rel_video_paths)):
        if video_path is None:
            continue
            
        folder_name = os.path.basename(folder)
        video_extension = os.path.splitext(video_path)[1].lower()
        
        html_content += f"""
            <div class="video-card" data-id="{folder_name}">
                <h3>{folder_name}</h3>
        """
        
        if video_extension == '.gif':
            html_content += f"""
                <img src="{video_path}" alt="{folder_name}">
            """
        else:
            html_content += f"""
                <video controls>
                    <source src="{video_path}" type="video/{video_extension[1:]}">
                    您的浏览器不支持HTML5视频。
                </video>
            """
            
        html_content += """
            </div>
        """
    
    html_content += """
        </div>
        
        <script>
            document.getElementById("timestamp").textContent = new Date().toLocaleString();
            
            function filterVideos() {
                const searchTerm = document.getElementById("searchInput").value.toLowerCase();
                const cards = document.querySelectorAll(".video-card");
                
                cards.forEach(card => {
                    const id = card.getAttribute("data-id").toLowerCase();
                    if (id.includes(searchTerm)) {
                        card.style.display = "";
                    } else {
                        card.style.display = "none";
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"HTML索引页面已生成: {html_path}")
    return html_path

def process_all_folders(args):
    """处理所有图片文件夹"""
    # 检查FFmpeg是否已安装
    if not check_ffmpeg():
        print("错误: 找不到FFmpeg。请确保FFmpeg已安装并添加到系统PATH中，或使用--ffmpeg_path选项指定FFmpeg路径。")
        return
        
    # 获取所有包含图片的文件夹
    folders = get_image_folders(args.input_dir)
    
    if not folders:
        print(f"在 {args.input_dir} 中没有找到包含图片序列的文件夹")
        return
    
    print(f"找到 {len(folders)} 个包含图片序列的文件夹")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换每个文件夹中的图片为视频
    output_files = []
    
    if args.parallel:
        print("使用并行处理模式...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for folder in folders:
                future = executor.submit(
                    convert_images_to_video_ffmpeg, 
                    folder, 
                    args.output_dir, 
                    args.fps,
                    args.format,
                    args.quality,
                    args.ffmpeg_path
                )
                futures.append(future)
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="处理文件夹"):
                output_files.append(future.result())
    else:
        print("使用顺序处理模式...")
        for folder in tqdm(folders, desc="处理文件夹"):
            output_file = convert_images_to_video_ffmpeg(
                folder, 
                args.output_dir, 
                args.fps,
                args.format,
                args.quality,
                args.ffmpeg_path
            )
            output_files.append(output_file)
    
    # 创建HTML索引
    if args.create_html:
        create_html_index(folders, output_files, args.output_dir)
    
    # 统计
    successful = sum(1 for f in output_files if f is not None)
    print(f"处理完成: 成功 {successful}/{len(folders)} 个文件夹")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用FFmpeg将五子棋PNG图片序列转换为视频")
    parser.add_argument("--input_dir", type=str, default=".", help="输入目录，包含多个图片序列文件夹")
    parser.add_argument("--output_dir", type=str, default="gomoku_videos", help="输出视频目录")
    parser.add_argument("--fps", type=float, default=2, help="视频帧率")
    parser.add_argument("--format", type=str, choices=["mp4", "avi", "webm", "gif"], default="mp4", help="视频格式")
    parser.add_argument("--quality", type=int, default=23, help="视频质量 (0-51，数值越小质量越高)")
    parser.add_argument("--parallel", action="store_true", help="启用并行处理")
    parser.add_argument("--workers", type=int, default=None, help="并行工作进程数量")
    parser.add_argument("--create_html", action="store_true", help="创建HTML索引页面")
    parser.add_argument("--ffmpeg_path", type=str, default=None, help="FFmpeg可执行文件路径")
    
    args = parser.parse_args()
    
    # 如果未指定workers，使用CPU核心数的一半
    if args.parallel and args.workers is None:
        import multiprocessing
        args.workers = max(1, multiprocessing.cpu_count() // 2)
    
    process_all_folders(args)

if __name__ == "__main__":
    main()
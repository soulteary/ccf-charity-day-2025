#!/bin/bash

# 视频目录
VIDEO_DIR="gomoku_videos"

# 输出目录
OUTPUT_DIR="output_videos"
mkdir -p "$OUTPUT_DIR"

# 检查FFmpeg是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo "错误: 未找到FFmpeg。请先安装FFmpeg。"
    exit 1
fi

echo "开始处理视频文件..."

# 获取所有mp4视频文件
videos=()
while IFS= read -r -d $'\0' file; do
    videos+=("$file")
done < <(find "$VIDEO_DIR" -name "*.mp4" -print0)

# 检查视频文件数量
count=${#videos[@]}
if [ $count -eq 0 ]; then
    echo "错误: 在 $VIDEO_DIR 目录中没有找到MP4视频文件。"
    exit 1
fi

echo "找到 $count 个视频文件。"

# 方案1: 2x2网格布局 (4个视频)
create_2x2_grid() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 3)) -ge $count ]; then
        echo "没有足够的视频来创建2x2网格。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]}, ${videos[$start_idx+1]}, ${videos[$start_idx+2]}, ${videos[$start_idx+3]}"
    echo "创建2x2网格, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" \
           -i "${videos[$start_idx+2]}" -i "${videos[$start_idx+3]}" \
        -filter_complex "\
            [0:v]scale=960:540[v0]; \
            [1:v]scale=960:540[v1]; \
            [2:v]scale=960:540[v2]; \
            [3:v]scale=960:540[v3]; \
            [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 方案2: 3x3网格布局 (9个视频)
create_3x3_grid() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 8)) -ge $count ]; then
        echo "没有足够的视频来创建3x3网格。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]} 到 ${videos[$start_idx+8]} (共9个)"
    echo "创建3x3网格, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" -i "${videos[$start_idx+2]}" \
           -i "${videos[$start_idx+3]}" -i "${videos[$start_idx+4]}" -i "${videos[$start_idx+5]}" \
           -i "${videos[$start_idx+6]}" -i "${videos[$start_idx+7]}" -i "${videos[$start_idx+8]}" \
        -filter_complex "\
            [0:v]scale=640:360[v0]; \
            [1:v]scale=640:360[v1]; \
            [2:v]scale=640:360[v2]; \
            [3:v]scale=640:360[v3]; \
            [4:v]scale=640:360[v4]; \
            [5:v]scale=640:360[v5]; \
            [6:v]scale=640:360[v6]; \
            [7:v]scale=640:360[v7]; \
            [8:v]scale=640:360[v8]; \
            [v0][v1][v2][v3][v4][v5][v6][v7][v8]xstack=inputs=9:layout=0_0|640_0|1280_0|0_360|640_360|1280_360|0_720|640_720|1280_720[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 方案3: 1+2布局 (主视频 + 2个小视频)
create_1plus2_layout() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 2)) -ge $count ]; then
        echo "没有足够的视频来创建1+2布局。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]}, ${videos[$start_idx+1]}, ${videos[$start_idx+2]}"
    echo "创建1+2布局, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" -i "${videos[$start_idx+2]}" \
        -filter_complex "\
            [0:v]scale=1080:1080[main]; \
            [1:v]scale=840:540[sub1]; \
            [2:v]scale=840:540[sub2]; \
            [main]pad=1920:1080:0:0:black[bg]; \
            [bg][sub1]overlay=1080:0[temp]; \
            [temp][sub2]overlay=1080:540[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 方案4: 1+4布局 (主视频 + 4个小视频)
create_1plus4_layout() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 4)) -ge $count ]; then
        echo "没有足够的视频来创建1+4布局。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]} 到 ${videos[$start_idx+4]} (共5个)"
    echo "创建1+4布局, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" -i "${videos[$start_idx+2]}" \
           -i "${videos[$start_idx+3]}" -i "${videos[$start_idx+4]}" \
        -filter_complex "\
            [0:v]scale=1280:720,pad=1920:1080:320:180:black[main]; \
            [1:v]scale=640:360[sub1]; \
            [2:v]scale=640:360[sub2]; \
            [3:v]scale=640:360[sub3]; \
            [4:v]scale=640:360[sub4]; \
            [main][sub1]overlay=0:0[temp1]; \
            [temp1][sub2]overlay=1280:0[temp2]; \
            [temp2][sub3]overlay=0:720[temp3]; \
            [temp3][sub4]overlay=1280:720[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 方案5: 横向排列3个视频
create_horizontal_3() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 2)) -ge $count ]; then
        echo "没有足够的视频来创建横向3视频布局。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]}, ${videos[$start_idx+1]}, ${videos[$start_idx+2]}"
    echo "创建横向3视频布局, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" -i "${videos[$start_idx+2]}" \
        -filter_complex "\
            [0:v]scale=640:1080[v0]; \
            [1:v]scale=640:1080[v1]; \
            [2:v]scale=640:1080[v2]; \
            [v0][v1][v2]hstack=inputs=3[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 方案6: 画中画布局 (大背景 + 右下角小视频)
create_picture_in_picture() {
    local start_idx=$1
    local output_file=$2
    
    if [ $((start_idx + 1)) -ge $count ]; then
        echo "没有足够的视频来创建画中画布局。"
        return 1
    fi
    
    echo "使用视频: ${videos[$start_idx]}, ${videos[$start_idx+1]}"
    echo "创建画中画布局, 输出: $output_file"
    
    ffmpeg -i "${videos[$start_idx]}" -i "${videos[$start_idx+1]}" \
        -filter_complex "\
            [0:v]scale=1920:1080,setdar=16/9[bg]; \
            [1:v]scale=480:480[pip]; \
            [bg][pip]overlay=main_w-overlay_w-30:main_h-overlay_h-30[v]" \
        -map "[v]" \
        -c:v libx264 -crf 18 -preset medium \
        "$output_file"
    
    return $?
}

# 随机选择布局并处理视频
process_with_random_layouts() {
    local i=0
    local processed=0
    
    # 定义布局函数数组
    layouts=(
        "create_2x2_grid"
        "create_3x3_grid"
        "create_1plus2_layout"
        "create_1plus4_layout"
        "create_horizontal_3"
        "create_picture_in_picture"
    )
    
    # 定义每个布局需要的视频数量
    video_counts=(4 9 3 5 3 2)
    
    # 布局名称（用于输出文件命名）
    layout_names=(
        "2x2网格"
        "3x3网格"
        "1+2布局"
        "1+4布局"
        "横向3视频"
        "画中画"
    )
    
    while [ $i -lt $count ]; do
        # 随机选择布局
        rand_idx=$((RANDOM % ${#layouts[@]}))
        layout_func=${layouts[$rand_idx]}
        videos_needed=${video_counts[$rand_idx]}
        layout_name=${layout_names[$rand_idx]}
        
        # 检查是否有足够的视频用于当前布局
        if [ $((i + videos_needed)) -gt $count ]; then
            # 尝试找一个需要较少视频的布局
            for j in {0..5}; do
                if [ ${video_counts[$j]} -le $((count - i)) ]; then
                    rand_idx=$j
                    layout_func=${layouts[$rand_idx]}
                    videos_needed=${video_counts[$rand_idx]}
                    layout_name=${layout_names[$rand_idx]}
                    break
                fi
            done
            
            # 如果找不到适合的布局，则结束处理
            if [ $((i + videos_needed)) -gt $count ]; then
                echo "没有足够的视频用于任何布局，处理结束。"
                break
            fi
        fi
        
        # 生成输出文件名
        base_name=$(basename "${videos[$i]}")
        output_file="$OUTPUT_DIR/${layout_name}_${base_name}"
        
        # 调用布局函数
        echo "-------------------------------------"
        echo "随机选择布局: $layout_name (需要 $videos_needed 个视频)"
        $layout_func $i "$output_file"
        result=$?
        
        if [ $result -eq 0 ]; then
            echo "成功创建 $layout_name 布局视频: $output_file"
            processed=$((processed + 1))
        else
            echo "创建 $layout_name 布局失败，跳过。"
        fi
        
        # 移动到下一组视频
        i=$((i + videos_needed))
    done
    
    echo "-------------------------------------"
    echo "处理完成! 共创建 $processed 个组合视频。"
}

# 执行随机布局处理
process_with_random_layouts

echo "全部视频处理完成！输出文件保存在: $OUTPUT_DIR"
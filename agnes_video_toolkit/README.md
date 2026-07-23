# Agnes Video V2.0 生成工具包

这个文件夹只保留视频生成过程中可复用的代码，不包含生成结果、prompt数据、progress文件或真实API Key。

## 文件说明

- `generate_multikey.py`：推荐使用。支持多个API Key并行生成，自动跳过已完成视频，支持断点续跑。
- `generate_videos.py`：单Key批量生成版本，作为备用脚本。
- `requirements.txt`：Python依赖，目前主要是 `requests`。
- `run_example.bat`：Windows命令行运行示例，不含真实API Key。

## 安装依赖

```bat
cd /d c:\Users\86147\Desktop\xixihaha-main\video_gen\agnes_video_toolkit
pip install -r requirements.txt
```

## 多Key批量生成示例

```bat
python generate_multikey.py -p ..\test_30s_prompt --api-keys "YOUR_API_KEY_1,YOUR_API_KEY_2" -o ..\output_videos --height 480 --width 832 --num-frames 81 --frame-rate 15
```

## 约30秒测试参数

已实测成功的接近30秒整数fps配置：

```bat
python generate_multikey.py -p ..\test_30s_prompt --api-keys "YOUR_API_KEY" -o ..\test_30s_video_417f14fps --height 480 --width 832 --num-frames 417 --frame-rate 14
```

理论时长：`417 / 14 ≈ 29.8s`。

注意：Agnes Video V2.0要求 `num_frames <= 441` 且满足 `8n + 1`。常规24fps下最长约18.38秒；如果要接近30秒，需要降低fps。

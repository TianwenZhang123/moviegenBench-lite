@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================================
echo   Agnes Video V2.0 批量生成示例
echo ========================================================
echo   请先把 YOUR_API_KEY 替换为你的真实 API Key
echo   Prompt目录和输出目录可按需修改
echo ========================================================
echo.

python generate_multikey.py -p ..\test_30s_prompt --api-keys "YOUR_API_KEY" -o ..\test_30s_video_417f14fps --height 480 --width 832 --num-frames 417 --frame-rate 14

pause

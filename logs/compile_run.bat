@echo off
cd /d C:\Users\Mcsof\a\rwkv\rwkv-openvino
call C:\Users\Mcsof\Application\openvino_2026.3.0\setupvars.bat
C:\Users\Mcsof\miniconda3\envs\AI\python.exe -u logs\compile_test.py out\rwkv7-13.3b-q4k_ov.xml CPU

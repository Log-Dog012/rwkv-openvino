@echo off
cd /d C:\Users\Mcsof\a\rwkv\rwkv-openvino
call C:\Users\Mcsof\Application\openvino_2026.3.0\setupvars.bat >nul 2>&1
set PYTHONPATH=C:\Users\Mcsof\Application\openvino_2026.3.0\python
C:\Users\Mcsof\miniconda3\envs\AI\python.exe -u scripts\rwkv7_ov_layerwise.py C:\Users\Mcsof\models\gguf\rwkv7-g1i-13.3b-Q4_K_M.gguf --n 8 --threads 8 --chunk 8 --ir-dir out\chunks_13.3b --prompt "The Eiffel Tower is located in the city of" > logs\e2e_run.log 2>&1
echo DONE %errorlevel% > logs\e2e_done.flag

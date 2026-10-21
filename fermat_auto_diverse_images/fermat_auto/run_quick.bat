@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python test_dataset.py
if errorlevel 1 goto failed
python test_metrics.py
if errorlevel 1 goto failed
python run.py --config quick.json --workers 2
if errorlevel 1 goto failed
pause
exit /b 0
:failed
 echo Failed. See error above.
pause
exit /b 1

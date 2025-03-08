@echo off
REM Simple script to set up a UV virtual environment and install requirements

echo Creating virtual environment with UV...
uv venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
uv pip install -r requirements.txt

echo Setup complete! Virtual environment is activated and dependencies are installed.
echo To manually activate this environment in the future, run: .venv\Scripts\activate.bat 
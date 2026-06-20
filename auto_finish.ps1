$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING="utf-8"

Write-Host "Starting automated training..."
.\venv\Scripts\python.exe train.py --epochs 80

Write-Host "Training complete. Starting evaluation..."
.\venv\Scripts\python.exe evaluate.py

Write-Host "Evaluation complete. Pushing to GitHub..."
git add .
git add -f results checkpoints
git commit -m "Automated final training and evaluation run"
git push

Write-Host "Project finished automatically!"

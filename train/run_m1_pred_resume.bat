@echo off
cd /d "F:/e/科研/论文发表/第六篇论文/train"
"F:/pyenvs/road/Scripts/python.exe" -u predict.py --ckpt ../runs/m1_ori_e1b/best.pt --manifest ../manifests/whu_rroad_test_remaining_m1.csv --out ../runs/m1_ori_e1b/pred >> ../runs/pred_m1_schtask.log 2>&1

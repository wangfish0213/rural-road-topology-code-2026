# -*- coding: utf-8 -*-
"""生成 run_m1_pred_resume.bat（GBK 编码，供 schtasks 启动 M1 断点推理）。"""
content = (
    "@echo off\r\n"
    'cd /d "F:/e/科研/论文发表/第六篇论文/train"\r\n'
    '"F:/pyenvs/road/Scripts/python.exe" -u predict.py '
    "--ckpt ../runs/m1_ori_e1b/best.pt "
    "--manifest ../manifests/whu_rroad_test_remaining_m1.csv "
    "--out ../runs/m1_ori_e1b/pred "
    ">> ../runs/pred_m1_schtask.log 2>&1\r\n"
)
p = r"F:\e\科研\论文发表\第六篇论文\train\run_m1_pred_resume.bat"
with open(p, "w", encoding="gbk", newline="") as f:
    f.write(content)
print("written:", p)
print(open(p, encoding="gbk").read())

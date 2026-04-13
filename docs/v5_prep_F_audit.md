# F Audit: probe.py vs cka.py layer coverage

**Case 2 — 不同分辨率同一模型**

- `negative_control_probe.py` 覆盖 7 个 transformer layer（对齐 probe 侧固定采样）。
- `negative_control_cka.py` 使用 `--layers 0 7 14 21 28` 在 28 层模型上取 5 个采样点，跨度覆盖全 28 层。
- 结论：两者采样粒度不同但作用于同一底层 transformer；属于 Case 2（同一模型不同分辨率），非语义冲突，不阻塞 v5 launch。
- 依据：`scripts/negative_control_probe.py` 与 `scripts/negative_control_cka.py` 的 LAYERS 定义；前 Worker `worker_v5_prep_script_hardening_and_F_audit_20260413T141748` 的 F 核实记录。

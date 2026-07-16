# Phase462 DP Step 3b 后处理说明

采集后首次判卷暴露分析器实现错误：它按 `arrival_ordinal` 检查 admit step 单调性，并拒绝同一步 admit 的零间隔。并发请求的 arrival 顺序不等于 admit 顺序，且同一步 admit 合法。

修复仅将每个 rank 的 `schedule_seq` 排序后计算 gap，并允许 gap=0；冻结的 route→drain 判卷顺序、route 不对称定义和 `H-route/H-drain/H-mixed` 分支均未修改。新增三点链中位延迟与 score 并列次数只用于报告。

| 项目 | SHA256 |
|---|---|
| 采集时分析器 | `2c38f98cc726ccaff82cf3543d2bd5dd1ffdb1d137f4d3f451ce63f75ff1b4a2` |
| 最终后处理分析器 | `7fc4d48fdbef7f3c9fe24a9b6d1ee85cca504ce4b4d8efd33a68fedd2d90e017` |

同一份完整原始事件流重放后机械判卷为 `H-mixed`。`diagnostic_only=true`，`valid_for_default=false`。

最终本地 patcher 另修正了 anchor 缺失错误消息中的字面量格式，仅改善失败诊断；采集注入内容与恢复逻辑未变。采集时工具哈希保留在 `tools_collection.sha256`，最终代码哈希保留在 `tools_postprocess.sha256`。

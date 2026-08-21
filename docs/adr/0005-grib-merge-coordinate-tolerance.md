# 气象变量合并采用坐标属性覆盖容差（compat='override'）

在解析与合并 NOAA GEFS 重预报的独立变量切片（如 `tmax_2m` 与 `tmin_2m`）时，`xr.merge` 与 `xr.concat` 统一显式指定 `compat='override'` 与 `coords='minimal'`，允许非关键坐标元数据微小差异。

原因：NOAA 历史重预报跨 20 年归档中，不同批次（如冬令时转换时段或跨年边界）在生成 TMAX 和 TMIN 独立 GRIB 文件时，其辅助坐标 `valid_time` 的底层 GRIB 消息头描述属性可能存在微小字段差异。xarray 默认的严格比对（`compat='no_conflicts'`）会误判冲突并抛出 `MergeError` 阻断批处理。由于两文件的物理时间戳与网格点完全一致，采用 `compat='override'` 可以确保数据矩阵无损合并的同时，赋予数据管道对 20 年全量历史数据的鲁棒性。

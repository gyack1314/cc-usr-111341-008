# 业务资料约定

parents可为多个父节点，taxonomy_version绑定标签当时可选目录。未列人工标签表示未知，不自动代表负例。family是训练隔离和统计计量键，公开日用于当时可用性；text保留原语种，不自动把翻译视为独立发明。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- taxonomy：id、parents、version。
- documents：id、family、language、published、priority、text。
- labels：id、document、node、taxonomy_version、confirmed_at。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。

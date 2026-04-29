# SHMEM 使用限制
1. GM2GM的highlevel RMA操作使用默认buffer，不支持并发操作，否则可能造成数据覆盖。若有并发需求，建议使用lowlevel接口。
2. barrier接口当前必须在Mix Kernel（包含mmad和GM2UB/UB2GM操作）中使用，可参考example样例。该限制待编译器更新后移除。

# SHMEM 常见问题
Q：  
A：  
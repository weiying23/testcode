##### SHMEM对外接口
1. 初始化共享内存模块
    ```python
    def shmem_init(mype, npes, mem_size) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |mype|本地处理单元索引，范围在 0 ~ npes 之间|
    |npes|处理单元总数|
    |mem_size|每个处理单元的内存大小，以字节为单位|
    |返回值|成功返回0，其他为错误码|

1. 完成共享内存模块
    ```python
    def shmem_finialize() -> None
    ```

1. 设置属性
    ```python
    def shmem_set_attributes(my_rank, n_ranks, local_mem_size, ip_port, attributes) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |my_rank|int类型，当前秩|
    |n_ranks|int类型，秩的总数|
    |local_mem_size|int类型，当前占用的共享内存大小，最大40G|
    |ip_port|str类型，服务器的IP和端口号，例如tcp://ip:port|
    |attributes|InitAttr类型，设置的属性|
    |返回值|成功返回0，其他为错误码|

1. 查询共享内存模块的当前初始化状态
    ```python
    def shmem_init_status() -> InitStatus
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回初始化状态。返回 SHMEM_STATUS_IS_INITIALIZED 表示初始化已完成|

1. 修改将用于初始化的属性中的数据操作引擎类型
    ```python
    def shmem_set_data_op_engine_type(attributes, vaue) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |attributes|InitAttr类型，属性集|
    |vaue|int类型，数据操作引擎类型的值|
    |返回值|成功时返回0,失败时返回错误代码|

1. 修改用于初始化的属性中的超时设置
    ```python
    def shmem_set_timeout(attributes, vaue) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |attributes|InitAttr类型，属性集|
    |vaue|int类型，数据操作引擎类型的值|
    |返回值|成功时返回0,失败时返回错误代码|

1. 注册一个Python解密处理程序
    ```python
    def set_conf_store_tls_key(tls_pk, tls_pk_pw, py_decrypt_func:Callable[[str], str]) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |tls_pk|私钥|
    |tls_pk_pw|私钥口令|
    |py_decrypt_func|私钥口令解密函数，一个Python函数，接受 (str cipher_text)，并返回 (str plain_text)|
    |返回值|成功时返回0,失败时返回错误代码|

1. 分配内存
    ```python
    def shmem_malloc(size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |size|分配内存大小|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 为多个元素分配内存
    ```python
    def shmem_calloc(nmemb, size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |nmemb|元素数量|
    |size|每个元素的大小|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 分配指定对齐方式内存
    ```python
    def shmem_align(alignment, size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |alignment|所需的内存对齐方式（必须是 2 的幂）|
    |size|要分配的字节数|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 释放被分配的内存空间
    ```python
    def shmem_free(ptr) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |ptr|要释放的内存|

1. 获取可用于在指定PE上直接引用目标地址的地址
    ```python
    def shmem_ptr(ptr, peer) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |ptr|远程可访问数据的对称地址|
    |peer|PE编号|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 获取PE值
    ```python
    def my_pe() -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|PE值|

1. 获取在特定团队中的PE号码
    ```python
    def team_my_pe(team_id) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |team_id|team的句柄|
    |返回值|PE值|

1. 程序中运行的PE数量
    ```python
    def pe_count() -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|PE数量|

1. 获取特定团队中的PE数量
    ```python
    def team_n_pes(team_id) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |team_id|team的句柄|
    |返回值|PE数量|

1. 设置由NPU发起的MTE操作所使用的UB参数
    ```python
    def mte_set_ub_params(offset, size, event) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |offset|UB的起始偏移量|
    |size|UB的大小|
    |event|用于同步的事件ID|
    |返回值|成功返回0|

1. 从现有的父团队中拆分出一个子团队
    ```python
    def team_split_strided(parent, start, stride, size)
    ```

    |参数/返回值|含义|
    |-|-|
    |parent|父团队ID|
    |start|新团队中PE子集的最低PE编号|
    |stride|父团队中团队PE编号之间的步长|
    |size|来自父团队的PE数量|
    |返回值|成功返回新团队ID|

1. 集体接口
    ```python
    def team_split_2d(parent, x_range)
    ```

    |参数/返回值|含义|
    |-|-|
    |parent|team句柄|
    |x_range|第一维度中的元素数量|
    |返回值|成功返回0|

1. 获取作为团队创建时传入的team配置
    ```python
    def shmem_team_get_config(team) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |parent|team id|
    |返回值|成功返回0|

1. 同步接口,将对称内存中连续的数据从本地处理单元（PE）复制到指定PE的地址上
    ```python
    def shmem_putmem(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|本地PE对称地址上的指针|
    |src|源数据本地内存中的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 同步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上。
    ```python
    def shmem_getmem(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 返回主版本号和次版本号
    ```python
    def shmem_info_get_version()
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回主版本号和次版本号|

1. 返回供应商定义的名称字符串
    ```python
    def shmem_info_get_name()
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回供应商定义的名称字符串|

1. 将一个团队中的给定PE编号转换为另一个团队中的对应PE编号参数
    ```python
    def team_translate_pe(src_team, src_pe, dest_team)
    ```

    |参数/返回值|含义|
    |-|-|
    |src_team|源团队ID|
    |src_pe|源PE编号|
    |dest_team|目标团队ID|
    |返回值|成功时，返回目标团队中指定PE的编号。出错时，返回-1|

1. 销毁一个team
    ```python
    def team_destroy(team)
    ```

    |参数/返回值|含义|
    |-|-|
    |team|team ID|

1. 获取运行时FFT配置
    ```python
    def get_ffts_config()
    ```

1. 异步接口,将本地处理单元（PE）上的连续数据复制到指定处理单元（PE）上的对称地址
    ```python
    def shmem_putmem_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|本地PE对称地址上的指针|
    |src|源数据本地内存中的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 异步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上
    ```python
    def shmem_getmem_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 异步接口,从指定的PE复制对称内存中的连续数据到本地PE的地址
    ```python
    def shmem_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 同步接口,从指定的PE复制对称内存中的连续数据到本地PE的地址
    ```python
    def shmem_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 所有进程通过广播调用exit()函数退出
    ```python
    def shmem_global_exit(status)
    ```

    |参数/返回值|含义|
    |-|-|
    |status|传递给exit()函数的状态值|

1. 获取团队中的PE编号，即PE的索引
    ```python
    def my_pe(team)
    ```

    |参数/返回值|含义|
    |-|-|
    |team|team id|
    |返回值|返回指定团队中PE的编号，出错时，返回-1|

1.
    ```python
    def pe_count(team)
    ```

    |参数/返回值|含义|
    |-|-|
    |team|team id|
    |返回值|返回指定团队中PE的数目，出错时，返回-1|

##### SHMEM类
1. OpEngineType枚举类
    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
    ```

1. OptionalAttr类
    ```python
    class OptionalAttr:
        def __init__(self):
    ```

    |属性|含义|
    |-|-|
    |version|版本|
    |data_op_engine_type|类型|
    |shm_init_timeout|init函数的超时时间|
    |shm_create_timeout|create函数的超时时间|
    |control_operation_timeout|控制操作的超时时间|

1. InitAttr类
    ```python
    class InitAttr:
        def __init__(self):
    ```

    |属性|含义|
    |-|-|
    |my_rank|当前进程的排名|
    |n_ranks|所有进程的总排名数|
    |ip_port|通信服务器的ip和端口|
    |local_mem_size|当前占用的共享内存大小|
    |option_attr|可选参数|

1. TeamConfig类
    ```python
    class TeamConfig:
    ```

    |属性|含义|
    |-|-|
    |num_contexts|一个团队（team）中可以同时运行的上下文数量|

1. InitStatus枚举类
    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```
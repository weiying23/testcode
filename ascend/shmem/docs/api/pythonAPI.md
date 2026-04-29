## SHMEM Python API Reference
### shmem.core API
##### 对外接口
1. 获取当前库版本函数
    ```python
    def get_version() -> str
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|version信息组成的字符串|

1. 初始化共享内存模块
    ```python
    def get_unique_id(empty: bool=False) -> UniqueID
    ```

    |参数/返回值|含义|
    |-|-|
    |empty|预留参数，无实际意义|
    |返回值|代表一个唯一 ID 的句柄|

1. 初始化共享内存模块
    ```python
    def init(device: int=None, uid: UniqueID=None, rank: int=None, nranks: int=None, mpi_comm=None, initializer_method: str="", mem_size: int=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |device|预留参数，无实际意义|
    |uid|用于初始化的唯一标识符|
    |rank|当前进程在 ACLSHMEM 作业中的排名|
    |nranks|参与 ACLSHMEM 作业的总进程数|
    |mpi_comm|预留参数，无实际意义|
    |initializer_method|指定初始化方法，必须为 "uid"|
    |mem_size|每个处理单元（PE）分配的对称内存大小|

1. 销毁共享内存模块
    ```python
    def finalize() -> None
    ```
   
1. 分配一个由 ACLSHMEM 支持的 NPU 缓冲区
    ```python
    def buffer(size, release=False, except_on_del=True) -> Buffer
    ```

    |参数/返回值|含义|
    |-|-|
    |size|要分配的缓冲区大小|
    |release|预留参数，无实际意义|
    |except_on_del|预留参数，无实际意义|
    |返回值|一个原始内存缓冲区，通过其地址和字节长度表示|

1. 释放缓冲区
    ```python
    def free(buf: Buffer) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |buf|需要释放的缓冲区|

1. 获取可用于在指定PE上直接引用目标地址的地址
    ```python
    def get_peer_buffer(buf: Buffer, pe: int) -> Buffer
    ```

    |参数/返回值|含义|
    |-|-|
    |buf|远程可访问数据的对称地址|
    |pe|PE编号|
    |返回值|一个原始内存缓冲区，通过其地址和字节长度表示|

1. 同步接口，从本地PE复制内存中的连续数据到指定的PE地址
    ```python
    def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|远程 PE 上的对称内存缓冲区（目标地址）|
    |src|包含源数据的内存缓冲区。|
    |signal_var|用作信号变量的对称内存缓冲区|
    |signal_operation|用于更新信号变量的原子操作|
    |remote_pe|远程处理单元（PE）的编号|
    |stream|预留参数，无实际意义|

1. 同步接口，在指定流上将对称内存中连续的数据从本地处理单元（PE）复制到指定PE的地址上
    ```python
    def put(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|远程 PE 上的对称目标缓冲区|
    |src|包含待发送数据的本地源缓冲区|
    |remote_pe|目标处理单元（PE）的编号|
    |stream|ACL 流对象|

1. 同步接口，在指定流上将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上
    ```python
    def get(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|用于写入数据的本地目标缓冲区|
    |src|远程 PE 上的对称源缓冲区|
    |remote_pe|目标处理单元（PE）的编号|
    |stream|ACL 流对象|

1. 同步接口，在指定的 PE 上对远程信号变量执行原子操作，该操作在给定的流上执行
    ```python
    def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream: int=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |signal_var |包含信号变量的对称内存缓冲区|
    |signal_val |用于远程原子信号操作的值|
    |signal_operation |要应用于远程信号变量的原子操作|
    |remote_pe|远程处理单元（PE）的编号|
    |stream|ACL 流对象|

1. 同步接口，在指定的 PE 上对远程信号变量执行原子比较操作，该操作在给定的流上执行
    ```python
    def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int=None) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |signal_var|包含信号变量的对称内存缓冲区|
    |signal_val|用于与 signal_var 所指向的值进行比较的数值|
    |signal_operation|用于比较 signal_var 所指向值与 signal_val 的比较操作符|
    |stream|ACL 流对象|

1. 获取PE值
    ```python
    def my_pe() -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|PE值|

1. 获取在特定team中的PE号码
    ```python
    def team_my_pe(team) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |team|目标team的id|
    |返回值|PE值|

1. 程序中运行的PE数量
    ```python
    def n_pes() -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|PE数量|

1. 获取特定team中的PE数量
    ```python
    def team_n_pes(team) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |team|目标team的id|
    |返回值|PE数量|

1. 查询共享内存模块的当前初始化状态
    ```python
    def init_status() -> InitStatus
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回初始化状态|

##### 类
1. UniqueId类
    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|含义|
    |-|-|
    |version| 版本信息|
    |my_rank| 当前进程的pe编号|
    |n_pes| 所有进程的pe总数|
    |internal| uid的内部信息|

1. InitStatus枚举类
    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```
   
1. SignalOp枚举类
    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```
   
1. ComparisonType枚举类
    ```python
    class ComparisonType(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

### shmem._pyshmem API
##### 对外接口
1. 初始化共享内存模块
    ```python
    def aclshmem_init(mype, npes, mem_size) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |mype|本地处理单元索引，范围在 0 ~ npes 之间|
    |npes|处理单元总数|
    |mem_size|每个处理单元的内存大小，以字节为单位|
    |返回值|成功返回0，其他为错误码|

1. 完成共享内存模块
    ```python
    def aclshmem_finalize() -> None
    ```

1. 查询共享内存模块的当前初始化状态
    ```python
    def aclshmemx_init_status() -> InitStatus
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回初始化状态。返回 ACLSHMEM_STATUS_IS_INITIALIZED 表示初始化已完成|

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
    def aclshmem_malloc(size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |size|分配内存大小|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 为多个元素分配内存
    ```python
    def aclshmem_calloc(nmemb, size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |nmemb|元素数量|
    |size|每个元素的大小|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 分配指定对齐方式内存
    ```python
    def aclshmem_align(alignment, size) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |alignment|所需的内存对齐方式（必须是 2 的幂）|
    |size|要分配的字节数|
    |返回值|分配成功返回指向已分配内存的指针；该内存未被成功分配或size为0，则返回NULL|

1. 释放被分配的内存空间
    ```python
    def aclshmem_free(ptr) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |ptr|要释放的内存|

1. 获取可用于在指定PE上直接引用目标地址的地址
    ```python
    def aclshmem_ptr(ptr, peer) -> None
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
    def aclshmem_team_get_config(team) -> int
    ```

    |参数/返回值|含义|
    |-|-|
    |team|team id|
    |返回值|成功返回0|

1. 同步接口,将内存中连续的数据从本地处理单元（PE）复制到指定PE的地址上
    ```python
    def aclshmem_putmem(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|远程PE对称地址上的指针|
    |src|源数据本地内存中的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 同步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上。
    ```python
    def aclshmem_getmem(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地内存的指针|
    |src|指向远程处理单元（PE）对称地址的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 同步接口,将本地内存中的数据按照sst的步距从本地处理单元（PE）复制到指定PE的地址上（按照dst的步距）
    ```python
    def aclshmem_{TYPE}_iput(dest, source, dst, sst, nelems, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |TYPE|数据类型，可选float/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
   |dest|指向远程目标（PE）数据的对称内存的指针|
   |source|源数据所在本地设备的指针|
   |dst|目标地址中连续元素之间的步长|
   |sst|源地址中连续元素之间的步长|
   |nelems|连续元素块的个数|
   |pe|远程PE的编号|

1. 同步接口,将对称内存中指定处理单元（PE）上的数据按照sst的步距复制到本地PE的地址上（按照dst的步距）
    ```python
    def aclshmem_{TYPE}_iget(dest, source, dst, sst, nelems, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |TYPE|数据类型，可选float/double/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
   |dest|目的数据所在本地设备的指针|
   |source|指向远程源数据的对称内存的指针|
   |dst|目标地址中连续元素之间的步长|
   |sst|源地址中连续元素之间的步长|
   |nelems|连续元素块的个数|
   |pe|远程PE的编号|

1. 同步接口,将本地内存中连续的数据从本地处理单元（PE）复制到指定PE的对称内存地址上
    ```python
    def aclshmem_put{BITS}(dst, src, elem_size, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dst|远程PE对称地址上的指针|
   |src|源数据本地内存中的指针|
   |elem_size|目标地址和源地址中元素的大小|
   |pe|远程PE的编号|

1. 同步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上。
    ```python
    def aclshmem_get{BITS}(dst, src, elem_size, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dst|指向本地处理单元（PE）内存的指针|
   |src|指向远程处理单元（PE）对称地址的指针|
   |elem_size|目标地址和源地址中元素的大小|
   |pe|远程PE的编号|

1. 同步接口,将对称内存中的数据按照sst的步距从本地处理单元（PE）复制到指定PE对称内存的地址上（按照dst的步距）
    ```python
    def aclshmem_iput{BITS}(dest, source, dst, sst, nelems, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dest|指向远程目标数据的对称内存的指针|
   |source|源数据所在本地设备的指针|
   |dst|目标地址中连续元素之间的步长|
   |sst|源地址中连续元素之间的步长|
   |nelems|连续元素块的个数|
   |pe|远程PE的编号|

1. 同步接口,将对称内存中指定处理单元（PE）上的数据按照sst的步距复制到本地PE的地址上（按照dst的步距）
    ```python
    def aclshmem_iget{BITS}(dest, source, dst, sst, nelems, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dest|目的数据所在本地设备的指针|
   |source|指向远程源数据的对称内存的指针|
   |dst|目标地址中连续元素之间的步长|
   |sst|源地址中连续元素之间的步长|
   |nelems|连续元素块的个数|
   |pe|远程PE的编号|

1. 返回主版本号和次版本号
    ```python
    def aclshmem_info_get_version()
    ```

    |参数/返回值|含义|
    |-|-|
    |返回值|返回主版本号和次版本号|

1. 返回供应商定义的名称字符串
    ```python
    def aclshmem_info_get_name()
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
    def aclshmem_putmem_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|远程PE对称地址上的指针|
    |src|源数据本地内存中的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 异步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上
    ```python
    def aclshmem_getmem_nbi(dst, src, elem_size, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）内存的指针|
    |src|指向远程处理单元（PE）对称地址的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|

1. 异步接口,将本地处理单元（PE）上的连续数据复制到指定处理单元（PE）上的对称地址
    ```python
    def aclshmem_put{BITS}_nbi(dst, src, elem_size, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dst|远程PE对称地址上的指针|
   |src|源数据本地内存中的指针|
   |elem_size|目标地址和源地址中元素的大小|
   |pe|远程PE的编号|

1. 异步接口,将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上
    ```python
    def aclshmem_get{BITS}_nbi(dst, src, elem_size, pe)
    ```

   |参数/返回值|含义|
   |-|-|
   |BITS|数据位宽，可选8/16/32/64/128|
   |dst|指向本地处理单元（PE）内存的指针|
   |src|指向远程处理单元（PE）对称地址的指针|
   |elem_size|目标地址和源地址中元素的大小|
   |pe|远程PE的编号|

1. 异步接口,从本地PE复制对称内存中的连续数据到指定的PE地址
    ```python
    def aclshmemx_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向远程处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 同步接口,从本地PE复制对称内存中的连续数据到指定的PE地址
    ```python
    def aclshmemx_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向远程处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 异步接口,从本地PE复制内存中的连续数据到指定的PE对称内存的地址
    ```python
    def aclshmemx_put{BITS}_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |BITS|数据位宽，可选8/16/32/64/128|
    |dst|指向远程处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 同步接口,从本地PE复制对称内存中的连续数据到指定的PE对称内存的地址
    ```python
    def aclshmemx_put{BITS}_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |参数/返回值|含义|
    |-|-|
    |BITS|数据位宽，可选8/16/32/64/128|
    |dst|指向远程处理单元（PE）对称地址的指针|
    |src|指向源数据本地内存的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |sig|要更新的信号字的对称地址|
    |signal|用于更新sig_addr的值|
    |sig_op|用于signal更新sig_addr的操作|
    |pe|远程PE的编号|

1. 所有进程通过广播调用exit()函数退出
    ```python
    def aclshmem_global_exit(status)
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

1. 获取指定团队中的PE数量
    ```python
    def pe_count(team)
    ```

    |参数/返回值|含义|
    |-|-|
    |team|team id|
    |返回值|返回指定团队中PE的数目，出错时，返回-1|

1. 阻塞式同步原语，用于等待指定 PE 上的信号变量（sig_addr）满足指定比较条件（如等于、大于等）时返回
    ```python
    def aclshmem_signal_wait_until(sig_addr, cmp, cmp_val)
    ```

    |参数/返回值|含义|
    |-|-|
    |sig_addr|指向对称内存中一个信号变量的指针|
    |cmp|比较类型|
    |cmp_val|比较值|
    |返回值|返回满足比较条件时sig_addr的值|

1. 阻塞式同步原语，用于等待指定 PE 上的信号变量（ivar）满足指定比较条件（如等于、大于等）时返回
    ```python
    def aclshmem_{TYPE}_wait_until(ivar, cmp, cmp_val)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivar|指向对称内存中一个信号变量的指针|
    |cmp|比较类型|
    |cmp_val|比较值|

1. 阻塞式同步原语，用于等待指定 PE 上的信号变量（ivar）不等于给定比较值cmp_value时返回
    ```python
    def aclshmem_{TYPE}_wait(ivar, cmp_val)
    ```

    |参数/返回值|含义|
    |-|-|
    |ivar|指向对称内存中一个信号变量的指针|
    |cmp_val|比较值|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）均满足给定的比较条件（ivars[i] cmp cmp_value）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_all(ivars_ptr, nelems, status_ptr, cmp, cmp_val)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_val|比较值|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）中至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_value）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_any(ivars_ptr, nelems, status_ptr, cmp, cmp_val, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_val|比较值|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素索引值|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）中至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_value）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_val, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |indices_ptr|对应 ivars_ptr 中所有满足条件元素的索引值|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_val|比较值|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素个数|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）均满足给定的比较条件（ivars[i] cmp cmp_values[i]）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）中至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_values[i]）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_val, cmp_values_ptr, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素索引值|

1. 阻塞式同步原语，用于等待指定 PE 上一组变量（ivars）中至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_values[i]）时返回
    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |indices_ptr|对应 ivars_ptr 中所有满足条件元素的索引值|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素个数|

1. 同步原语，用于检查指定 PE 上变量（ivar）是否满足给定的比较条件（ivar cmp cmp_value）
    ```python
    def aclshmem_{TYPE}_test(ivar, cmp, cmp_value, res_out_ptr)
    ```

    |参数/返回值| 含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivar|指向对称内存中一个信号变量的指针|
    |cmp|比较类型|
    |cmp_value|比较值|
    |res_out_ptr|接收device侧接口返回值。满足比较条件返回1，否则返回0|

1. 同步原语，用于检查指定 PE 上一组变量（ivars）中是否至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_value）
    ```python
    def aclshmem_{TYPE}_test_any(ivars_ptr, nelems, status_ptr, cmp, cmp_value, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_value|比较值|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素索引值|

1. 同步原语，用于检查指定 PE 上一组变量（ivars）中是否至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_value）
    ```python
    def aclshmem_{TYPE}_test_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_value, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |indices_ptr|对应 ivars_ptr 中所有满足条件元素的索引值|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_value|比较值|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素个数|

1. 同步原语，用于检查指定 PE 上一组变量（ivars）是否均满足给定的比较条件（ivars[i] cmp cmp_values[i]）
    ```python
    def aclshmem_{TYPE}_test_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|
    |res_out_ptr|接收device侧接口返回值。满足比较条件返回1，否则返回0|

1. 同步原语，用于检查指定 PE 上一组变量（ivars）中是否至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_values[i]）
    ```python
    def aclshmem_{TYPE}_test_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_val, cmp_values_ptr, res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素索引值|

1. 同步原语，用于检查指定 PE 上一组变量（ivars）中是否至少有一个变量满足给定的比较条件（ivars[i] cmp cmp_values[i]）
    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr,res_out_ptr)
    ```

    |参数/返回值|含义|
    |-|-|
    |TYPE|数据类型，可选float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char|
    |ivars_ptr|指向对称内存中长度为 nelems 的数组|
    |nelems|数组元素个数|
    |indices_ptr|对应 ivars_ptr 中所有满足条件元素的索引值|
    |status_ptr|可选本地掩码数组|
    |cmp|比较类型|
    |cmp_values_ptr|比较值数组|
    |res_out_ptr|接收device侧接口返回值。即满足比较条件的元素个数|

1. 同步接口，在指定流上将对称内存中连续的数据从本地处理单元（PE）复制到指定PE的地址上
    ```python
    def aclshmemx_putmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|远程PE对称地址上的指针|
    |src|源数据本地内存中的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|
    |stream|使用指定的流进行拷贝（如果 stream 为 NULL，则使用默认流）|

1. 同步接口，在指定流上将对称内存中指定处理单元（PE）上的连续数据复制到本地PE的地址上
    ```python
    def aclshmemx_getmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |dst|指向本地处理单元（PE）内存地址的指针|
    |src|指向远程处理单元（PE）对称地址的指针|
    |elem_size|目标地址和源地址中元素的大小|
    |pe|远程PE的编号|
    |stream|使用指定的流进行拷贝（如果 stream 为 NULL，则使用默认流）|

1. 同步接口，在指定的 PE 上对远程信号变量执行原子操作，该操作在给定的流上执行
    ```python
    def aclshmemx_signal_op_on_stream(sig, signal, sig_op, pe, stream) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |sig|本地地址，指向源信号变量，该变量在目标 PE 上可访问|
    |signal|用于原子操作的值|
    |sig_op|在远程信号变量上执行的操作|
    |pe|远程PE的编号|
    |stream|使用指定的流进行拷贝（如果 stream 为 NULL，则使用默认流）|

1. 同步接口，在指定的 PE 上对远程信号变量执行原子比较操作，该操作在给定的流上执行
    ```python
    def aclshmemx_signal_wait_until_on_stream(sig, cmp, cmp_val, stream) -> None
    ```

    |参数/返回值|含义|
    |-|-|
    |sig|源信号变量的本地地址|
    |cmp|用于比较 sig 所指向值与 cmp_val 的比较操作符|
    |cmp_val|用于与 sig 所指向的值进行比较的数值|
    |stream|使用指定的流进行拷贝（如果 stream 为 NULL，则使用默认流）|

##### 类
1. OpEngineType枚举类
    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
        UDMA
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

1. UniqueId类
    ```python
    class UniqueId:
        def __init__(self):
    ```

    |属性|含义|
    |-|-|
    |version| 版本信息|
    |my_rank| 当前进程的pe编号|
    |n_pes| 所有进程的pe总数|
    |internal| uid的内部信息|

1. InitStatus枚举类
    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```
   
1. SignalOp枚举类
    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```
   
1. CmpOp枚举类
    ```python
    class CmpOp(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```
import time
from multiprocessing import Pool, Manager, Process

from tqdm import tqdm


def show_progress(total_len, share):
    """
    子进程显示进度条
    :param total_len: 进度条总长度
    :param share: 目前进度条进度 manager.Value(typecode=int, )
    :return:
    """
    count = 0
    with tqdm(total=total_len) as bar:
        while True:
            value = share.value
            time.sleep(0.2)
            if value - count > 0:
                bar.update(value - count)
                count += value - count

            if count == total_len:
                return


def reduce(queue, reduce_func, reduce_param):
    params = []
    while True:
        rst = queue.get()
        # 结束任务
        if isinstance(rst, str) and rst == 'EOF':
            break
        params.append(rst)
    if reduce_param is not None:
        reduce_func(params, *reduce_param)
    else:
        reduce_func(params)


class MultiTaskExecutor(object):
    def __init__(self, target, pool_size=16, use_queue=False, reduce_func=None, reduce_param=None, queue_max_size=-1):
        self.target = target
        self.pool_size = pool_size
        self.use_queue = use_queue
        self.reduce_func = reduce_func
        self.reduce_param = reduce_param
        self.queue = None
        self.queue_max_size = queue_max_size

    def execute(self, tasks):
        assert isinstance(tasks, list) or isinstance(tasks, int), 'tasks ValueError'
        assert isinstance(tasks, list) and len(tasks) > 0 or tasks > 0, '任务数量小于0'

        if self.reduce_func is not None:
            assert self.use_queue, '使用了reduce函数必须启用queue'

        manager = Manager()
        lock = manager.Lock()
        share_value = manager.Value(typecode=int, value=0)

        if self.use_queue:
            self.queue = manager.Queue(self.queue_max_size) if self.queue_max_size != -1 else manager.Queue()

        # 进度条子进程
        task_length = len(tasks) if isinstance(tasks, list) else tasks
        count_process = Process(target=show_progress, args=(task_length, share_value))
        count_process.start()

        # reduce 子函数
        reduce_process = None
        if self.reduce_func is not None:
            reduce_process = Process(target=reduce, args=(self.queue, self.reduce_func, self.reduce_param))
            reduce_process.start()

        # 任务池分配任务
        pool = Pool(self.pool_size)
        if isinstance(tasks, list):
            for task in tasks:
                pool.apply_async(self._target_func, args=(share_value, lock, *task))
        else:
            for _ in range(tasks):
                pool.apply_async(self._target_func, args=(share_value, lock))
        pool.close()
        pool.join()
        # reduce 子函数
        if reduce_process is not None:
            time.sleep(1)
            self.queue.put('EOF')
            reduce_process.join()

        time.sleep(1)
        count_process.terminate()

    def _target_func(self, share_value, lock, *args):
        rst = self.target(*args)
        if self.use_queue:
            self.queue.put(rst)
        with lock:
            share_value.value += 1

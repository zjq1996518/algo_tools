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
        rsts = queue.get()
        # 结束任务
        if isinstance(rsts, str) and rsts == 'EOF':
            break
        params += rsts
    if reduce_param is not None:
        reduce_func(params, *reduce_param)
    else:
        reduce_func(params)


class MultiTaskExecutor(object):
    def __init__(self, target, pool_size=16, reduce_func=None, reduce_param=None, queue_max_size=-1):
        self.target = target
        self.pool_size = pool_size
        self.reduce_func = reduce_func
        self.reduce_param = reduce_param
        self.queue = None
        self.queue_max_size = queue_max_size
        self.use_queue = self.reduce_func is not None

    def execute(self, tasks):
        assert isinstance(tasks, list), 'tasks ValueError'
        assert isinstance(tasks, list) and len(tasks) > 0, '任务数量小于0'

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
        batch_size = len(tasks) // self.pool_size

        if isinstance(tasks, list):
            for i in range(self.pool_size):
                batch_tasks = tasks[i*batch_size:(i+1)*batch_size] if i != self.pool_size - 1 else tasks[i*batch_size:]
                pool.apply_async(self._target_func, args=(batch_tasks, share_value, lock))
        pool.close()
        pool.join()
        # reduce 子函数
        if reduce_process is not None:
            time.sleep(1)
            self.queue.put('EOF')
            reduce_process.join()

        time.sleep(1)
        count_process.terminate()

    def _target_func(self, tasks, share_value, lock):
        rsts = []
        for i, task in enumerate(tasks):
            rst = self.target(*task)
            rsts.append(rst)
            if ((i != 0 and i % 1000 == 0) or i == len(tasks) - 1) and len(rsts) > 0:
                if self.use_queue:
                    self.queue.put([rst for rst in rsts if rst is not None])
                with lock:
                    share_value.value += len(rsts)
                rsts.clear()

from functools import wraps
import time


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        results = function(*args, **kwargs)
        t1 = time.time()
        print('Total running time: %s minutes' % str((t1 - t0) / 60))
        return results
    return function_timer


@fn_timer
def test_f(x):
    s = 1
    for i in range(x):
        s += (x + 1)
        print(s)
    return s


if __name__ == '__main__':
    print(test_f(10000))
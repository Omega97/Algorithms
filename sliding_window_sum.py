"""
Sliding Window Sum (SWS) algorithms
for sliding statistics on streaming data
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time


class SWS:
    """Sliding Window Sum with only one bucket"""
    def __init__(self, window_size, n_units):
        self.window_size = window_size
        self.n_units = n_units
        self.count = None

    def run(self, bit_stream):
        self.count = 0
        for b in bit_stream:
            self.count -= self.count / self.window_size
            self.count += b
            yield self.count


class SWSSequential(SWS):
    """
    Sliding Window Sum with buckets
    that move along with the window
    """
    def __init__(self, window_size, n_units):
        super().__init__(window_size, n_units)
        self.buckets = None
        self.k = None
        self._compute_k()

    def _compute_k(self, tol=1e-6):
        """
        Compute the value of k that minimizes gives the
        correct update for the sliding window sum
        """
        n = self.window_size
        m = self.n_units
        s = n / m
        gamma = (s - 1) * (n - s) / s ** 2
        f = lambda x: (x - x ** m) / (1 - x) - gamma
        x0 = gamma / (gamma + 1)

        # Newton's method for computing the root of f
        for i in range(10):
            y = f(x0)
            y_der = (f(x0 + tol) - f(x0 - tol)) / (2 * tol)
            x0 -= y / y_der
        self.k = (s-1) / x0

    def run(self, bit_stream):
        self.buckets = np.zeros(self.n_units)
        for b in bit_stream:
            self.buckets -= self.buckets / self.window_size * self.n_units
            self.buckets[0] += b
            self.buckets[1:] += self.buckets[:-1] / self.k
            yield np.sum(self.buckets)


class SWSFlexible(SWS):
    """
    More general method, where we can preemptively
    choose the size of the buckets
    """
    def __init__(self, window_size, bucket_sizes, tol=1e-6):
        assert np.abs(np.sum(bucket_sizes) - window_size) < tol
        super().__init__(window_size, len(bucket_sizes))
        self.bucket_sizes = np.array([1] + list(bucket_sizes))
        self.inv_bucket_sizes = 1 / self.bucket_sizes
        self.buckets = None
        self.sum = None

    def run(self, bit_stream):
        self.buckets = np.zeros(self.n_units + 1)
        self.sum = 0
        for b in bit_stream:
            density = self.buckets * self.inv_bucket_sizes
            self.buckets[0] = b
            self.buckets[1:] += density[:-1] - density[1:]
            self.sum += density[0] - density[-1]
            yield self.sum


class SWSMovingBuckets(SWS):
    """
    Buckets move along with the data stream, making
    the sum computation more precise. It also works
    perfectly until the window size is reached.
    """
    def __init__(self, window_size, n_units):
        super().__init__(window_size, n_units)
        self.buckets = None
        self.bucket_sizes = None
        self.default_size = window_size / (n_units-1)
        self.sum = None

    def _add_bucket(self, b):
        self.sum += b
        self.buckets = np.insert(self.buckets, 0, b)
        self.bucket_sizes = np.insert(self.bucket_sizes, 0, 1)

    def _pour_in_first_bucket(self, b):
        self.buckets[0] += b
        self.bucket_sizes[0] += 1
        self.sum += b

    def _spill_last_bucket(self):
        diff = self.buckets[-1] / self.bucket_sizes[-1]
        self.buckets[-1] -= diff
        self.sum -= diff
        self.bucket_sizes[-1] -= 1
        if self.buckets[-1] < .5:
            self.sum -= self.buckets[-1]
            self.buckets = self.buckets[:-1]
            self.bucket_sizes = self.bucket_sizes[:-1]

    def _get_best_size(self):
        """Get the best size for the next bucket"""
        s = self.default_size
        if self.buckets[0] == 0:
            s = self.window_size
        return s

    def run(self, bit_stream):
        self.sum = 0.
        self.buckets = np.array([])
        self.bucket_sizes = np.array([])

        for b in bit_stream:
            if len(self.buckets) == 0:
                self._add_bucket(b)
            else:
                if self.bucket_sizes[0] >= self._get_best_size():
                    self._add_bucket(b)
                else:
                    self._pour_in_first_bucket(b)
                while np.sum(self.bucket_sizes) > self.window_size:
                    self._spill_last_bucket()
            yield self.sum


class SWSAdaptiveBuckets(SWSMovingBuckets):
    """
    Use the SWSMovingBuckets algorithm, but adapt the
    size of the buckets based on the data distribution
    (work in progress...)
    """
    def __init__(self, window_size, n_units):
        super().__init__(window_size, n_units)
        self.default_size = window_size / (n_units+1)

    def _get_best_size(self):
        """Get the best size for the next bucket
        (we only need to improve this method)"""
        x = self.buckets[0] / self.bucket_sizes[0]
        y = 2*(2*(x-1/2))**2 + 1    # approximates Shannon entropy
        return self.default_size * y


def test_and_plot(n_units=8, window_size=2**8, stream_size=2000):
    np.random.seed(0)

    # create a bit stream
    x_ = np.arange(stream_size)
    # bits = np.random.random() * np.ones(stream_size)
    bits = np.random.random(stream_size)
    bits += np.sin(x_ / 30) * 0.2
    bits = np.array(bits < 0.5, dtype=int)

    # compute the ground truth
    true_count = np.cumsum(bits)
    true_count[window_size:] = true_count[window_size:] - true_count[:-window_size]

    # run the SWS algorithms
    sws = [SWS(window_size, n_units),
           SWSSequential(window_size, n_units),
           # SWSFlexible(window_size, np.ones(n_units) * window_size / n_units),
           SWSMovingBuckets(window_size, n_units),
           # SWSAdaptiveBuckets(window_size, n_units),
           ]

    # plots
    y = []
    times = []
    rmse = []
    for i in range(len(sws)):
        start = time()
        y.append(np.array(list(sws[i].run(bits))))
        times.append(time() - start)
        rmse.append(np.sqrt(np.mean((true_count - y[i]) ** 2)))

    plt.title(f'Window Size = {window_size}, n_buckets = {n_units}')
    plt.plot(true_count, label='true count', linewidth=2)
    for i in range(len(y)):
        plt.plot(y[i], linewidth=3, alpha=0.5,
                 label=f'{sws[i].__class__.__name__} (rmse%={rmse[i]/window_size:.2%})')

    plt.plot([window_size, window_size], [0, max(true_count)], 'k--', alpha=0.4)
    plt.legend()
    # plt.show()

    # print rmse
    print(f'\nRMSEs: {np.round(rmse, 2)}')


def main(stream_size=1200):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    plt.sca(ax[0, 0])
    test_and_plot(n_units=4, window_size=80, stream_size=stream_size)
    plt.ylabel('count')

    plt.sca(ax[0, 1])
    test_and_plot(n_units=20, window_size=80, stream_size=stream_size)

    plt.sca(ax[1, 0])
    test_and_plot(n_units=4, window_size=160, stream_size=stream_size)
    plt.ylabel('count')
    plt.xlabel('time')

    plt.sca(ax[1, 1])
    test_and_plot(n_units=20, window_size=160, stream_size=stream_size)
    plt.xlabel('time')

    plt.show()


if __name__ == '__main__':
    main()

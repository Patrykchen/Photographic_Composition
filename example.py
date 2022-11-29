# coding=utf-8
class BinarySearch(object):
    def binary_search(self, array, data):
        """二分查找法递归实现"""
        if len(array) == 0:
            return False
        array.sort()
        mid_index = len(array) // 2
        if array[mid_index] == data:
            return True
        return self.binary_search(array[mid_index + 1:], data) if data > array[mid_index] else \
            self.binary_search(array[:mid_index], data)

if __name__ == '__main__':
    array = [50, 77, 55, 29, 10, 30, 66, 18, 80, 51]
    search = BinarySearch()
    print('搜索结果：', search.binary_search(array, 77))
    print('搜索结果：', search.binary_search(array, 777))
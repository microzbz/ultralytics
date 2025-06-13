import math

def find_closest_point_with_index(target, points):
    """
    找到目标点在点列表中的最近点，并返回其索引、坐标和距离。

    参数:
        target (list): 目标点 [x, y]
        points (list): 点列表 [[x1, y1], [x2, y2], ...]

    返回:
        closest_index (int): 最近点在列表中的索引
        closest_point (list): 最近点的坐标
        min_distance (float): 最近距离
    """
    min_distance = float('inf')
    closest_index = -1
    closest_point = None

    for index, point in enumerate(points):
        dx = point[0] - target[0]
        dy = point[1] - target[1]
        distance = math.sqrt(dx**2 + dy**2)

        if distance < min_distance:
            min_distance = distance
            closest_point = point
            closest_index = index

    return closest_index, closest_point


target = [2, 3]
points = [[1, 1], [4, 5], [2, 4], [3, 2]]

index, point, distance = find_closest_point_with_index(target, points)

print("最近点索引:", index)
print("最近点坐标:", point)
print("最近距离:", distance)
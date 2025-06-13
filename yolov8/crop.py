import matplotlib.pyplot as plt

# 点坐标
pts = [[529.3309936523438, 1128.4700927734375], [1807.50244140625, 1082.4429931640625], [1776.80712890625, 230.03652954101562], [498.63568115234375, 276.0636901855469]]

# 回到第一个点，闭合轮廓
pts.append(pts[0])

# 拆分 x 和 y
x = [p[0] for p in pts]
y = [p[1] for p in pts]

plt.figure(figsize=(6, 8))
plt.plot(x, y, 'bo-')  # 点+线
plt.gca().invert_yaxis()  # 图像坐标系，y轴向下
plt.title("YOLOv8 OBB Box")
plt.grid(True)
plt.show()

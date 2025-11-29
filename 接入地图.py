import folium

# 1. 创建地图对象，中心点设为北京 (纬度, 经度)
m = folium.Map(location=[39.9042, 116.4074], zoom_start=12)

# 2. 添加一个标记
folium.Marker(
    [39.9042, 116.4074],
    popup='北京天安门',
    tooltip='点击查看'
).add_to(m)

# 3. 保存为网页文件
m.save("my_map.html")
print("地图已生成，请在浏览器打开 my_map.html")
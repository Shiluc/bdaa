import streamlit as st
import pandas as pd
import plotly.express as px

# --- 页面配置 ---
st.set_page_config(
    page_title="全球 GDP 可视化看板",
    page_icon="🌍",
    layout="wide"
)

# --- 1. 模拟数据 (2023/2024 估算数据，单位：十亿美元) ---
# 为了演示方便，这里硬编码了主要经济体的数据
data = {
    'Country': [
        'United States', 'China', 'Germany', 'Japan', 'India',
        'United Kingdom', 'France', 'Brazil', 'Italy', 'Canada',
        'Russia', 'Mexico', 'Australia', 'South Korea', 'Spain',
        'Indonesia', 'Turkey', 'Netherlands', 'Saudi Arabia', 'Switzerland'
    ],
    'ISO_Alpha_3': [
        'USA', 'CHN', 'DEU', 'JPN', 'IND',
        'GBR', 'FRA', 'BRA', 'ITA', 'CAN',
        'RUS', 'MEX', 'AUS', 'KOR', 'ESP',
        'IDN', 'TUR', 'NLD', 'SAU', 'CHE'
    ],
    'GDP_Billion_USD': [
        27360, 17794, 4456, 4212, 3730,
        3340, 3030, 2173, 2254, 2140,
        1997, 1788, 1723, 1712, 1580,
        1371, 1108, 1118, 1108, 888
    ],
    'Region': [
        'Americas', 'Asia', 'Europe', 'Asia', 'Asia',
        'Europe', 'Europe', 'Americas', 'Europe', 'Americas',
        'Europe', 'Americas', 'Oceania', 'Asia', 'Europe',
        'Asia', 'Asia', 'Europe', 'Asia', 'Europe'
    ]
}

df = pd.DataFrame(data)

# --- 2. 侧边栏控制区 ---
st.sidebar.header("⚙️ 筛选选项")
selected_region = st.sidebar.multiselect(
    "选择区域 (留空则显示全部):",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# 根据筛选过滤数据
if selected_region:
    filtered_df = df[df['Region'].isin(selected_region)]
else:
    filtered_df = df

# --- 3. 主界面布局 ---
st.title("🌍 2024 全球主要经济体 GDP 交互仪表盘")
st.markdown("该看板展示了全球主要国家的国内生产总值（GDP）估算数据。您可以缩放地图并悬停查看详情。")

# 关键指标卡片 (Top 3)
col1, col2, col3 = st.columns(3)
top_3 = filtered_df.nlargest(3, 'GDP_Billion_USD')

if len(top_3) >= 3:
    col1.metric(label=f"🥇 {top_3.iloc[0]['Country']}", value=f"${top_3.iloc[0]['GDP_Billion_USD']:,} B")
    col2.metric(label=f"🥈 {top_3.iloc[1]['Country']}", value=f"${top_3.iloc[1]['GDP_Billion_USD']:,} B")
    col3.metric(label=f"🥉 {top_3.iloc[2]['Country']}", value=f"${top_3.iloc[2]['GDP_Billion_USD']:,} B")

st.markdown("---")

# --- 4. 交互式地图 ---
st.subheader("🗺️ 全球 GDP 热力地图")

fig_map = px.choropleth(
    filtered_df,
    locations="ISO_Alpha_3",
    color="GDP_Billion_USD",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Plasma,
    projection="natural earth",
    title="全球各国家/地区 GDP 分布 (颜色越亮 GDP 越高)",
    labels={'GDP_Billion_USD': 'GDP (十亿美元)'}
)
fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# --- 5. 图表与数据表 ---
col_chart, col_data = st.columns([3, 2])

with col_chart:
    st.subheader("📊 GDP 排名 (Top 15)")
    fig_bar = px.bar(
        filtered_df.sort_values('GDP_Billion_USD', ascending=True).tail(15),
        x='GDP_Billion_USD',
        y='Country',
        orientation='h',
        text='GDP_Billion_USD',
        color='Region',
        title="按 GDP 排序的国家",
        labels={'GDP_Billion_USD': 'GDP (十亿美元)', 'Country': '国家'}
    )
    fig_bar.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_data:
    st.subheader("📝 详细数据")
    st.dataframe(
        filtered_df[['Country', 'GDP_Billion_USD', 'Region']].sort_values('GDP_Billion_USD', ascending=False),
        hide_index=True,
        height=400
    )
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# 读取数据
df = pd.read_csv('data.csv')

# 1. 数据概览
def data_overview():
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

# 2. 公司基本信息分析
def company_info_analysis():
    # 公司地理分布
    plt.figure(figsize=(12, 6))
    df['com_addr'].value_counts().plot(kind='bar')
    plt.title('公司地理分布')
    plt.xlabel('地区')
    plt.ylabel('公司数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 公司成立时间分布
    df['born_year'] = pd.to_datetime(df['born_data']).dt.year
    plt.figure(figsize=(12, 6))
    df['born_year'].value_counts().sort_index().plot(kind='bar')
    plt.title('公司成立年份分布')
    plt.xlabel('年份')
    plt.ylabel('公司数量')
    plt.tight_layout()
    plt.show()

    # 公司存活时间分析
    df['live_days'] = (pd.to_datetime(df['death_data']) - pd.to_datetime(df['born_data'])).dt.days
    plt.figure(figsize=(12, 6))
    sns.histplot(df['live_days'], bins=50)
    plt.title('公司存活时间分布')
    plt.xlabel('天数')
    plt.ylabel('公司数量')
    plt.tight_layout()
    plt.show()

    # 行业分类分析
    plt.figure(figsize=(12, 6))
    df['cat'].value_counts().plot(kind='bar')
    plt.title('行业分布')
    plt.xlabel('行业')
    plt.ylabel('公司数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 3. 融资情况分析
def funding_analysis():
    # 融资轮次分布
    plt.figure(figsize=(10, 6))
    df['financing'].value_counts().plot(kind='bar')
    plt.title('融资轮次分布')
    plt.xlabel('融资轮次')
    plt.ylabel('公司数量')
    plt.tight_layout()
    plt.show()

    # 融资金额分析
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_money'], bins=50)
    plt.title('融资金额分布')
    plt.xlabel('融资金额（千元）')
    plt.ylabel('公司数量')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# 4. 失败原因分析
def failure_reason_analysis():
    # 主要失败原因统计
    reasons = df['death_reason'].str.split().explode()
    reason_counts = reasons.value_counts()
    plt.figure(figsize=(12, 6))
    reason_counts.plot(kind='bar')
    plt.title('主要失败原因')
    plt.xlabel('原因')
    plt.ylabel('次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 失败原因与行业的关系
    industry_reasons = df.groupby('cat')['death_reason'].apply(lambda x: ' '.join(x)).str.split()
    industry_reason_counts = industry_reasons.apply(Counter)
    top_reasons = pd.DataFrame(industry_reason_counts.tolist(), index=industry_reason_counts.index).fillna(0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(top_reasons, cmap='YlOrRd')
    plt.title('各行业失败原因热力图')
    plt.tight_layout()
    plt.show()

# 5. 创始人/CEO分析
def ceo_analysis():
    # CEO背景信息统计
    education_keywords = ['硕士', '博士', 'MBA']
    for keyword in education_keywords:
        df[keyword] = df['ceo_des'].str.contains(keyword, na=False)
    
    plt.figure(figsize=(10, 6))
    df[education_keywords].sum().plot(kind='bar')
    plt.title('CEO教育背景')
    plt.xlabel('学历')
    plt.ylabel('人数')
    plt.tight_layout()
    plt.show()

# 6. 投资机构分析
def investor_analysis():
    # 主要投资机构统计
    investors = df['invest_name'].str.split('&').explode()
    investor_counts = investors.value_counts()
    plt.figure(figsize=(12, 6))
    investor_counts.head(20).plot(kind='bar')
    plt.title('前20大投资机构')
    plt.xlabel('投资机构')
    plt.ylabel('投资次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 7. 行业分析
def industry_analysis():
    # 各行业公司数量对比
    industry_counts = df['cat'].value_counts()
    plt.figure(figsize=(12, 6))
    industry_counts.plot(kind='bar')
    plt.title('各行业公司数量')
    plt.xlabel('行业')
    plt.ylabel('公司数量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 各行业平均存活时间对比
    industry_lifetime = df.groupby('cat')['live_days'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    industry_lifetime.plot(kind='bar')
    plt.title('各行业平均存活时间')
    plt.xlabel('行业')
    plt.ylabel('平均存活天数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 8. 相关性分析
def correlation_analysis():
    # 融资金额与存活时间的关系
    plt.figure(figsize=(10, 6))
    plt.scatter(df['total_money'], df['live_days'])
    plt.title('融资金额与存活时间关系')
    plt.xlabel('融资金额（千元）')
    plt.ylabel('存活时间（天）')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# 9. 时间序列分析
def time_series_analysis():
    # 公司成立数量的时间趋势
    df['born_year'] = pd.to_datetime(df['born_data']).dt.year
    yearly_foundings = df['born_year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    yearly_foundings.plot()
    plt.title('每年新成立公司数量')
    plt.xlabel('年份')
    plt.ylabel('公司数量')
    plt.tight_layout()
    plt.show()

    # 公司倒闭数量的时间趋势
    df['death_year'] = pd.to_datetime(df['death_data']).dt.year
    yearly_deaths = df['death_year'].value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    yearly_deaths.plot()
    plt.title('每年倒闭公司数量')
    plt.xlabel('年份')
    plt.ylabel('公司数量')
    plt.tight_layout()
    plt.show()

# 主函数
if __name__ == "__main__":
    data_overview()
    company_info_analysis()
    funding_analysis()
    failure_reason_analysis()
    ceo_analysis()
    investor_analysis()
    industry_analysis()
    correlation_analysis()
    time_series_analysis()
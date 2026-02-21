"""
构造CSI300/CSI500数据集并保存为CSV
包含个股特征和市场特征
"""
import qlib
import pandas as pd
import numpy as np
from qlib.data import D
from qlib.config import REG_CN
import os
from datetime import datetime

def init_qlib():
    """初始化 qlib"""
    try:
        # 尝试多个可能的数据路径
        data_paths = [
            './data/cn_data',  # 修复：数据在cn_data子目录下
            './data',
            '~/.qlib/qlib_data/cn_data',
            'C:/Users/gugu/.qlib/qlib_data/cn_data'
        ]
        
        for path in data_paths:
            try:
                qlib.init(provider_uri=path, region=REG_CN)
                print(f"[OK] qlib 初始化成功，数据路径: {path}")
                
                # 验证数据是否可用
                try:
                    test_stocks = list(D.instruments('csi300'))
                    if test_stocks and len(test_stocks) > 2 and test_stocks[0] != 'market':
                        print(f"  [OK] 验证成功：CSI300成分股数量: {len(test_stocks)}")
                        return True
                    else:
                        print(f"  [WARN] 警告：CSI300成分股获取异常，返回: {test_stocks}")
                except Exception as e:
                    print(f"  [WARN] 验证失败: {e}")
                
                return True
            except Exception as e:
                print(f"  [ERROR] 路径 {path} 初始化失败: {e}")
                continue
        
        print("[WARN] 无法找到数据，将使用示例数据模式")
        return False
    except Exception as e:
        print(f"[ERROR] qlib 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_stock_features(instruments, start_time, end_time):
    """
    获取个股特征
    
    Parameters:
    -----------
    instruments : list
        股票代码列表
    start_time : str
        开始时间，格式：'YYYY-MM-DD'
    end_time : str
        结束时间，格式：'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        个股特征数据，MultiIndex (datetime, instrument)
    """
    print(f"\n获取个股特征...")
    print(f"  股票数量: {len(instruments)}")
    print(f"  时间范围: {start_time} 至 {end_time}")
    
    # 定义个股特征
    stock_features = {
        # 基础OHLCV
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume',
        # 直接用 $vwap 在部分数据集中可能为空，后面会用 OHLC 近似补全
        'vwap': '$vwap',
        
        # 价格特征（历史收益率，用于预测）
        'return_1d': '($close - Ref($close, 1)) / Ref($close, 1)',   # 当前相对于1天前的收益率
        'return_5d': '($close - Ref($close, 5)) / Ref($close, 5)',   # 当前相对于5天前的收益率
        'return_20d': '($close - Ref($close, 20)) / Ref($close, 20)',  # 当前相对于20天前的收益率
        'high_low_ratio': '$high / $low',
        'price_range': '($high - $low) / $close',
        
        # 移动平均
        'ma5': 'Mean($close, 5)',
        'ma10': 'Mean($close, 10)',
        'ma20': 'Mean($close, 20)',
        'ma30': 'Mean($close, 30)',
        'ma60': 'Mean($close, 60)',
        
        # 技术指标（使用基础函数计算，避免使用不支持的操作符）
        'price_strength': '($close - Min($low, 14)) / (Max($high, 14) - Min($low, 14))',  # 价格相对强度
        'price_position': '($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20))',  # 价格在20日区间的位置
        
        # 成交量特征
        'volume_ma5': 'Mean($volume, 5)',
        'volume_ma20': 'Mean($volume, 20)',
        'volume_ratio': '$volume / Mean($volume, 20)',
        'price_volume_corr': 'Corr($close, $volume, 10)',
        
        # 波动率
        'volatility_5d': 'Std($close, 5) / Mean($close, 5)',
        'volatility_20d': 'Std($close, 20) / Mean($close, 20)',
        
        # 动量
        'momentum_5d': 'Ref($close, 5) / $close - 1',
        'momentum_20d': 'Ref($close, 20) / $close - 1',
        
        # 相对强度
        'relative_strength': '$close / Mean($close, 20)',
    }
    
    fields = list(stock_features.values())
    field_names = list(stock_features.keys())
    
    try:
        # 获取数据
        print(f"  正在获取数据...")
        print(f"  股票代码示例: {instruments[:3] if len(instruments) > 3 else instruments}")
        print(f"  特征字段数量: {len(fields)}")
        
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq='day'
        )
        
        print(f"  原始数据形状: {data.shape}")
        print(f"  原始列名: {list(data.columns[:5])}...")
        
        # 检查数据是否为空
        if data.empty:
            print(f"  [WARN] 警告：获取的数据为空！")
            return None
        
        # 检查是否有NaN值
        nan_count = data.isna().sum().sum()
        if nan_count > 0:
            print(f"  [WARN] 警告：数据中包含 {nan_count} 个NaN值")
        
        # 重命名列
        if len(data.columns) == len(field_names):
            data.columns = field_names
        else:
            print(f"  [WARN] 警告：列数不匹配，原始列数: {len(data.columns)}, 期望列数: {len(field_names)}")
            # 只重命名能匹配的列
            min_len = min(len(data.columns), len(field_names))
            data.columns = list(field_names[:min_len]) + list(data.columns[min_len:])
        
        # 如果 vwap 全是 NaN，则用 OHLC 近似计算 vwap
        if 'vwap' in data.columns:
            non_na_vwap = data['vwap'].notna().sum()
            if non_na_vwap == 0:
                print("  [INFO] vwap 列全为空，使用 (high + low + close) / 3 进行近似计算")
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
                else:
                    print("  [WARN] 缺少 high/low/close 列，无法补全 vwap")
        
        print(f"  [OK] 成功获取数据，形状: {data.shape}")
        print(f"  数据预览（前3行）:")
        print(data.head(3))
        
        return data
        
    except Exception as e:
        print(f"  [ERROR] 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"  提示: 请先下载数据或检查股票代码是否正确")
        return None

def calculate_market_features(stock_data):
    """
    从个股数据计算市场特征
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        个股特征数据，MultiIndex (datetime, instrument)
    
    Returns:
    --------
    pd.DataFrame
        市场特征数据，Index为datetime
    """
    print(f"\n计算市场特征...")
    
    try:
        market_features = {}
        
        # stock_data 是 MultiIndex (datetime, instrument)
        # 需要按datetime分组计算市场特征
        
        # 重置索引以便操作
        stock_reset = stock_data.reset_index()
        
        # 确保datetime是datetime类型
        stock_reset['datetime'] = pd.to_datetime(stock_reset['datetime'])
        
        # 1. 市场平均收益率（使用单只股票的日收益率）
        ret_col = 'return_1d' if 'return_1d' in stock_reset.columns else None
        
        if ret_col is not None:
            market_features['market_return'] = stock_reset.groupby('datetime')[ret_col].mean()
            market_features['market_return_std'] = stock_reset.groupby('datetime')[ret_col].std()
        
        # 2. 市场平均价格
        if 'close' in stock_reset.columns:
            market_features['market_avg_close'] = stock_reset.groupby('datetime')['close'].mean()
            market_features['market_close_std'] = stock_reset.groupby('datetime')['close'].std()
        
        # 3. 市场总成交量
        if 'volume' in stock_reset.columns:
            market_features['market_total_volume'] = stock_reset.groupby('datetime')['volume'].sum()
            market_features['market_avg_volume'] = stock_reset.groupby('datetime')['volume'].mean()
        
        # 4. 市场宽度（上涨股票比例）
        if ret_col is not None:
            def calc_breadth(group):
                up_count = (group > 0).sum()
                total_count = group.count()
                return up_count / total_count if total_count > 0 else 0
            market_features['market_breadth'] = stock_reset.groupby('datetime')[ret_col].apply(calc_breadth)
        
        # 5. 市场波动率
        if 'volatility_20d' in stock_reset.columns:
            market_features['market_volatility'] = stock_reset.groupby('datetime')['volatility_20d'].mean()
        
        # 转换为DataFrame
        market_df = pd.DataFrame(market_features)
        market_df.index.name = 'datetime'
        
        print(f"  [OK] 计算了 {len(market_features)} 个市场特征")
        print(f"  特征: {list(market_features.keys())}")
        
        return market_df
        
    except Exception as e:
        print(f"  [ERROR] 计算市场特征失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_labels(data, horizons=[1, 5, 20]):
    """
    添加标签列（未来收益率）
    
    在量化交易中，标签通常是未来收益率，用于预测股票未来的表现。
    常见的标签选择：
    1. 未来1日收益率：用于短期预测
    2. 未来5日收益率：用于中期预测
    3. 未来20日收益率：用于长期预测
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含close价格的数据（MultiIndex: datetime, instrument）
    horizons : list
        预测时间范围（天数），默认[1, 5, 20]
    
    Returns:
    --------
    pd.DataFrame
        添加了标签列的数据
    """
    print(f"\n添加标签列（未来收益率）...")
    
    try:
        # 重置索引以便操作
        data_reset = data.reset_index()
        
        if 'close' not in data_reset.columns:
            print("  [WARN] 数据中没有close列，无法计算标签")
            return data
        
        # 确保datetime是datetime类型
        data_reset['datetime'] = pd.to_datetime(data_reset['datetime'])
        
        # 按股票代码和日期排序
        data_reset = data_reset.sort_values(['instrument', 'datetime'])
        
        print(f"  计算标签范围: {horizons} 天")
        print(f"  总数据量: {len(data_reset)}")
        
        # 为每个horizon计算未来收益率
        for horizon in horizons:
            label_name = f'label_{horizon}d'
            print(f"  计算 {label_name}...")
            
            try:
                # 按股票分组，计算未来收益率
                # 使用shift(-horizon)获取未来horizon天的收盘价
                data_reset[label_name] = data_reset.groupby('instrument')['close'].shift(-horizon)
                
                # 计算未来收益率 = (未来价格 - 当前价格) / 当前价格
                data_reset[label_name] = (data_reset[label_name] - data_reset['close']) / data_reset['close']
                
                # 统计有效标签数量
                valid_count = data_reset[label_name].notna().sum()
                print(f"    [OK] {label_name}: {valid_count}/{len(data_reset)} 个有效值 ({valid_count/len(data_reset)*100:.1f}%)")
                
            except Exception as e:
                print(f"    [ERROR] 计算{label_name}失败: {e}")
                data_reset[label_name] = None
        
        # 调整列顺序：个股特征 → label 列 → 市场特征
        cols = list(data_reset.columns)
        label_cols = [c for c in cols if c.startswith('label_')]
        market_cols = [c for c in cols if c.startswith('market_')]
        other_cols = [c for c in cols if c not in label_cols and c not in market_cols]
        new_cols = other_cols + label_cols + market_cols
        data_reset = data_reset[new_cols]
        
        # 重新设置索引
        if 'datetime' in data_reset.columns and 'instrument' in data_reset.columns:
            data_reset = data_reset.set_index(['datetime', 'instrument'])
        
        print(f"  [OK] 标签添加完成")
        return data_reset
        
    except Exception as e:
        print(f"  [ERROR] 添加标签失败: {e}")
        import traceback
        traceback.print_exc()
        return data

def merge_features(stock_data, market_data):
    """
    合并个股特征和市场特征
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        个股特征数据
    market_data : pd.DataFrame
        市场特征数据
    
    Returns:
    --------
    pd.DataFrame
        合并后的数据
    """
    print(f"\n合并个股特征和市场特征...")
    
    try:
        # 将市场特征添加到个股数据
        # stock_data 是 MultiIndex (datetime, instrument)
        # market_data 是 Index (datetime)
        
        # 重置索引以便合并
        stock_reset = stock_data.reset_index()
        market_reset = market_data.reset_index()
        
        # 确保datetime列类型一致（转换为datetime类型）
        if 'datetime' in stock_reset.columns:
            stock_reset['datetime'] = pd.to_datetime(stock_reset['datetime'])
        if 'datetime' in market_reset.columns:
            market_reset['datetime'] = pd.to_datetime(market_reset['datetime'])
        
        # 合并市场特征
        merged = stock_reset.merge(
            market_reset,
            on='datetime',
            how='left'
        )
        
        # 设置索引
        merged = merged.set_index(['datetime', 'instrument'])
        
        print(f"  [OK] 合并完成，最终数据形状: {merged.shape}")
        print(f"  特征总数: {len(merged.columns)}")
        print(f"    - 个股特征: {len(stock_data.columns)}")
        print(f"    - 市场特征: {len(market_data.columns)}")
        
        # 检查是否有标签列
        label_cols = [col for col in merged.columns if col.startswith('label_')]
        if label_cols:
            print(f"    - 标签列: {len(label_cols)} ({', '.join(label_cols)})")
        
        return merged
        
    except Exception as e:
        print(f"  [ERROR] 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_to_csv(data, filename, output_dir='./datasets'):
    """
    保存数据为CSV文件
    
    Parameters:
    -----------
    data : pd.DataFrame
        要保存的数据
    filename : str
        文件名（不含扩展名）
    output_dir : str
        输出目录
    """
    print(f"\n保存数据为CSV...")
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 文件路径
        filepath = os.path.join(output_dir, f"{filename}.csv")
        
        # 重置索引以便保存为CSV
        data_to_save = data.reset_index()
        
        # 保存
        data_to_save.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"  [OK] 数据已保存到: {filepath}")
        print(f"  数据形状: {data_to_save.shape}")
        print(f"  文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        
        return filepath
        
    except Exception as e:
        print(f"  [ERROR] 保存失败: {e}")
        return None

def build_dataset_csi300(start_time='2020-01-01', end_time='2023-12-31', output_dir='./datasets'):
    """
    构造CSI300数据集
    
    Parameters:
    -----------
    start_time : str
        开始时间
    end_time : str
        结束时间
    output_dir : str
        输出目录
    """
    print("=" * 70)
    print("构造CSI300数据集")
    print("=" * 70)
    
    # 初始化qlib
    has_data = init_qlib()
    
    if not has_data:
        print("\n[WARN] 无法初始化qlib数据，将创建示例数据集模板")
        return create_template_dataset('CSI300', start_time, end_time, output_dir)
    
    try:
        # 获取CSI300成分股
        print("  正在获取CSI300成分股...")
        csi300_stocks_raw = D.instruments('csi300')
        print(f"  原始返回类型: {type(csi300_stocks_raw)}")
        print(f"  原始返回内容: {csi300_stocks_raw}")
        
        csi300_stocks = list(csi300_stocks_raw) if hasattr(csi300_stocks_raw, '__iter__') else []
        
        # 过滤掉配置项
        csi300_stocks = [s for s in csi300_stocks if s not in ['market', 'filter_pipe']]
        
        print(f"  过滤后CSI300成分股数量: {len(csi300_stocks)}")
        
        if not csi300_stocks or len(csi300_stocks) < 10:
            print("[WARN] 无法获取CSI300成分股，尝试从文件读取...")
            # 尝试直接从文件读取
            try:
                import pandas as pd
                csi300_file = './data/cn_data/instruments/csi300.txt'
                if os.path.exists(csi300_file):
                    # 文件是制表符分隔的，格式：股票代码\t开始日期\t结束日期
                    df = pd.read_csv(csi300_file, header=None, sep='\t', names=['stock', 'start', 'end'])
                    # 注意：qlib数据中的股票代码格式是 SH600000，不需要转换
                    # 直接使用原始格式
                    csi300_stocks = df['stock'].dropna().unique().tolist()
                    print(f"  [OK] 从文件读取到 {len(csi300_stocks)} 只股票")
                    print(f"  示例股票: {csi300_stocks[:5]}")
                else:
                    print(f"  [ERROR] 文件不存在: {csi300_file}")
                    raise FileNotFoundError
            except Exception as e:
                print(f"  ✗ 从文件读取失败: {e}")
                print("  使用示例股票代码")
                csi300_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', 
                               '600519.SH', '000858.SZ', '002415.SZ', '300015.SZ',
                               '600887.SH', '000063.SZ']
        
        print(f"\nCSI300成分股数量: {len(csi300_stocks)}")
        print(f"示例股票: {csi300_stocks[:5]}")
        
        # 获取个股特征
        stock_data = get_stock_features(csi300_stocks, start_time, end_time)
        
        if stock_data is None or stock_data.empty:
            print("\n[WARN] 无法获取实际数据，创建示例数据集模板")
            return create_template_dataset('CSI300', start_time, end_time, output_dir)
        
        # 计算市场特征
        market_data = calculate_market_features(stock_data)
        
        if market_data is None or market_data.empty:
            print("\n[WARN] 无法计算市场特征，仅保存个股特征")
            filename = f"CSI300_stock_features_{start_time}_{end_time}"
            return save_to_csv(stock_data, filename, output_dir)
        
        # 合并特征
        merged_data = merge_features(stock_data, market_data)
        
        if merged_data is None or merged_data.empty:
            print("\n[WARN] 合并失败，分别保存个股和市场特征")
            save_to_csv(stock_data, f"CSI300_stock_{start_time}_{end_time}", output_dir)
            save_to_csv(market_data, f"CSI300_market_{start_time}_{end_time}", output_dir)
            return None
        
        # 添加标签列（未来收益率）
        merged_data = add_labels(merged_data, horizons=[1, 5, 20])
        
        # 保存为CSV
        filename = f"CSI300_dataset_{start_time}_{end_time}"
        return save_to_csv(merged_data, filename, output_dir)
        
    except Exception as e:
        print(f"\n[ERROR] 构造数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_dataset_csi500(start_time='2020-01-01', end_time='2023-12-31', output_dir='./datasets'):
    """
    构造CSI500数据集
    """
    print("\n" + "=" * 70)
    print("构造CSI500数据集")
    print("=" * 70)
    
    # 初始化qlib
    has_data = init_qlib()
    
    if not has_data:
        print("\n[WARN] 无法初始化qlib数据，将创建示例数据集模板")
        return create_template_dataset('CSI500', start_time, end_time, output_dir)
    
    try:
        # 获取CSI500成分股
        print("  正在获取CSI500成分股...")
        csi500_stocks_raw = D.instruments('csi500')
        csi500_stocks = list(csi500_stocks_raw) if hasattr(csi500_stocks_raw, '__iter__') else []
        
        # 过滤掉配置项
        csi500_stocks = [s for s in csi500_stocks if s not in ['market', 'filter_pipe']]
        
        print(f"  过滤后CSI500成分股数量: {len(csi500_stocks)}")
        
        if not csi500_stocks or len(csi500_stocks) < 10:
            print("[WARN] 无法获取CSI500成分股，尝试从文件读取...")
            # 尝试直接从文件读取
            try:
                import pandas as pd
                csi500_file = './data/cn_data/instruments/csi500.txt'
                if os.path.exists(csi500_file):
                    # 文件是制表符分隔的，格式：股票代码\t开始日期\t结束日期
                    df = pd.read_csv(csi500_file, header=None, sep='\t', names=['stock', 'start', 'end'])
                    # 注意：qlib数据中的股票代码格式是 SH600000，不需要转换
                    # 直接使用原始格式
                    csi500_stocks = df['stock'].dropna().unique().tolist()
                    print(f"  [OK] 从文件读取到 {len(csi500_stocks)} 只股票")
                    print(f"  示例股票: {csi500_stocks[:5]}")
                else:
                    print(f"  [ERROR] 文件不存在: {csi500_file}")
                    raise FileNotFoundError
            except Exception as e:
                print(f"  ✗ 从文件读取失败: {e}")
                print("  使用示例股票代码")
                csi500_stocks = ['000858.SZ', '002415.SZ', '300015.SZ', '600887.SH', 
                               '000063.SZ', '002304.SZ', '300059.SZ', '600009.SH',
                               '000725.SZ', '002142.SZ']
        
        print(f"\nCSI500成分股数量: {len(csi500_stocks)}")
        print(f"示例股票: {csi500_stocks[:5]}")
        
        # 获取个股特征
        stock_data = get_stock_features(csi500_stocks, start_time, end_time)
        
        if stock_data is None or stock_data.empty:
            print("\n[WARN] 无法获取实际数据，创建示例数据集模板")
            return create_template_dataset('CSI500', start_time, end_time, output_dir)
        
        # 计算市场特征
        market_data = calculate_market_features(stock_data)
        
        if market_data is None or market_data.empty:
            print("\n[WARN] 无法计算市场特征，仅保存个股特征")
            filename = f"CSI500_stock_features_{start_time}_{end_time}"
            return save_to_csv(stock_data, filename, output_dir)
        
        # 合并特征
        merged_data = merge_features(stock_data, market_data)
        
        if merged_data is None or merged_data.empty:
            print("\n[WARN] 合并失败，分别保存个股和市场特征")
            save_to_csv(stock_data, f"CSI500_stock_{start_time}_{end_time}", output_dir)
            save_to_csv(market_data, f"CSI500_market_{start_time}_{end_time}", output_dir)
            return None
        
        # 添加标签列（未来收益率）
        merged_data = add_labels(merged_data, horizons=[1, 5, 20])
        
        # 保存为CSV
        filename = f"CSI500_dataset_{start_time}_{end_time}"
        return save_to_csv(merged_data, filename, output_dir)
        
    except Exception as e:
        print(f"\n[ERROR] 构造数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_template_dataset(index_name, start_time, end_time, output_dir):
    """
    创建示例数据集模板（当无法获取实际数据时）
    """
    print(f"\n创建 {index_name} 示例数据集模板...")
    
    # 创建日期范围
    dates = pd.date_range(start=start_time, end=end_time, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # 只保留工作日
    
    # 示例股票
    sample_stocks = ['000001.SZ', '000002.SZ', '600000.SH'] if index_name == 'CSI300' else ['000858.SZ', '002415.SZ', '300015.SZ']
    
    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, sample_stocks], names=['datetime', 'instrument'])
    
    # 创建示例数据（NaN值，需要用户填充）
    # 列顺序：个股特征 → label 列 → 市场特征
    columns = [
        # 基础特征
        'open', 'high', 'low', 'close', 'volume', 'vwap',
        # 价格特征（历史收益率）
        'return_1d', 'return_5d', 'return_20d', 'high_low_ratio', 'price_range',
        # 移动平均
        'ma5', 'ma10', 'ma20', 'ma30', 'ma60',
        # 技术指标
        'price_strength', 'price_position',
        # 成交量
        'volume_ma5', 'volume_ma20', 'volume_ratio', 'price_volume_corr',
        # 波动率
        'volatility_5d', 'volatility_20d',
        # 动量
        'momentum_5d', 'momentum_20d',
        # 相对强度
        'relative_strength',
        # 标签（未来收益率）
        'label_1d', 'label_5d', 'label_20d',
        # 市场特征
        'market_return', 'market_return_std', 'market_avg_close', 'market_close_std',
        'market_total_volume', 'market_avg_volume', 'market_breadth', 'market_volatility'
    ]
    
    template_df = pd.DataFrame(index=index, columns=columns)
    template_df = template_df.reset_index()
    
    # 保存模板
    filename = f"{index_name}_dataset_template_{start_time}_{end_time}"
    filepath = save_to_csv(template_df, filename, output_dir)
    
    print(f"\n  [WARN] 这是数据模板，所有值为NaN，需要填充实际数据")
    print(f"  模板包含 {len(columns)} 个特征列")
    print(f"  时间范围: {start_time} 至 {end_time}")
    print(f"  示例股票: {sample_stocks}")
    
    return filepath

def main():
    """主函数"""
    print("=" * 70)
    print("CSI300/CSI500 数据集构造工具")
    print("=" * 70)
    print("\n功能：")
    print("  1. 获取CSI300/CSI500个股特征（27个特征）")
    print("  2. 计算市场特征（8个特征）")
    print("  3. 合并特征并保存为CSV文件")
    print("  4. 方便后续转换为Pickle格式")
    
    # 设置参数
    # 注意：根据数据文件，数据只到2020-09-25，所以调整时间范围
    start_time = '2020-01-01'
    end_time = '2020-09-25'  # 数据只到2020-09-25
    output_dir = './datasets'
    
    print(f"\n参数设置：")
    print(f"  开始时间: {start_time}")
    print(f"  结束时间: {end_time}")
    print(f"  输出目录: {output_dir}")
    
    # 构造CSI300数据集
    csi300_file = build_dataset_csi300(start_time, end_time, output_dir)
    
    # 构造CSI500数据集
    csi500_file = build_dataset_csi500(start_time, end_time, output_dir)
    
    # 总结
    print("\n" + "=" * 70)
    print("数据集构造完成")
    print("=" * 70)
    
    if csi300_file:
        print(f"\n[OK] CSI300数据集: {csi300_file}")
    if csi500_file:
        print(f"[OK] CSI500数据集: {csi500_file}")
    
    print(f"\n下一步：")
    print(f"  1. 检查CSV文件数据")
    print(f"  2. 使用 pandas 读取CSV: pd.read_csv('{csi300_file or csi500_file}')")
    print(f"  3. 转换为Pickle: df.to_pickle('dataset.pkl')")

if __name__ == '__main__':
    main()

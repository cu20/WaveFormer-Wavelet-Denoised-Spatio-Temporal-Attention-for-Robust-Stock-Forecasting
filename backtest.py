"""
改进的回测脚本：加入数据泄露检查、交易成本和详细诊断
"""
import numpy as np
import tensorflow as tf
import os
import pickle
import argparse
from WaveFormer import WaveFormerModel
from data_tools import create_tf_dataset, compute_feature_stats, features_num, calc_ic

def load_model(model_path, d_feat, d_model, t_num_heads, s_num_heads, 
               gate_input_start_index, gate_input_end_index, dropout, beta):
    """加载保存的模型"""
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return None
    
    try:
        loaded_model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        model = WaveFormerModel(
            d_feat=d_feat,
            d_model=d_model,
            t_num_heads=t_num_heads,
            s_num_heads=s_num_heads,
            t_dropout_rate=dropout,
            s_dropout_rate=dropout,
            beta=beta,
            gate_input_start_index=gate_input_start_index,
            gate_input_end_index=gate_input_end_index,
            n_epochs=1,
            lr=1e-5,
            seed=0,
            train_stop_loss_thred=None,
            model_save_path=None
        )
        
        model.model = loaded_model
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def backtest_strategy_improved(model, test_dataset, num_stocks=4, rebalance_freq=1, 
                               transaction_cost=0.001, slippage=0.0005, print_details=True):
    """
    改进的回测策略，加入交易成本和详细诊断
    
    Parameters:
    -----------
    transaction_cost : float
        单边交易成本（手续费+印花税），默认0.1%
    slippage : float
        滑点成本，默认0.05%
    """
    print(f"\n开始回测（改进版）...")
    print(f"策略：每个批次选择预测值最高的 {num_stocks} 只股票")
    print(f"调仓频率：每 {rebalance_freq} 个批次调仓一次")
    print(f"交易成本：单边 {transaction_cost*100:.2f}% + 滑点 {slippage*100:.2f}%")
    
    model.model.trainable = False
    
    all_predictions = []
    all_labels = []
    all_batch_returns = []
    all_batch_returns_after_cost = []  # 扣除交易成本后的收益
    all_selected_indices = []
    all_selected_predictions = []
    all_selected_returns = []
    
    # 诊断统计
    batch_ic_list = []
    batch_ric_list = []
    prediction_sign_accuracy = []  # 预测值符号与实际收益符号的一致性
    total_transactions = 0
    
    batch_idx = 0
    previous_selected = set()  # 上一期持仓
    
    for data in test_dataset:
        features, labels = data
        labels_np = labels.numpy()
        
        # 模型预测
        predictions = model.model(features, training=False).numpy()
        
        all_predictions.append(predictions)
        all_labels.append(labels_np)
        
        # 计算批次内IC
        batch_ic, batch_ric = calc_ic(predictions, labels_np)
        if not np.isnan(batch_ic):
            batch_ic_list.append(batch_ic)
        if not np.isnan(batch_ric):
            batch_ric_list.append(batch_ric)
        
        # 每个 rebalance_freq 个批次调仓一次
        if batch_idx % rebalance_freq == 0:
            # 选择预测值最高的 num_stocks 只股票
            top_k = min(num_stocks, len(predictions))
            selected_indices = np.argsort(predictions)[-top_k:][::-1]
            selected_set = set(selected_indices)
            
            # 获取选中股票的实际收益
            selected_returns = labels_np[selected_indices]
            selected_preds = predictions[selected_indices]
            
            # 计算投资组合收益（等权重，未扣成本）
            portfolio_return = np.mean(selected_returns)
            
            # 计算交易成本
            # 换手率 = (新持仓 - 旧持仓交集) / 持仓数量
            if batch_idx == 0:
                turnover = 1.0  # 首次建仓，100%换手
            else:
                overlap = len(previous_selected & selected_set)
                turnover = (num_stocks - overlap) / num_stocks
            
            # 交易成本 = 换手率 * (买入成本 + 卖出成本 + 滑点)
            # 假设每次调仓都是全换，成本 = 换手率 * (单边成本 * 2 + 滑点)
            cost_per_trade = turnover * (transaction_cost * 2 + slippage)
            portfolio_return_after_cost = portfolio_return - cost_per_trade
            
            total_transactions += int(turnover * num_stocks)
            
            # 诊断：预测值符号与实际收益符号的一致性
            pred_sign = np.sign(selected_preds)
            actual_sign = np.sign(selected_returns)
            sign_accuracy = np.mean(pred_sign == actual_sign)
            prediction_sign_accuracy.append(sign_accuracy)
            
            all_batch_returns.append(portfolio_return)
            all_batch_returns_after_cost.append(portfolio_return_after_cost)
            all_selected_indices.append(selected_indices)
            all_selected_predictions.append(selected_preds)
            all_selected_returns.append(selected_returns)
            
            previous_selected = selected_set
            
            if print_details and batch_idx < 10:
                print(f"\n批次 {batch_idx}:")
                print(f"  批次大小: {len(predictions)}")
                print(f"  预测值统计: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, mean={np.mean(predictions):.4f}, std={np.std(predictions):.4f}")
                print(f"  预测值中正数个数: {np.sum(predictions > 0)}, 负数个数: {np.sum(predictions < 0)}")
                print(f"  批次IC: {batch_ic:.4f}, 批次RIC: {batch_ric:.4f}")
                print(f"  选择的股票索引: {selected_indices}")
                print(f"  选择的预测值: {selected_preds}")
                print(f"  实际收益: {selected_returns}")
                print(f"  符号准确率: {sign_accuracy:.2%}")
                print(f"  换手率: {turnover:.2%}")
                print(f"  投资组合收益(未扣成本): {portfolio_return:.4f}")
                print(f"  投资组合收益(扣成本后): {portfolio_return_after_cost:.4f}")
        
        batch_idx += 1
    
    # 合并所有结果
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # 计算回测指标（未扣成本）
    portfolio_returns = np.array(all_batch_returns)
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    total_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    
    # 计算回测指标（扣成本后）
    portfolio_returns_after_cost = np.array(all_batch_returns_after_cost)
    cumulative_returns_after_cost = np.cumprod(1 + portfolio_returns_after_cost) - 1
    total_return_after_cost = cumulative_returns_after_cost[-1] if len(cumulative_returns_after_cost) > 0 else 0
    
    # 年化收益率
    num_periods = len(portfolio_returns)
    if num_periods > 0:
        trading_days_per_batch = 5
        total_trading_days = num_periods * trading_days_per_batch
        years = total_trading_days / 250.0
        annualized_return = (1 + total_return) ** (1.0 / years) - 1 if years > 0 else 0
        annualized_return_after_cost = (1 + total_return_after_cost) ** (1.0 / years) - 1 if years > 0 else 0
    else:
        annualized_return = 0
        annualized_return_after_cost = 0
    
    # 收益率统计
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    mean_return_after_cost = np.mean(portfolio_returns_after_cost)
    
    # 夏普比率
    sharpe_ratio = mean_return / (std_return + 1e-8) * np.sqrt(250 / trading_days_per_batch) if std_return > 0 else 0
    sharpe_ratio_after_cost = mean_return_after_cost / (std_return + 1e-8) * np.sqrt(250 / trading_days_per_batch) if std_return > 0 else 0
    
    # 最大回撤
    if len(cumulative_returns) > 0:
        running_max = np.maximum.accumulate(1 + cumulative_returns)
        drawdown = (1 + cumulative_returns) / running_max - 1
        max_drawdown = np.min(drawdown)
    else:
        max_drawdown = 0
    
    # 胜率
    win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
    win_rate_after_cost = np.sum(portfolio_returns_after_cost > 0) / len(portfolio_returns_after_cost) if len(portfolio_returns_after_cost) > 0 else 0
    
    # 平均收益
    positive_returns = portfolio_returns[portfolio_returns > 0]
    negative_returns = portfolio_returns[portfolio_returns <= 0]
    avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
    avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
    
    # 计算整体IC
    overall_ic, overall_ric = calc_ic(all_predictions, all_labels)
    
    results = {
        # 未扣成本
        'total_return': total_return,
        'annualized_return': annualized_return,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        
        # 扣成本后
        'total_return_after_cost': total_return_after_cost,
        'annualized_return_after_cost': annualized_return_after_cost,
        'mean_return_after_cost': mean_return_after_cost,
        'sharpe_ratio_after_cost': sharpe_ratio_after_cost,
        'win_rate_after_cost': win_rate_after_cost,
        
        # 诊断指标
        'num_batches': len(portfolio_returns),
        'ic': overall_ic,
        'ric': overall_ric,
        'batch_ic_mean': np.mean(batch_ic_list) if batch_ic_list else 0,
        'batch_ic_std': np.std(batch_ic_list) if batch_ic_list else 0,
        'batch_ric_mean': np.mean(batch_ric_list) if batch_ric_list else 0,
        'sign_accuracy': np.mean(prediction_sign_accuracy) if prediction_sign_accuracy else 0,
        'total_transactions': total_transactions,
        
        # 详细数据
        'cumulative_returns': cumulative_returns,
        'cumulative_returns_after_cost': cumulative_returns_after_cost,
        'portfolio_returns': portfolio_returns,
        'portfolio_returns_after_cost': portfolio_returns_after_cost,
        'all_predictions': all_predictions,
        'all_labels': all_labels
    }
    
    return results

def print_backtest_results(results):
    """打印回测结果"""
    print("\n" + "="*80)
    print("回测结果汇总（改进版）")
    print("="*80)
    
    print(f"\n【收益指标 - 未扣成本】")
    print(f"  累计收益率:     {results['total_return']*100:.2f}%")
    print(f"  年化收益率:     {results['annualized_return']*100:.2f}%")
    print(f"  平均每期收益:   {results['mean_return']*100:.4f}%")
    print(f"  收益标准差:     {results['std_return']*100:.4f}%")
    
    print(f"\n【收益指标 - 扣成本后】")
    print(f"  累计收益率:     {results['total_return_after_cost']*100:.2f}%")
    print(f"  年化收益率:     {results['annualized_return_after_cost']*100:.2f}%")
    print(f"  平均每期收益:   {results['mean_return_after_cost']*100:.4f}%")
    print(f"  成本影响:       {(results['total_return'] - results['total_return_after_cost'])*100:.2f}%")
    
    print(f"\n【风险指标】")
    print(f"  最大回撤:       {results['max_drawdown']*100:.2f}%")
    print(f"  夏普比率(未扣成本): {results['sharpe_ratio']:.4f}")
    print(f"  夏普比率(扣成本后): {results['sharpe_ratio_after_cost']:.4f}")
    
    print(f"\n【交易统计】")
    print(f"  总交易批次:     {results['num_batches']}")
    print(f"  总交易次数:     {results['total_transactions']}")
    print(f"  胜率(未扣成本): {results['win_rate']*100:.2f}%")
    print(f"  胜率(扣成本后): {results['win_rate_after_cost']*100:.2f}%")
    print(f"  平均盈利:       {results['avg_win']*100:.4f}%")
    print(f"  平均亏损:       {results['avg_loss']*100:.4f}%")
    
    print(f"\n【预测质量诊断】")
    print(f"  整体IC:        {results['ic']:.4f} {'⚠️ 过低' if abs(results['ic']) < 0.03 else ''}")
    print(f"  整体RIC:       {results['ric']:.4f} {'⚠️ 过低' if abs(results['ric']) < 0.05 else ''}")
    print(f"  批次IC均值:    {results['batch_ic_mean']:.4f}")
    print(f"  批次IC标准差:  {results['batch_ic_std']:.4f}")
    print(f"  批次RIC均值:   {results['batch_ric_mean']:.4f}")
    print(f"  符号准确率:    {results['sign_accuracy']*100:.2f}%")
    
    # 警告信息
    print(f"\n【⚠️ 警告检查】")
    warnings = []
    if abs(results['ic']) < 0.03:
        warnings.append("IC过低，预测能力可能不足")
    if results['annualized_return'] > 100:
        warnings.append("年化收益率过高，可能存在数据泄露或未来函数")
    if results['win_rate'] > 0.8:
        warnings.append("胜率过高，可能存在过拟合或数据问题")
    if results['sign_accuracy'] < 0.5:
        warnings.append("符号准确率低于50%，预测方向可能有问题")
    
    if warnings:
        for w in warnings:
            print(f"  ⚠️ {w}")
    else:
        print("  ✅ 未发现明显问题")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='改进的回测脚本：加入交易成本和详细诊断')
    parser.add_argument('--model_path', type=str, default='saved_models/best_model_seed_0',
                       help='模型保存路径')
    parser.add_argument('--num_stocks', type=int, default=4,
                       help='每个批次选择的股票数量（默认4只）')
    parser.add_argument('--rebalance_freq', type=int, default=1,
                       help='调仓频率，每N个批次调仓一次（默认1）')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                       help='单边交易成本（默认0.1%%）')
    parser.add_argument('--slippage', type=float, default=0.0005,
                       help='滑点成本（默认0.05%%）')
    
    args = parser.parse_args()
    
    # 模型参数
    d_feat = features_num
    d_model = 512
    t_num_heads = 4
    s_num_heads = 2
    dropout = 0.5
    gate_input_start_index = 27
    gate_input_end_index = 34
    beta = 5
    
    # 加载数据
    print("Loading test data...")
    data_path = 'data'
    train_pkl_path = os.path.join(data_path, "pickle_train")
    test_pkl_path = os.path.join(data_path, 'pickle_test')
    
    with open(train_pkl_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_pkl_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    # 计算特征统计量
    feature_stats = compute_feature_stats(train_data, features_num)
    print("Feature stats computed.")
    
    # 创建测试数据集
    test_dataset = create_tf_dataset(test_data, shuffle=False, feature_stats=feature_stats)
    print(f"测试数据集批次数: {len(test_dataset)}")
    
    # 加载模型
    model = load_model(args.model_path, d_feat, d_model, t_num_heads, s_num_heads,
                      gate_input_start_index, gate_input_end_index, dropout, beta)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # 执行回测
    results = backtest_strategy_improved(model, test_dataset, 
                                        num_stocks=args.num_stocks,
                                        rebalance_freq=args.rebalance_freq,
                                        transaction_cost=args.transaction_cost,
                                        slippage=args.slippage)
    
    # 打印结果
    print_backtest_results(results)
    
    # 保存结果
    output_file = 'backtest_results_improved.npz'
    np.savez(output_file,
             cumulative_returns=results['cumulative_returns'],
             cumulative_returns_after_cost=results['cumulative_returns_after_cost'],
             portfolio_returns=results['portfolio_returns'],
             portfolio_returns_after_cost=results['portfolio_returns_after_cost'],
             total_return=results['total_return'],
             total_return_after_cost=results['total_return_after_cost'],
             annualized_return=results['annualized_return'],
             annualized_return_after_cost=results['annualized_return_after_cost'],
             sharpe_ratio=results['sharpe_ratio'],
             sharpe_ratio_after_cost=results['sharpe_ratio_after_cost'],
             max_drawdown=results['max_drawdown'],
             win_rate=results['win_rate'],
             win_rate_after_cost=results['win_rate_after_cost'],
             ic=results['ic'],
             ric=results['ric'],
             sign_accuracy=results['sign_accuracy'])
    print(f"\n回测结果已保存到: {output_file}")

if __name__ == "__main__":
    main()

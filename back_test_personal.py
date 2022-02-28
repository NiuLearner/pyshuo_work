import os
import sys
import datetime as dt
import pandas as pd
import pandas.tseries.offsets as toffsets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdate
sys.path.append('/mnt/mfs/open_lib/')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import shared_tools.back_test as bt
from shared_tools.send_email import send_email
from shared_tools.Factor_Evaluation_Common_Func import to_final_position



class BackTest:
    def __init__(self, factor_df, stk_r_df, univer_r_df, forbid_days,
                 factor_name, univer_name, mail_to='pyshuo@tongyicapital.com'):
        self.factor_df = factor_df
        self.stk_r_df = stk_r_df
        self.univer_r_df = univer_r_df
        self.forbid_days = forbid_days
        self.factor_name = factor_name
        self.univer_name = univer_name
        self.mail_to = mail_to

    def win_ratio_wtl(self, pnl_df, hold_period=20):
        '''
        计算胜率和盈亏比
        :param hold_period:
        :return:
        '''
        pnl_df_temp = pnl_df.iloc[2:, ]
        tradeday_num = len(pnl_df_temp)
        trade_winloss = []
        for i in range(int(np.ceil(tradeday_num / hold_period))):
            try:
                trade_winloss_temp = pnl_df_temp.iloc[hold_period * i:hold_period * (i + 1)].sum()
                trade_winloss.append(trade_winloss_temp)
            except:
                trade_winloss_temp = pnl_df_temp.iloc[hold_period * i:].sum()
                trade_winloss.append(trade_winloss_temp)
        trade_win = [i for i in trade_winloss if i > 0]
        trade_loss = [i for i in trade_winloss if i < 0]
        win_ratio = len(trade_win) / len(trade_winloss)
        win_to_loss = np.abs(np.mean(trade_win) / np.mean(trade_loss))
        return win_ratio, win_to_loss

    def AZ_Rank_IC(self, signal, pct_n, min_valids=None, lag=0):
        signal = signal.shift(lag)
        signal = signal.replace(0, np.nan)
        signal_rank = signal.rank(method='min', axis=1)
        pct_n_rank = pct_n.rank(method='min', axis=1)
        corr_df = signal_rank.corrwith(pct_n_rank, axis=1).dropna()
        if min_valids is not None:
            signal_valid = signal.count(axis=1)
            signal_valid[signal_valid < min_valids] = np.nan
            signal_valid[signal_valid >= min_valids] = 1
            corr_signal = corr_df * signal_valid
        else:
            corr_signal = corr_df
        return corr_signal

    def AZ_Rank_IR(self, signal, pct_n, min_valids=None, lag=0):
        corr_signal = self.AZ_Rank_IC(signal, pct_n, min_valids, lag)
        ic_mean = corr_signal.mean()
        ic_std = corr_signal.std()
        ir = ic_mean / ic_std
        return ir, corr_signal

    def back_test(self, hold_period=20, need_figure=True) -> pd.DataFrame:
        '''
        计算多空组合仓位，每日盈亏，以及累计盈亏
        :param factor_df: 因子数据
        :param stk_r_df: 股票日收益率
        :param hold_period: 持仓周期
        :return:
        '''
        long_pos = 1
        short_pos = -1
        pos_df = pd.DataFrame(columns=self.stk_r_df.columns)
        stk_r_df_temp = self.stk_r_df.loc[self.factor_df.index, :]
        tradeday_num = len(self.factor_df)

        for i in range(int(np.ceil(tradeday_num / hold_period))):
            stk_temp = self.factor_df.iloc[i, :]
            try:
                pos_df_temp = pd.DataFrame(columns=stk_r_df_temp.columns,
                                           index=stk_r_df_temp.index[hold_period * i:hold_period * (i + 1)])
            except:
                pos_df_temp = pd.DataFrame(columns=stk_r_df_temp.columns,
                                           index=stk_r_df_temp.index[hold_period * i:])
            cut_long = np.percentile(stk_temp.dropna(), 85)
            cut_short = np.percentile(stk_temp.dropna(), 15)

            hold_stk = stk_temp[stk_temp <= cut_long].index
            sell_stk = stk_temp[stk_temp >= cut_short].index
            hold_stk_pos = long_pos / len(hold_stk)
            sell_stk_pos = short_pos / len(sell_stk)
            for j in pos_df_temp.columns:
                if j in hold_stk:
                    pos_df_temp[j] = hold_stk_pos
                elif j in sell_stk:
                    pos_df_temp[j] = sell_stk_pos
                else:
                    pos_df_temp[j] = 0
            pos_df = pd.concat([pos_df, pos_df_temp], axis=0)
        pos_df = to_final_position(pos_df, self.forbid_days)
        pnl_df = (pos_df.shift(1) * stk_r_df_temp).sum(axis=1)
        cumpnl_df = pnl_df.cumsum()

        IC_IR, IC = self.AZ_Rank_IR(self.factor_df, self.stk_r_df, lag=1)
        sharpe = bt.AZ_Sharpe_y(pnl_df)
        Pot = bt.AZ_Pot(pos_df, cumpnl_df[-1])
        annual_return = bt.AZ_annual_return(pos_df, self.stk_r_df)
        MaxDrowdown = bt.AZ_MaxDrawdown(cumpnl_df).min()
        win_ratio, win_to_loss = self.win_ratio_wtl(pnl_df, hold_period=20)
        metrics_df = pd.Series({
            'IC均值': IC.mean(),
            'IR': IC_IR,
            '夏普比率': sharpe,
            '多空组合年化收益率': annual_return,
            '最大回撤': MaxDrowdown,
            '胜率': win_ratio,
            '盈亏比': win_to_loss
        }).round(4)

        if need_figure:
            fmt = '%.2f%%'
            fig = plt.figure(figsize=[16, 8])
            ax1 = fig.add_subplot(111)
            ax1.plot(cumpnl_df.fillna(0), label=f"{self.factor_name}_{self.univer_name}累计收益")
            ax1.plot(self.univer_r_df.cumsum(), label=f'{self.univer_name}累计收益')
            ax1.grid(linestyle='--')
            ax1.legend(loc='upper center', bbox_to_anchor=(0.65, 1.08),
                       borderaxespad=0., ncol=2, prop={'size': 16})
            ax1.grid(True)
            ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
            ax2 = ax1.twinx()
            ax2.bar(x=cumpnl_df.index,
                    height=bt.AZ_MaxDrawdown(cumpnl_df).values * 100, color='gray',
                    label=f"{self.factor_name}回撤")
            yticks = mtick.FormatStrFormatter(fmt)
            ax2.yaxis.set_major_formatter(yticks)
            ax2.xaxis.set_major_formatter(mdate.DateFormatter('%Y/%m/%d'))
            ax2.legend(loc='upper center', bbox_to_anchor=(0.15, 1.08),
                       borderaxespad=0., prop={'size': 16})
            fig_path = f"/mnt/mfs/temp/temp_fig/temp_fit_" \
                f"{dt.datetime.today().strftime('%Y%m%d%H%M%S')}_fig.png"
            fig.savefig(fig_path)
            to = [self.mail_to, ]
            file_path_list = [fig_path]
            text = f'{self.factor_name}回测图'
            subject = f'{self.factor_name}_{self.univer_name}因子回测结果'
            try:
                send_email(text, to, file_path_list, subject)
            finally:
                os.remove(fig_path)
            plt.close()
        else:
            pass

        return pos_df, pnl_df, cumpnl_df, metrics_df


# %%

class BackTest():
    def __init__(self, factor_data, rtn_data, univ_data, start_date, end_date,
                 factor_type, capital=10000, tradedays=None, refreshdays=None, stock_weights=None,
                 rf_rate=0.04, use_pctchg=True):
        self.position_record = {}       # 每个交易日持仓记录
        self.portfolio_record = {}      # 组合净值每日记录
        self.factor_data = factor_data  # 因子数据
        self.rtn_data = rtn_data        # 日收益率数据
        self.univ_data = univ_data      # 基础市场收益率数据
        self.capital = capital          # 初始资金额
        self.use_pctchg = use_pctchg    # 使用收益率或数值金额进行回测
        self.rf_rate = rf_rate          # 无风险利率

        if factor_type not in ['ls', 'hs']:
            raise TypeError('PARAM:请输入正确的因子类型，\'ls\'或者\'hg\'')
        if factor_type == 'ls' and use_pctchg ==True:
            raise AssertionError('PARAM:use_pctchg=True, factor_type can\'t be \'ls\'')

        self.factor_type = factor_type  # 'ls'多空策略 / 'hg'只做多
        self.start_date = start_date    # 回测开始日期
        self.end_date = end_date        # 回测结束日期

        self.curdate = None  # 当前调仓交易日对应日期
        self.lstdate = None  # 上一个调仓交易日对应日期

        if tradedays:  # 回测期内所有交易日list
            tradedays = pd.to_datetime(tradedays)
        else:
            tradedays = pd.to_datetime(self.univ_data.index)
        self.tradedays = sorted(tradedays)
        if stock_weights:
            self.stock_weights = stock_weights
        else:
            self.stock_weights = self.get_stocks_weights()
        if refreshdays:
            self.refreshdays = refreshdays
        else:
            self.refreshdays = self.get_refresh_days()

    def _get_date_idx(self, date):
        """
        返回传入的交易日对应在全部交易日列表中的下标索引
        """
        datelist = list(self.tradedays)
        date = pd.to_datetime(date)
        try:
            idx = datelist.index(date)
        except ValueError:
            datelist.append(date)
            datelist.sort()
            idx = datelist.index(date)
            if idx == 0:
                return idx + 1
            else:
                return idx - 1
        return idx

    def get_refresh_days(self):
        """
        获取调仓日期（回测期内的每个月首个交易日）
        可以根据持仓周期对return进行调整
        """
        tdays = self.tradedays
        sindex = self._get_date_idx(self.start_date)
        eindex = self._get_date_idx(self.end_date)
        tdays = tdays[sindex:eindex + 1]
        return [nd for td, nd in zip(tdays[:-1], tdays[1:])
                if td.month != nd.month]

    def _get_stock_weights(self, factor_series):
        """
        等权重做空因子排名前10%做多排名后10%的股票,或者只做多因子排名前15%的股票
        """
        df_stk = pd.DataFrame(index=factor_series.index)
        if self.factor_type == 'ls':
            cut_long = np.percentile(factor_series.dropna(), 80)
            cut_short = np.percentile(factor_series.dropna(), 10)
            long_stk_code = factor_series[factor_series <= cut_long].index
            short_stk_code = factor_series[factor_series >= cut_short].index
            stk_code =long_stk_code.append(short_stk_code)
            long_weight = 1/len(long_stk_code)
            short_weight = -1/len(short_stk_code)
            stock_weights_temp = {}
            for stk in stk_code:
                if stk in long_stk_code:
                    stock_weights_temp.update({stk:long_weight})
                if stk in short_stk_code:
                    stock_weights_temp.update({stk: short_weight})
                stock_weights_temp = pd.Series(stock_weights_temp, name=factor_series.name)
            stock_weights = pd.concat([df_stk, stock_weights_temp], axis=1)
            return stock_weights
        elif self.factor_type == 'hg':
            cut_long = np.percentile(factor_series.dropna(), 85)
            stk_code = factor_series[factor_series <= cut_long].index
            long_weight = 1/len(stk_code)
            stock_weights_temp = {}
            for stk in stk_code:
                stock_weights_temp.update({stk:long_weight})
                stock_weights_temp = pd.Series(stock_weights_temp, name=factor_series.name)
            stock_weights = pd.concat([df_stk, stock_weights_temp], axis=1)
            return stock_weights

    def get_stocks_weights(self):
        """
        获取所有调仓日期的股票权重
        """
        tdays = self.refreshdays
        factor_tdays = self.factor_data.loc[tdays]
        stocks_weights = pd.DataFrame(index=factor_tdays.columns)
        for i in range(factor_tdays.shape[0]):
            stock_weights = self._get_stock_weights(factor_series=factor_tdays.iloc[i,:])
            stocks_weights = stocks_weights.merge(stock_weights, how='left', left_index=True, right_index=True)
        return stocks_weights

    def _get_stocks_weights(self, date):
        """
        根据传入的交易日日期（当月第一个交易日）获取对应
        前一截面（上个月最后一个交易日）的该层对应的各股票权重
        """
        idx = self._get_date_idx(date)
        date = self.tradedays[idx - 1]
        cur_stk_weights = self.stock_weights.loc[:, date]
        return cur_stk_weights.dropna()

    def cal_weighted_pctchg(self, date):
        weights = self._get_stocks_weights(self.curdate)  # 取上一个截面的股票权重列表
        codes = weights.index
        pct_chg = self.rtn_data.loc[date, codes].values
        return codes, np.nansum(pct_chg * weights.values)  # 当天的股票收益*上期期末(当期期初)的股票权重=当天持仓盈亏

    def _get_latest_mktval(self, date):
        """
        获取传入交易日对应持仓市值
        """
        holdings = self.position_record[self.lstdate].items()
        holding_codes = [code for code, num in holdings]
        holding_nums = np.asarray([num for code, num in holdings])
        latest_price = self.rtn_data.loc[date, holding_codes].values
        holding_mktval = np.sum(holding_nums * latest_price)
        return holding_mktval

    def update_port_netvalue(self, date):
        """
        更新每日净值
        """
        if self.use_pctchg:
            stk_codes, cur_wt_pctchg = self.cal_weighted_pctchg(date)
            self.portfolio_record[date] = cur_wt_pctchg
        else:
            holding_mktval = self._get_latest_mktval(date)
            total_val = self.capital + holding_mktval
            self.portfolio_record[date] = total_val

    def rebalance(self, stocks_data):
        """
        调仓，实际将上一交易日对应持仓市值加入到可用资金中
        """
        if self.position_record:
            self.capital += self._get_latest_mktval(self.curdate)
        self._buy(stocks_data)

    def _buy(self, new_stocks_to_buy):
        """
        根据最新股票列表买入，更新可用资金以及当日持仓
        """
        codes = new_stocks_to_buy.index

        trade_price = self.rtn_data.loc[codes, self.curdate]
        stks_avail = trade_price.dropna().index

        weights = new_stocks_to_buy.loc[stks_avail]
        amount = weights / np.sum(weights) * self.capital
        nums = amount / trade_price.loc[stks_avail]

        self.capital -= np.sum(amount)
        self.position_record[self.curdate] = {code: num for code, num in zip(stks_avail, nums)}

    def run_backtest(self):
        """
        回测主函数
        """
        start_idx = self._get_date_idx(self.start_date)
        end_idx = self._get_date_idx(self.end_date)

        hold = False
        for date in self.tradedays[start_idx:end_idx + 1]:  # 对回测期内全部交易日遍历，每日更新净值
            if date in self.refreshdays:  # 如果当日为调仓交易日，则进行调仓
                hold = True
                idx = self.refreshdays.index(date)
                if idx == 0:
                    # 首个调仓交易日
                    self.curdate = date
                self.lstdate, self.curdate = self.curdate, date

                if not self.use_pctchg:
                    stocks_to_buy = self._get_stocks_weights(date)
                    if len(stocks_to_buy) > 0:
                        # 采用复权价格回测的情况下, 如果待买入股票列表非空，则进行调仓交易
                        self.rebalance(stocks_to_buy)

            if hold:
                # 在有持仓的情况下，对净值每日更新计算
                self.update_port_netvalue(date)

        # 回测后进行的处理
        self.after_backtest()

    def _get_benchmark(self):
        start_date, end_date = self.portfolio_record.index[0], self.portfolio_record.index[-1]
        return self.univ_data.loc[start_date:end_date]

    def after_backtest(self):
        # 主要针对净值记录格式进行调整，将pctchg转换为净值数值；
        # 同时将持仓记录转化为矩
        self.portfolio_record = pd.DataFrame(self.portfolio_record, index=[0]).T
        if self.use_pctchg:
            self.portfolio_record.columns = ['netval_pctchg']
            self.portfolio_record['net_value'] = self.capital * (1 + self.portfolio_record['netval_pctchg']).cumprod()
            # 将基准列加入到净值记录表中
            self.portfolio_record['benchmark_pctchg'] = self._get_benchmark()
            self.portfolio_record['benchmark_nv'] = (1 + self.portfolio_record['benchmark_pctchg']).cumprod()
            # 上期期末(本期期初)的股票权重就可以看成本期(期末)的股票持仓
            self.position_record = self.stock_weights.T.shift(1).T.dropna(how='all', axis=1)
        else:
            self.portfolio_record.columns = ['net_value']
            nv_ret = self.portfolio_record['net_value'] / self.portfolio_record['net_value'].shift(1) - 1
            self.portfolio_record['netval_pctchg'] = nv_ret.fillna(0)
            # 将基准列加入到净值记录表中
            bm = self._get_benchmark()
            self.portfolio_record['benchmark_nv'] = bm / bm[0]
            bm_ret = self.portfolio_record['benchmark_nv'] / self.portfolio_record['benchmark_nv'].shift(1) - 1
            self.portfolio_record['benchmark_pctchg'] = bm_ret.fillna(0)
            # 每期期初买入的股票数量就是每期的仓位
            # self.position_record = pd.DataFrame.from_dict(self.position_record)
            # 上期期末(本期期初)的股票权重就可以看成本期(期末)的股票持仓
            self.position_record = self.stock_weights.T.shift(1).T.dropna(how='all', axis=1)

    # 回测指标函数
    def _te(self, start_date=None, end_date=None):
        """
        跟踪误差
        """
        if start_date and end_date:
            pr = self.portfolio_record.loc[start_date:end_date]
        else:
            pr = self.portfolio_record
        td = (pr['netval_pctchg'] - pr['benchmark_pctchg'])
        te = np.sqrt(min(len(pr), 252)) * np.sqrt(1 / (len(td) - 1) * np.sum((td - np.mean(td)) ** 2))
        return te

    def _ic_rate(self, start_date=None, end_date=None):
        """ 信息比率
        """
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret_vol = self._annual_vol(excess_acc_ret, start_date, end_date)
        return (ann_excess_ret - self.rf_rate) / ann_excess_ret_vol

    def _turnover_rate(self, start_date=None, end_date=None):
        """ 换手率(双边,不除以2)
        """
        positions = self.position_record.fillna(0).T
        if start_date and end_date:
            positions = positions.loc[start_date:end_date]
        turnover_rate = np.sum(np.abs(positions - positions.shift(1)), axis=1)
        turnover_rate = np.mean(turnover_rate) * 12
        return turnover_rate

    def _winning_rate(self, start_date=None, end_date=None):
        """ 相对基准日胜率
        """
        nv_pctchg = self.portfolio_record['netval_pctchg']
        bm_pctchg = self.portfolio_record['benchmark_pctchg']
        if start_date and end_date:
            nv_pctchg, bm_pctchg = nv_pctchg.loc[start_date:end_date], bm_pctchg.loc[start_date:end_date]
        win_daily = (nv_pctchg > bm_pctchg)
        win_rate = np.sum(win_daily) / len(win_daily)
        return win_rate

    def _get_date_gap(self, start_date=None, end_date=None, freq='d'):
        if start_date is None and end_date is None:
            start_date = self.portfolio_record.index[0]
            end_date = self.portfolio_record.index[-1]
        days = (end_date - start_date) / toffsets.timedelta(1)
        if freq == 'y':
            return days / 365
        elif freq == 'q':
            return days / 365 * 4
        elif freq == 'M':
            return days / 365 * 12
        elif freq == 'd':
            return days

    def _annual_return(self, net_vals=None, start_date=None, end_date=None):
        """ 年化收益
        """
        if net_vals is None:
            net_vals = self.portfolio_record['net_value']
        if start_date and end_date:
            net_vals = net_vals.loc[start_date:end_date]
        total_rtn = net_vals.values[-1] / net_vals.values[0] - 1
        date_gap = self._get_date_gap(start_date, end_date, freq='d')
        exp = 365 / date_gap
        ann_ret = (1 + total_rtn) ** exp - 1
        if date_gap <= 365:
            return total_rtn
        else:
            return ann_ret

    def _annual_vol(self, net_vals=None, start_date=None, end_date=None):
        """ 年化波动
        """
        if net_vals is None:
            net_vals = self.portfolio_record['net_value']
        ret_per_period = net_vals / net_vals.shift(1) - 1
        ret_per_period = ret_per_period.fillna(0)
        if start_date and end_date:
            ret_per_period = ret_per_period.loc[start_date:end_date]
        ann_vol = ret_per_period.std() * np.sqrt(min(len(ret_per_period), 252))
        return ann_vol

    def _max_drawdown(self, acc_rets=None, start_date=None, end_date=None):
        """ 最大回撤
        """
        if acc_rets is None:
            acc_rets = self.portfolio_record['net_value'] / self.portfolio_record['net_value'].values[0] - 1
        if start_date and end_date:
            acc_rets = acc_rets.loc[start_date:end_date]
        max_drawdown = (1 - (1 + acc_rets) / (1 + acc_rets.expanding().max())).max()
        return max_drawdown

    def _sharpe_ratio(self, start_date=None, end_date=None, ann_ret=None, ann_vol=None):
        """ 夏普比率
        """
        if ann_ret is None:
            ann_ret = self._annual_return(None, start_date, end_date)
        if ann_vol is None:
            ann_vol = self._annual_vol(None, start_date, end_date)
        return (ann_ret - self.rf_rate) / ann_vol

    def _get_excess_acc_ret(self, start_date=None, end_date=None):
        bm_ret = self.portfolio_record['benchmark_pctchg']
        nv_ret = self.portfolio_record['netval_pctchg']
        if start_date and end_date:
            bm_ret = bm_ret.loc[start_date:end_date]
            nv_ret = nv_ret.loc[start_date:end_date]
        excess_ret = nv_ret.values.flatten() - bm_ret.values.flatten()
        excess_acc_ret = pd.Series(np.cumprod(1 + excess_ret), index=nv_ret.index)
        return excess_acc_ret

    def _ann_excess_ret(self, start_date=None, end_date=None):
        """ 年化超额收益
        """
        excess_acc_ret = self._get_excess_acc_ret(start_date, end_date)
        ann_excess_ret = self._annual_return(net_vals=excess_acc_ret, start_date=start_date, end_date=end_date)
        return ann_excess_ret

    def summary(self, start_date=None, end_date=None):
        # 如果没有指定周期,那默认就是全周期
        if start_date is None and end_date is None:
            start_date, end_date = self.portfolio_record.index[0], self.portfolio_record.index[-1]

        ann_ret = self._annual_return(None, start_date, end_date)  # 年化收益
        ann_vol = self._annual_vol(None, start_date, end_date)  # 年化波动
        sharpe = self._sharpe_ratio(start_date, end_date)  # 夏普比率
        max_wd = self._max_drawdown(None, start_date, end_date)  # 最大回撤
        ann_excess_ret = self._ann_excess_ret(start_date, end_date)  # 年化超额收益
        te = self._te(start_date, end_date)  # 跟踪误差
        ic_rate = self._ic_rate(start_date, end_date)  # 信息比率
        win_rate = self._winning_rate(start_date, end_date)  # 相对基准日胜率
        turnover_rate = self._turnover_rate(start_date, end_date)  # 换手率
        summary = {
            '年度收益': ann_ret,
            '年度波动': ann_vol,
            '夏普比率': sharpe,
            '最大回撤': max_wd,
            '年度超额收益': ann_excess_ret,
            '跟踪误差': te,
            '信息比率': ic_rate,
            '日胜率': win_rate,
            '换手率': turnover_rate
        }
        return pd.Series(summary)

    def summary_yearly(self):
        # 先要运行回测,产生结果
        if len(self.portfolio_record) == 0:
            raise RuntimeError("请运行回测函数后再查看回测统计.")
        #
        all_dates = self.portfolio_record.index
        # 每年第一个交易日列表
        start_dates = all_dates[:1].tolist() + list(
            before_date for before_date, after_date in zip(all_dates[1:], all_dates[:-1])
            if before_date.year != after_date.year)
        # 每年最后一个交易日列表
        end_dates = list(before_date for before_date, after_date in zip(all_dates[:-1], all_dates[1:])
                         if before_date.year != after_date.year) + all_dates[-1:].tolist()
        #
        res = pd.DataFrame()
        # 按年统计
        for sdate, edate in zip(start_dates, end_dates):
            summary_year = self.summary(sdate, edate)
            summary_year.name = str(sdate.year)
            res = pd.concat([res, summary_year], axis=1)
        # 整个周期统计一次
        summary_all = self.summary()
        summary_all.name = '总计'
        res = pd.concat([res, summary_all], axis=1)
        res = res.T[['年度收益', '年度波动', '夏普比率', '最大回撤', '年度超额收益', '跟踪误差', '信息比率', '日胜率', '换手率']]
        return res


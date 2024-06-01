from rqalpha.api import *


def init(context):
    context.S1 = "510500.XSHG"
    context.UNIT = 10000
    context.INIT_S = 2
    context.MARGIN = 0.08
    context.FIRST_P = 0
    context.holdid = 0
    context.sellcount = 0
    context.inited = False
    logger.info("RunInfo: {}".format(context.run_info))


def before_trading(context):
    pass


def current_p(context):
    return context.FIRST_P - ((context.holdid * context.MARGIN) * context.FIRST_P)


def next_buy_p(context):
    if context.portfolio.cash < context.UNIT:
        return -1
    return context.FIRST_P - (((context.holdid + 1) * context.MARGIN) * context.FIRST_P)


def next_sell_p(context):
    if context.portfolio.market_value < context.UNIT:
        return -1
    return context.FIRST_P - (((context.holdid - 1) * context.MARGIN) * context.FIRST_P)


def handle_bar(context, bar_dict):
    bar = bar_dict[context.S1]
    if context.inited is True:
        nextB = next_buy_p(context)
        nextS = next_sell_p(context)
    if context.inited is False:
        context.inited = True
        order_value(context.S1, context.UNIT * context.INIT_S, price=bar.close)
        context.current_cash = 0
        context.holdid = 0
        context.FIRST_P = bar.open
        logger.info("Make first fire portfolio: {}".format(context.portfolio))

    elif bar.low <= nextB <= bar.high:
        res = order_value(context.S1, context.UNIT, nextB)
        if res.status == ORDER_STATUS.FILLED:
            context.holdid += 1
        else:
            logger.info("Buy failed: {}".format(res))

    elif bar.high < nextB:
        res = order_value(context.S1, context.UNIT, price=bar.high)
        if res.status == ORDER_STATUS.FILLED:
            context.holdid += 1
        else:
            logger.info("Buy failed: {}".format(res))

    elif bar.low <= nextS <= bar.high:
        res = order_value(context.S1, -1 * context.UNIT, price=nextS)
        if res.status == ORDER_STATUS.FILLED:
            context.holdid -= 1
            context.sellcount += 1
            logger.info("----- Sell count: {}".format(context.sellcount))
        else:
            logger.info("Sell failed: {}".format(res))

    elif nextS != -1 and bar.low > nextS:
        res = order_value(context.S1, -1 * context.UNIT, price=bar.low)
        if res.status == ORDER_STATUS.FILLED:
            context.holdid -= 1
            context.sellcount += 1
            logger.info("----- Sell count: {}".format(context.sellcount))
        else:
            logger.info("Sell failed: {}".format(res))


def after_trading(context):
    logger.info("Hold count: {}".format(context.holdid + 1))
    profit = (context.portfolio.cash + context.portfolio.market_value - context.portfolio.starting_cash)
    profit_pct = profit / (context.portfolio.market_value - profit)
    logger.info("after_trading: market_value {}, profit {}, percent {}".
                format(context.portfolio.market_value, profit, profit_pct))

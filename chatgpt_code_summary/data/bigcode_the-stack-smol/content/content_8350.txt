import pyupbit
import time
from datetime import datetime


# 초기화 준비
def init_prepairing(investable_coins_map, all_market_codes, all_market_names, order_money):
    # 이전 투자 시 코인 별 전날 대비 상승률
    prev_coins_map = pyupbit.get_prev_dict(investable_coins_map, all_market_codes, all_market_names)
    # 투자할 만한 코인 목록 가져오기
    investable_coins_map = get_investable_coin_map(all_market_codes, all_market_names)

    slack_message = f"""
    현재코인들 수익률 ::: {investable_coins_map}
    이전코인들 수익률 ::: {prev_coins_map}
    """
    pyupbit.send_message(pyupbit.get_slack_channel(), slack_message)
    # 투자 할 코인 1개 가져오기
    best_coin = get_best_coin_name(investable_coins_map, prev_coins_map)
    # 매수
    init(best_coin, order_money)


# 계좌에 보유한 코인이 없는 상태로 만들고 -> 매수 시작!
def init(best_coin='', order_money=0):
    init_counter = 0
    print(f"이번시간에 투자할 코인은? {best_coin}")
    # 가장 살만할 것 같은 코인 매수
    response = pyupbit.order_best_coin(best_coin, order_money)
    print(f'주문 결과 ::: {response} / uuid ::: {pyupbit.get_order_bid_uuid(response.json())}')
    # 주문 성공 시 매수 완료 될 때 까지 대기
    if 200 <= response.status_code <= 299:
        # 매수 신청 후 매수 될 때까지 대기
        while pyupbit.get_my_coin_info() is None:
            # 1초에 한번 매수 되었는지 확인
            time.sleep(1)
            init_counter = init_counter + 1
            print('매수 체결 대기 중...')
            if init_counter >= 30:
                print(f'아직 사지지 않았습니다. 30초 후 다시 초기화 작업 시작합니다..')
                # 너무 오래 걸리면 주문 취소, 30초 후 다시 매수 시도
                pyupbit.cancel_order(pyupbit.get_order_bid_uuid(response.json()))
                time.sleep(30)
                init(best_coin, order_money)
    # 주문 실패 시 재 주문 시도(10초 후)
    else:
        print(f'재 주문 시도(10초 후 다시 초기화 작업 시작합니다.)...{response.status_code} / {response.json()}')
        time.sleep(10)
        init(best_coin, order_money)


# 투자해도 될 것 같은 코인 목록 조회
def get_investable_coin_map(market_codes=[], market_names=[]):
    investable_coins_map = {}
    i = 0
    for code in market_codes:
        # coin = { 코인 코드 : 현재가와 1차 저항선 간 차이% }
        coin = pyupbit.get_investable_coins(code, market_names[i])
        if coin is not None:
            investable_coins_map.update(coin)
        time.sleep(0.3)
        i = i + 1
    return investable_coins_map


# 투자해도 될 것 같은 코인 중 가장 좋을 것 같은 코인 조회
def get_best_coin_name(investable_coins_map={}, prev_coins_map={}):
    while True:
        if dict(investable_coins_map):
            print(f'original_map ::: {investable_coins_map}')
            if dict(prev_coins_map):
                print(f'prev_coins_map ::: {prev_coins_map}')
                # 코인 맵에서 이전 상승률 보다 현재 상승률이 낮은 코인 제거
                filtered_map = pyupbit.map_filtering(prev_coins_map, investable_coins_map)
                print(f'original_map :: {investable_coins_map} / filtered_map :: {filtered_map}')
                investable_coins_map = filtered_map
            if dict(investable_coins_map):
                # investable_coins_map = { 코인 코드 : 현재가와 1차 저항선 간 차이% }
                # 투자 대상 코인을 현재가와 1차 저항선 간 차이 기준으로 정렬(asc)
                coins_map = sorted(investable_coins_map.items(), reverse=True, key=lambda item: item[1])
                # 현재가와 1차 저항선 간 차이가 가장 작은 코인
                best_coin = list(coins_map[0])[0]
                # 현재가와 1차 저항선 간 차이
                coin_dynamic_rate = list(coins_map[0])[1]
                slack_message = f"best_coin ::: {best_coin} / change_rate(현재가 - 1차 저항선) ::: {coin_dynamic_rate}%"
                print(slack_message)
                pyupbit.send_message(pyupbit.get_slack_channel(), slack_message)
                return best_coin
        else:
            slack_message = f':meow_code: 살만한 코인이 없습니다.. 10분 후 다시 초기화 작업 시작합니다..'
            print(slack_message)
            time.sleep(600)
            pyupbit.send_message(pyupbit.get_slack_channel(), slack_message)
            return recursive_get_investable_coin_map(prev_coins_map)


# 살만한 코인이 없는 경우 코인 목록 재 조회
def recursive_get_investable_coin_map(prev_coins_map={}):
    # 전체 코인 코드
    all_market_codes = pyupbit.all_market_names.view_market_codes()
    # 전체 코인 이름
    all_market_names = pyupbit.all_market_names.view_market_names()
    investable_coins_map = get_investable_coin_map(all_market_codes, all_market_names)
    return get_best_coin_name(investable_coins_map, prev_coins_map)


# 빡침 스코어 기록기(안씀)
def calc_profit_score(rage_score=0, prev_profit_rate=0, current_profit_rate=0):
    """
    매도 할 타이밍은 스코어가 5점 이상인 경우로 한다.
    1. 절대 수익률이 100% 보다 높은 경우
      - 직전 수익률 보다 떨어졌을 때(+)
        rage_score = rage_score + minus_change_rate * 2
      - 직전 수익률 보다 올라갔을 때(-)
        rage_score = rage_score + minus_change_rate / 2
    2. 절대 수익률이 100% 보다 낮은 경우는 그냥 97% 미만일 때 매도 처리(빡침 스코어는 계산)
      - 직전 수익률 보다 떨어졌을 때(+)
        rage_score = rage_score + minus_change_rate * 2
      - 직전 수익률 보다 올라갔을 때(-)
        rage_score = rage_score + minus_change_rate * 1.5
    3. 빡침 스코어가 마이너스인 경우 0으로 처리
    """
    # 마이너스 변동폭(마이너스 / 플러스 반대)
    minus_change_rate = prev_profit_rate - current_profit_rate
    # 빡침 스코어 계산 하기!
    # 수익률 100% 이상
    if current_profit_rate >= 100:
        # 하락중... (그냥 팔까...)
        if minus_change_rate >= 0:
            rage_score = rage_score + minus_change_rate * 3
        # 상승중! (가즈아!!)
        else:
            rage_score = rage_score + minus_change_rate / 2
    # 수익률 100% 미만
    else:
        # 하락중... (아..)
        if minus_change_rate >= 0:
            rage_score = rage_score + minus_change_rate * 2
        # 상승중! (제발!!)
        else:
            rage_score = rage_score + minus_change_rate * 2
    slack_message = f'현재 점수는 ::: {round(rage_score, 2)} / 변동폭은 ::: {round(-minus_change_rate, 2)}% / 직전 수익률은 ::: {prev_profit_rate}% / 현재 수익률은 ::: {current_profit_rate}%'
    print(slack_message)
    if rage_score >= 6.5:
        pyupbit.send_message(pyupbit.get_slack_channel(), slack_message)
    elif rage_score < 0:
        rage_score = 0
    return rage_score


# 매도 / 매수 메인 로직(안씀)
def working(market='', my_investment={}, prev_profit_rate=100, score=0, has_minus_exp=False):
    # 해당 코인의 현재 상태(분 캔들) 조회
    coin_candle = pyupbit.view_candle_min(market)
    # 내가 매수 한 코인 단가
    buy_unit_price = pyupbit.get_my_coin_unit_price(my_investment)
    # 내 계좌에 남은 현금
    #krw_balance = pyupbit.get_my_krw_balance(my_investment)
    # 내 계좌에 남은 코인 수
    #my_coin_balance = pyupbit.get_my_coin_total_amount(my_investment)
    # 현재 코인 단가
    current_unit_price = pyupbit.get_current_coin_price(coin_candle)
    # 수익률(100%가 매수 시점 단가)
    profit_rate = pyupbit.get_profit_rate(current_unit_price, buy_unit_price)
    # 스코어(매도시점용)
    score = calc_profit_score(score, prev_profit_rate, profit_rate)
    slack_message1 = f"코인명 ::: {market}(현재빡침점수 : {round(score, 2)}), 매수단가 ::: {buy_unit_price}, 현재단가 ::: {current_unit_price}, 수익률 ::: {str(profit_rate)}%"
    print(slack_message1)
    if profit_rate < 100:
        has_minus_exp = True
    # 수익률 한번이라도 100% 미만인 경우 수익률 기준으로 매도 결정
    if has_minus_exp and profit_rate >= 100:
        pyupbit.sell_all()
        pyupbit.send_message(pyupbit.get_slack_channel(), f'[구사일생으로 팔았음.-{str(datetime.today())}]' + slack_message1)
        print('sell!!')
    else:
        # 매수할 만 하고 코인 단가가 내가 샀을때 보다 살짝 떨어져 있을 때 추가 매수 -> 일단 막기!!
        # if target_price >= current_unit_price and 99 >= profit_rate >= 97:
        # if krw_balance >= 10000:
        # 추가 매수 기능 막음
        # available_coin_amount = pyupbit.get_possible_order_volume(coin_candle, 10000)
        # pyupbit.order_10000(market, available_coin_amount, 'bid')
        # pyupbit.send_message('#myinvestment', f'[Buying!!-{str(datetime.today())}]' + slack_message1)
        #    print('buy!!')
        # 매도 매수 시점 판단 빡침 스코어 기준으로 변경!
        if score > 5:
            pyupbit.sell_all()
            pyupbit.send_message(pyupbit.get_slack_channel(), f'[빡쳐서 팔았음!!-{str(datetime.today())}]' + slack_message1)
            print('sell!!')
        # 수익률이 너무 떨어질 것 같을때 매도
        elif profit_rate < 99:
            pyupbit.sell_all()
            pyupbit.send_message(pyupbit.get_slack_channel(), f'[하락해서 팔았음... -{str(datetime.today())}]' + slack_message1)
            print('sell...')
        # 그 외 상태일 경우
        else:
            print('thinking...')
    # 수익률, 스코어 반환
    return [profit_rate, score, has_minus_exp]


# 잘 될 것 같은 코인 계산(안씀)
def get_rocketboosting_coins(candle_data, market_name):
    d = candle_data
    # 코인 코드
    market = pyupbit.get_market(d)
    # 목표 코인 단가( 오늘 시작가 + (어제 고가 - 어제 저가) * 0.5 )
    target_price = pyupbit.get_target_price_to_buy(market)
    # 코인 현재 단가
    current_price = pyupbit.get_current_coin_price(d)
    # 전날 대비 변동 률
    change_rate = pyupbit.get_change_rate(d)
    coin_info = pyupbit.get_coin_info_with_candle(d, market_name)
    # 현재 코인 단가가 목표가 보다 높고 단가가 1원 이상인 코인만 필터
    if current_price >= target_price and pyupbit.get_today_opening_price(d) > 1:
        print(f'대상 : {coin_info}')
        pyupbit.send_message(pyupbit.get_slack_channel(), coin_info)
        return {market: change_rate}
    else:
        #print(f'비대상 ::: {coin_info}')
        return None


# 코인 변동률 맵 조회(전체)(안씀)
def get_coin_rate_map(market_codes=[]):
    result_map = {}
    for market in market_codes:
        d = pyupbit.get_candle_data(market)
        # 전날 대비 변동 률
        change_rate = pyupbit.get_change_rate(d)
        result_map.update({market: change_rate})
        time.sleep(0.2)
    return result_map


# 일 캔들 데이터로 코인 정보 조회
def get_coin_info_with_candle(d, market_name):
    # 코인 코드
    market = pyupbit.get_market(d)
    # 목표 코인 단가( 오늘 시작가 + (어제 고가 - 어제 저가) * 0.5 )
    target_price = pyupbit.get_target_price_to_buy(market)
    # 코인 현재 단가
    current_price = pyupbit.get_current_coin_price(d)
    # 오늘 시가
    today_open_price = pyupbit.get_today_opening_price(d)
    # 어제 고가
    prev_high_price = pyupbit.get_yesterday_high_price(d)
    # 어제 저가
    prev_low_price = pyupbit.get_yesterday_low_price(d)
    # 기준선
    standard_price = pyupbit.calc_standard_line(prev_high_price, prev_low_price, today_open_price)
    # 1차 지지선
    first_low_price = pyupbit.first_lower_line(standard_price, prev_high_price)
    # 2차 지지선
    second_low_price = pyupbit.second_lower_line(standard_price, prev_high_price, prev_low_price)
    # 1차 저항선
    first_high_price = pyupbit.first_higher_line(standard_price, prev_low_price)
    # 2차 저항선
    second_high_price = pyupbit.second_higher_line(standard_price, prev_high_price, prev_low_price)
    coin_info = f"""
    현재시간 : {datetime.today()}
    코인명: {market} ({market_name}:{str(pyupbit.get_change_rate(d))}%) 
    opening_p:{str(pyupbit.get_today_opening_price(d))} 
    high_p(오늘[어제]):{str(pyupbit.get_today_high_price(d))}[{str(pyupbit.get_yesterday_high_price(d))}] 
    low_p(오늘[어제]):{str(pyupbit.get_today_low_price(d))}[{str(pyupbit.get_yesterday_low_price(d))}] 
    prev_p:{str(pyupbit.get_yesterday_close_price(d))} 
    change_p:{str(pyupbit.get_change_price(d))}
    기준선 : {standard_price}
    1차 지지선 : {first_low_price}
    2차 지지선 : {second_low_price}
    1차 저항선 : {first_high_price}
    2차 저항선 : {second_high_price}
    목표가 : {first_high_price}
    현재가 : {current_price}
    """
    return coin_info


# 목표 코인 단가 계산(안씀)
def get_target_price_to_buy(market="KRW-BTC"):
    d = pyupbit.get_candle_data(market)
    return d[0]['opening_price'] + (d[1]['high_price'] - d[1]['low_price']) * 0.5


"""
맵 객체 값으로 나쁜 코인 필터링(수익률 필터링)
직전 수익률과 현재 수익률 기준으로
투자 하지 말아야 할 코인들 필터링(직전 보다 현재 가격이 같거나 높은 코인들.. old_value <= new_value)
"""


# 나쁜 코인 필터링
def map_filtering(original_map, new_map):
    bad_arr = []
    for old_key, old_value in original_map.items():
        if old_key in new_map:
            new_value = new_map[old_key]
            # 요 부등호가 중요함!
            if old_value >= new_value:
                bad_arr.append(old_key)
    print(f'나쁜코인목록 ::: {bad_arr}')
    for old_key in bad_arr:
        new_map.pop(old_key, None)
    return new_map

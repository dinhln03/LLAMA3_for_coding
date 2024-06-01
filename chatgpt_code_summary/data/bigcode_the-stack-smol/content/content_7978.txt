import pandas as pd
import numpy as np

from scipy import stats


def columns_views(player_1_df, player_2_df):

    columns = list(player_1_df.columns)
    if list(player_1_df.columns) == list(player_2_df.columns):
        columns = list(player_1_df.columns)

        player_1 = list(player_1_df.values[0])
        player_2 = list(player_2_df.values[0])

        views = []
        for column, player1, player2 in zip(columns, player_1, player_2):
            print('column : {} _ player1-{} , player2-{} < diff : {} >'.format(
                column, player1, player2, abs(player1 - player2)
            ))
            views.append(abs(player1 - player2))

        print(views)


def convert_preferred_foot(df):

    df['preferred_foot'] = df['preferred_foot'].replace('Right', 1)
    df['preferred_foot'] = df['preferred_foot'].replace('Left', 2)

    return df


def convert_work_rate(df):

    convert = {
        'High': 3,
        'Medium': 2,
        'Low': 1
    }

    work_rate = df['work_rate'].values[0].split('/')

    attack = work_rate[0]
    defense = work_rate[1]

    df['attack'] = convert[attack]
    df['defense'] = convert[defense]

    # work_rateの削除処理
    df = df.drop(columns='work_rate')

    return df


def euclidean_distance(v1, v2):
    # ユーグリッド距離を算出
    # https://qiita.com/shim0mura/items/64918dad83d162ef2ac2#ユークリッド距離

    # どちらも同じ値を返す
    # distance = np.linalg.norm(v1 - v2)
    distance = np.sqrt(np.power(v1 - v2, 2).sum())

    # 0から1までの値で似ていれば似ているほど1に近くなる、みたいな類似度として分かりやすい値が欲しい。
    # 0での除算エラーを防ぐためにこのdに1を足して逆数をとるとそのような値を取ることが出来る。
    # 1/(1+d)

    # print('distance', distance)

    return 1 / (1 + distance)


def cos_similarity(v1, v2):
    # Scipyを使ってコサイン類似度を求める方法
    # import scipy.spatial.distance as dis

    # print(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    # print(dis.cosine(v1, v2))

    # return dis.cosine(v1, v2)

    # cos類似度を算出
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ピアソンの積率相関係数
def pearson_product_moment_correlation_coefficien(v1, v2):
    # corr = np.corrcoef(v1, v2)[0, 1]
    corr = stats.pearsonr(v1, v2)

    return corr


# スピアマンの順位相関係数
def spearman_rank_correlation_coefficient(v1, v2):
    corr = stats.spearmanr(v1, v2)

    return corr


# ケンドールの順位相関係数
def kendalltau_rank_correlation_coefficient(v1, v2):
    corr = stats.kendalltau(v1, v2)

    return corr


def similarity(v1_df, v2_df):
    v1_value = v1_df.values[0]
    v2_value = v2_df.values[0]

    print('v1_value', v1_value)
    print('v2_value', v2_value)

    # リストをps.Seriesに変換
    s1 = pd.Series(list(v1_value))
    s2 = pd.Series(list(v2_value))

    # 相関係数を計算
    res = s1.corr(s2)
    print(res)

    corr = pearson_product_moment_correlation_coefficien(
        v1_value, v2_value
    )
    print('pearson_product_moment_correlation_coefficien', corr)

    corr = spearman_rank_correlation_coefficient(
        v1_value, v2_value
    )
    print('spearman_rank_correlation_coefficient', corr)

    corr = kendalltau_rank_correlation_coefficient(
        v1_value, v2_value
    )
    print('kendalltau_rank_correlation_coefficient', corr)

    e_distance = euclidean_distance(v1_value, v2_value)
    print('e_distance', e_distance)

    # return euclidean_distance(v1_value, v2_value)
    # return res
    return cos_similarity(v1_value, v2_value)


# 数値型の整形
def shaping_num(value):

    if '+' in value:
        value = str(value).split('+')
        value = int(value[0]) + int(value[1])
        return value

    if '-' in value:
        value = str(value).split('-')
        value = int(value[0]) - int(value[1])
        return value

    return value


def need_columns(df):
    columns = [
        'height_cm',
        'weight_kg',
        'preferred_foot',
        'weak_foot',
        'skill_moves',
        'work_rate',
        'player_tags',

        'pace',
        'shooting',
        'passing',
        'dribbling',
        'defending',
        'physic',
        'player_traits',

        'attacking_crossing',
        'attacking_finishing',
        'attacking_heading_accuracy',
        'attacking_short_passing',
        'attacking_volleys',

        'skill_dribbling',
        'skill_curve',
        'skill_fk_accuracy',
        'skill_long_passing',
        'skill_ball_control',

        'movement_acceleration',
        'movement_sprint_speed',
        'movement_agility',
        'movement_reactions',
        'movement_balance',

        'power_shot_power',
        'power_jumping',
        'power_stamina',
        'power_strength',
        'power_long_shots',

        'mentality_aggression',
        'mentality_interceptions',
        'mentality_positioning',
        'mentality_vision',
        'mentality_penalties',
        'mentality_composure',

        'defending_marking',
        'defending_standing_tackle',
        'defending_sliding_tackle'
    ]

    columns += [
        'ls', 'st', 'rs',
        'lw', 'lf', 'cf', 'rf', 'rw',
        'lam', 'cam', 'ram',
        'lm', 'lcm', 'cm', 'rcm', 'rm',
        'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
        'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]
    # ls,st,rs,lw,lf,cf,rf,rw,
    # lam,cam,ram,lm,lcm,cm,rcm,rm,
    # lwb,ldm,cdm,rdm,rwb,lb,lcb,cb,rcb,rb

    return df[columns]


def convert_num_values(player_1_df, player_2_df):
    num_values = [
        'pace',
        'shooting',
        'passing',
        'dribbling',
        'defending',
        'physic',
        'attacking_crossing',
        'attacking_finishing',
        'attacking_heading_accuracy',
        'attacking_short_passing',
        'attacking_volleys',
        'skill_dribbling',
        'skill_curve',
        'skill_fk_accuracy',
        'skill_long_passing',
        'skill_ball_control',
        'movement_acceleration',
        'movement_sprint_speed',
        'movement_agility',
        'movement_reactions',
        'movement_balance',
        'power_shot_power',
        'power_jumping',
        'power_stamina',
        'power_strength',
        'power_long_shots',
        'mentality_aggression',
        'mentality_interceptions',
        'mentality_positioning',
        'mentality_vision',
        'mentality_penalties',
        'mentality_composure',
        'defending_marking',
        'defending_standing_tackle',
        'defending_sliding_tackle'
    ]

    num_values += [
        'ls', 'st', 'rs',
        'lw', 'lf', 'cf', 'rf', 'rw',
        'lam', 'cam', 'ram',
        'lm', 'lcm', 'cm', 'rcm', 'rm',
        'lwb', 'ldm', 'cdm', 'rdm', 'rwb',
        'lb', 'lcb', 'cb', 'rcb', 'rb'
    ]

    for v in num_values:
        # player1のデータの数値の整形
        value = player_1_df[v].values.astype(str)[0]
        value = shaping_num(str(value))

        # player_1_df[v] = float(value) * 0.01
        player_1_df[v] = float(value)

        # player2のデータの数値の整形
        value = player_2_df[v].values.astype(str)[0]
        value = shaping_num(str(value))

        # player_2_df[v] = float(value) * 0.01
        player_2_df[v] = float(value)

    return player_1_df, player_2_df


def convert_traits(player_1_df, player_2_df):

    # 選手特性関連の処理
    traits_list = [
        'Backs Into Player',  # FIFA 18だけの項目
        'Bicycle Kicks',
        'Chip Shot',
        'Dives Into Tackles',
        'Early Crosser',
        'Fancy Passes',
        'Finesse Shot',
        'Flair',
        'Giant Throw-In',
        'GK Cautious With Crosses',
        'GK Comes For Crosses',
        'GK Flat Kick',
        'GK Long Thrower',
        'GK Save With Foot',
        'Injury Prone',
        'Leadership',
        'Long Passer',
        'Long Shot Taker',
        'Long Throw-In',
        'One Club Player',
        'Outside Foot Shot',
        'Play Maker',
        'Power Header',
        'Rushes Out Of Goal',
        'Second Wind',
        'Set Play Specialist',
        'Solid Player',
        'Speed Dribbler',
        'Swerve',
        'Takes Powerful Driven Free Kicks',
        'Team Player',
        'Technical Dribbler'
    ]

    player_1_df_player_traits = player_1_df['player_traits']
    player_2_df_player_traits = player_2_df['player_traits']

    player_1_df = player_1_df.drop(columns='player_traits')
    player_2_df = player_2_df.drop(columns='player_traits')

    for trait in traits_list:
        trait_value = 0
        for p_trait in player_1_df_player_traits.values[0].split(','):
            if trait in p_trait:
                trait_value = 1
                break
        player_1_df[trait] = trait_value

        trait_value = 0
        for p_trait in player_2_df_player_traits.values[0].split(','):
            if trait in p_trait:
                trait_value = 1
                break
        player_2_df[trait] = trait_value

    return player_1_df, player_2_df


def players_comparison(player_1, player_2):
    df = pd.read_csv('data/players_18.csv')

    player_1_df = df.query('sofifa_id == {}'.format(player_1))
    player_2_df = df.query('sofifa_id == {}'.format(player_2))
    # david_silva = df.query('sofifa_id == 189881')

    player_1_df = need_columns(player_1_df)
    player_2_df = need_columns(player_2_df)

    # num_valuesの変換処理
    player_1_df, player_2_df = convert_num_values(player_1_df, player_2_df)

    # 選手特性関連の処理
    player_1_df, player_2_df = convert_traits(player_1_df, player_2_df)

    # 選手タグ関連の処理
    player_1_df = player_1_df.drop(columns='player_tags')
    player_2_df = player_2_df.drop(columns='player_tags')

    # 利き足の変換
    player_1_df = convert_preferred_foot(player_1_df)
    player_2_df = convert_preferred_foot(player_2_df)

    # 攻撃/守備の優先度の変換
    player_1_df = convert_work_rate(player_1_df)
    player_2_df = convert_work_rate(player_2_df)

    # print(player_1_df.values[0])
    # print(player_2_df.values)
    cos = similarity(player_1_df, player_2_df)
    print('cos', cos)

    # カラムの表示
    # columns_views(player_1_df, player_2_df)


shinji_kagawa = 189358
david_silva = 178088
# david_silva = 41
# 香川真司 : 189358
# 本田圭佑 : 186581
# 清武弘嗣 : 210126
# イニエスタ: 41
# スモーリング : 189881
# セルヒオ・ラモス : 155862
# マリオ・ゲッツェ : 192318
# ユリアン・ヴァイグル : 222028
# ファン・マタ : 178088
# イスコ : 197781
# ダビド・シルバ : 168542
# マルク・バルトラ : 198141
# ロメル・ルカク : 192505
# デブルイネ : 192985
# モドリッチ : 177003
# クロース : 182521
# ラキティッチ : 168651
# ウサマ・デンベレ : 231443
# リオネル・メッシ : 158023
# フンメルス : 178603
# ピケ: 152729
# ボアテング : 183907
# メスト・エジル : 176635
# マルコ・ロイス : 188350
# イヴァン・ペリシッチ : 181458
# トーマス・ミュラー : 189596
# オスカル : 188152
# ヤルモレンコ : 194794
# エデン・アザール : 183277
# ネイマール : 190871
# ロッベン : 9014
# サラー : 209331
# ハリー・ケイン : 202126
# ムバッペ : 231747
# グリーズマン : 194765
# ジェラール・ピケ : 152729


players_comparison(shinji_kagawa, david_silva)

# columns_views(shinji_kagawa, david_silva)

# Weak Foot(逆足)
# https://www.fifplay.com/encyclopedia/weak-foot/

# Work Rate(作業率)
# https://www.fifplay.com/encyclopedia/work-rate/

# ユークリッド距離 vs コサイン類似度
# https://enjoyworks.jp/tech-blog/2242

import requests
from bs4 import BeautifulSoup
import jinja2
import re


class Chara:
    name = ''
    job = ''
    hp = 0
    mp = 0
    str = 0
    end = 0
    dex = 0
    agi = 0
    mag = 0
    killer = ""
    counter_hp = ""
    skills = ""
    passive_skills = ""


class HtmlParser:
    def __init__(self, text):
        self.soup = BeautifulSoup(text, 'html.parser')
        self.soup_ptr = self.soup.find("div", class_='toc')

    def get_next_div(self):
        found = False
        while not found:
            self.soup_ptr = self.soup_ptr.find_next_sibling("div", class_='basic')
            if self.soup_ptr.find("table") is not None:
                found = True
        return self.soup_ptr


def parse_effs(effs_str):
    effs = []
    if "カウンター率" in effs_str:
        effs.append("Ability.counter_rate")
    if "ペネトレーション率" in effs_str:
        effs.append("Ability.pene_rate")
    if "必殺技ゲージ" in effs_str:
        effs.append("Ability.energy_bar")
    if "クリティカル率" in effs_str:
        effs.append("Ability.crit_rate")
    if "ガード率" in effs_str:
        effs.append("Ability.guard_rate")

    if "カウンター発生" in effs_str:
        effs.append("SuccessUp.counter")
    if "ペネトレーション発生" in effs_str:
        effs.append("SuccessUp.pene")
    if "クリティカル発生" in effs_str:
        effs.append("SuccessUp.crit")
    if "ガード発生" in effs_str:
        effs.append("SuccessUp.guard")

    if "力と魔力" in effs_str:
        effs.append("Ability.str")
        effs.append("Ability.mag")
    elif "力と" in effs_str:
        effs.append("Ability.str")
    elif "力、魔力" in effs_str:
        effs.append("Ability.str")
        effs.append("Ability.mag")
    elif "魔力" in effs_str:
        effs.append("Ability.mag")
    elif "の力" in effs_str:
        effs.append("Ability.str")
    elif "、力" in effs_str:
        effs.append("Ability.str")
    elif effs_str.startswith("力"):
        effs.append("Ability.str")

    if "敏捷" in effs_str:
        effs.append("Ability.agi")
    if "器用" in effs_str:
        effs.append("Ability.dex")
    if "耐久" in effs_str:
        effs.append("Ability.end")

    if "火属性耐性" in effs_str:
        effs.append("Endurance.fire")
    if "地属性耐性" in effs_str:
        effs.append("Endurance.earth")
    if "風属性耐性" in effs_str:
        effs.append("Endurance.wind")
    if "水属性耐性" in effs_str:
        effs.append("Endurance.ice")
    if "雷属性耐性" in effs_str:
        effs.append("Endurance.thunder")
    if "光属性耐性" in effs_str:
        effs.append("Endurance.light")
    if "闇属性耐性" in effs_str:
        effs.append("Endurance.dark")

    if "物理耐性" in effs_str:
        effs.append("Endurance.phy")
    if "魔法耐性" in effs_str:
        effs.append("Endurance.mag")

    if "全体攻撃ダメージ" in effs_str:
        effs.append("Endurance.foes")
    if "単体攻撃ダメージ" in effs_str:
        effs.append("Endurance.foe")

    if "火属性攻撃" in effs_str:
        effs.append("Damage.fire")
    if "地属性攻撃" in effs_str:
        effs.append("Damage.earth")
    if "風属性攻撃" in effs_str:
        effs.append("Damage.wind")
    if "水属性攻撃" in effs_str:
        effs.append("Damage.ice")
    if "雷属性攻撃" in effs_str:
        effs.append("Damage.thunder")
    if "光属性攻撃" in effs_str:
        effs.append("Damage.light")
    if "闇属性攻撃" in effs_str:
        effs.append("Damage.dark")

    if "全体攻撃被ダメージ" in effs_str:
        effs.append("Endurance.foes")
    if "単体攻撃被ダメージ" in effs_str:
        effs.append("Endurance.foe")

    if "HP" in effs_str or "ＨＰ" in effs_str:
        effs.append("Recover.hp_turn")
    if "MP" in effs_str or "ＭＰ" in effs_str:
        effs.append("Recover.mp_turn")

    return effs


def gen_eff_str(effs, scope, val_for_eff=None, turn=None):
    eff_enums = []
    for e in effs:
        if turn and val_for_eff:
            eff_enums.append(f"Effect({scope}, {e}, {val_for_eff}, {turn})")
        elif val_for_eff:
            eff_enums.append(f"Effect({scope}, {e}, {val_for_eff})")
        else:
            eff_enums.append(f"Effect({scope}, {e}, 0)")
    ret = ", ".join(eff_enums)
    return ret


def parse_turns(text):
    m = re.match(r".+(\d)ターンの間.+", text, re.UNICODE)
    if m is None:
        return None
    turn = m.group(1)
    return turn


def parse_scope(scope_str):
    if "敵全体" in scope_str:
        scope = "Scope.foes"
    elif "敵単体" in scope_str:
        scope = "Scope.foe"
    elif "味方全体" in scope_str:
        scope = "Scope.my_team"
    elif "自分" in scope_str:
        scope = "Scope.my_self"
    else:
        raise ValueError
    return scope


def parse_atk(text):
    scope = parse_scope(text)

    if "超強威力" in text:
        power = "Power.ultra"
    elif "超威力" in text:
        power = "Power.super"
    elif "強威力" in text:
        power = "Power.high"
    elif "中威力" in text:
        power = "Power.mid"
    elif "弱威力" in text:
        power = "Power.low"
    else:
        raise ValueError

    m = re.match(r".+(\w)属性(\w\w)攻撃.+", text, re.UNICODE)
    attr = m.group(1)
    phy_mag = m.group(2)

    if attr == "火":
        attr_dmg = "Damage.fire"
    elif attr == "地":
        attr_dmg = "Damage.earth"
    elif attr == "風":
        attr_dmg = "Damage.wind"
    elif attr == "水":
        attr_dmg = "Damage.ice"
    elif attr == "雷":
        attr_dmg = "Damage.thunder"
    elif attr == "光":
        attr_dmg = "Damage.light"
    elif attr == "闇":
        attr_dmg = "Damage.dark"
    else:
        raise ValueError

    if phy_mag == "物理":
        atk = "Attack.phy"
    elif phy_mag == "魔法":
        atk = "Attack.mag"
    else:
        raise ValueError

    temp_boost = ""

    if "技発動時のみ力を上昇" in text or "技発動時のみ魔力を上昇" in text:
        temp_boost = "temp_boost=True, "

    boost_by_buff = ""
    m = re.match(r".*自分の(\w+)上昇効果1つにつき、この技の威力が(\d+)[％%]上昇.*", text, re.UNICODE)
    if m is not None:
        up_val = int(m.group(2))
        up_val /= 100
        up_indexes = m.group(1)
        effs = parse_effs(up_indexes)
        enum_str = gen_eff_str(effs, "Scope.my_self", up_val)
        boost_by_buff = f'boost_by_buff=[{enum_str}],'

    atk_str = f"{scope}, {power}, {attr_dmg}, {atk}, {temp_boost} {boost_by_buff}"
    return atk_str


def parse_debuff(text, turn):
    m = re.match(r".*(敵単体|敵全体)(.+?)を?(\d+)[％%]減少.*", text, re.UNICODE)
    if m is None:
        m = re.match(r".+(敵単体|敵全体)(.+被ダメージ).*を(\d+)[％%]増加.*", text, re.UNICODE)
        if m is None:
            return None

    scope = parse_scope(m.group(1))

    effs_str = m.group(2)
    effs = parse_effs(effs_str)

    down_val = int(m.group(3))
    down_val /= 100

    enum_str = gen_eff_str(effs, scope, down_val, turn)
    return enum_str


def parse_recover_hp(text, turn):
    m = re.match(r".*(\d+)ターンの間、(味方全体|自分)に(\d+)[％%]の(HP|ＨＰ)治癒付与.*", text, re.UNICODE)
    if m:
        turn = m.group(1)
        scope = parse_scope(m.group(2))
        up_val = int(m.group(3)) / 100
        return f"Effect({scope}, Recover.hp_turn, {up_val}, {turn})"

    m = re.match(r".*味方全体に\w+の(HP|ＨＰ)回復技.*", text, re.UNICODE)
    if m:
        return f"Effect(Scope.my_team, Recover.hp_imm, 0.8)"

    m = re.match(r".*味方全体の(HP|ＨＰ)を(\d+)[％%]回復.*", text, re.UNICODE)
    if m:
        up_val = int(m.group(2)) / 100
        return f"Effect(Scope.my_team, Recover.hp_imm, {up_val})"


def parse_recover_mp(text, turn):
    m = re.match(r".*(\d+)ターンの間、(味方全体|自分)に(\d+)[％%]の(MP|ＭＰ)回復.*", text, re.UNICODE)
    if m:
        turn = m.group(1)
        scope = parse_scope(m.group(2))
        up_val = int(m.group(3)) / 100
        return f"Effect({scope}, Recover.mp_turn, {up_val}, {turn})"

    m = re.match(r".*味方全体の(MP|ＭＰ)を(\d+)[％%]回復.*", text, re.UNICODE)
    if m:
        up_val = int(m.group(2)) / 100
        return f"Effect(Scope.my_team, Recover.mp_imm, {up_val})"

    m = re.match(r".*自分の(MP|ＭＰ)を(\d+)[％%]回復.*", text, re.UNICODE)
    if m:
        up_val = int(m.group(2)) / 100
        return f"Effect(Scope.my_self, Recover.mp_imm, {up_val})"


def parse_buff(text, turn):
    m = re.match(r".*(味方全体|自分)(.+)を(\d+)[％%](上昇|軽減).*", text, re.UNICODE)
    if m is None:
        return None

    scope = parse_scope(m.group(1))

    effs_str = m.group(2)
    effs = parse_effs(effs_str)

    up_val = int(m.group(3))
    up_val /= 100

    enum_str = gen_eff_str(effs, scope, up_val, turn)
    return enum_str


def parse_passive_buff(text):
    ret_effs = []

    scope = "Scope.my_self"

    m = re.match(r"(.+)が(\d+)[％%]上昇.*", text, re.UNICODE)
    if m:
        effs_str = m.group(1)
        effs = parse_effs(effs_str)
        effs = [e for e in effs if "Ability" not in e]

        up_val = int(m.group(2))
        up_val /= 100
        enum_str = gen_eff_str(effs, scope, up_val)
        ret_effs.append(enum_str)

    m = re.match(r".*毎ターン(.+)が(\d+)[％%]回復.*", text, re.UNICODE)
    if m:
        effs_str = m.group(1)
        effs = parse_effs(effs_str)

        up_val = int(m.group(2))
        up_val /= 100
        enum_str = gen_eff_str(effs, scope, up_val)
        ret_effs.append(enum_str)

    return ", ".join(ret_effs)


def parse_adj_buff(text):

    ret_effs = []

    m = re.match(r".*(敵単体|敵全体)の(.+)上昇効果を解除.*", text, re.UNICODE)
    if m:
        scope = parse_scope(m.group(1))

        effs_str = m.group(2)
        effs = parse_effs(effs_str)

        for e in effs:
            enum_str = f"Effect({scope}, AdjBuff.clear_buff, 0, 0, {e})"
            ret_effs.append(enum_str)

    m = re.match(r".*(自分|味方全体)のステイタス上昇効果.*(\d+)ターン延長", text, re.UNICODE)
    if m:
        scope = parse_scope(m.group(1))
        turn_val = m.group(2)
        enum_str = f"Effect({scope}, AdjBuff.extend_buff, {turn_val}, 0)"
        ret_effs.append(enum_str)

    m = re.match(r".*(敵単体|敵全体)のステイタス減少効果.*(\d+)ターン延長", text, re.UNICODE)
    if m:
        scope = parse_scope(m.group(1))
        turn_val = m.group(2)
        enum_str = f"Effect({scope}, AdjBuff.extend_debuff, {turn_val}, 0)"
        ret_effs.append(enum_str)

    m = re.match(r".*(敵単体|敵全体)のステイタス上昇効果.*(\d+)ターン減少", text, re.UNICODE)
    if m:
        scope = parse_scope(m.group(1))
        turn_val = m.group(2)
        enum_str = f"Effect({scope}, AdjBuff.shorten_buff, {turn_val}, 0)"
        ret_effs.append(enum_str)

    return ", ".join(ret_effs)


def gen_skill_str(text, is_special=False):
    text = text.replace("\n", "")
    text = text.replace("・", "")
    print(text)
    atk_str = ""
    mp_str = ""
    special_str = ""
    is_fast_str = ""

    if "すばやく" in text:
        is_fast_str = "is_fast=True, "

    if "攻撃。" in text:
        # has attack
        atk_str = parse_atk(text)

    turn = parse_turns(text)

    texts = []
    texts_tmp = text.split("し、")
    scope_guessing = None
    for txt in texts_tmp:
        if "味方全体" in txt:
            scope_guessing = "味方全体"
        elif "自分" in txt:
            scope_guessing = "自分"
        if "自分" not in txt and "味方全体" not in txt:
            if scope_guessing:
                txt = scope_guessing + txt
        texts.extend(txt.split("さらに"))
    buffs_eff = []
    debuffs_eff = []
    adj_buffs_eff = []
    for t in texts:
        b = parse_buff(t, turn)
        d = parse_debuff(t, turn)
        a = parse_adj_buff(t)
        rhp = parse_recover_hp(t, turn)
        rmp = parse_recover_mp(t, turn)
        if b:
            buffs_eff.append(b)
        if rhp:
            buffs_eff.append(rhp)
        if rmp:
            buffs_eff.append(rmp)
        if d:
            debuffs_eff.append(d)
        if a:
            adj_buffs_eff.append(a)
    buffs_str = f"buffs=[{', '.join(buffs_eff)}],"
    debuffs_str = f"debuffs=[{', '.join(debuffs_eff)}],"
    adj_buffs_str = f"adj_buffs=[{', '.join(adj_buffs_eff)}],"
    # print(buffs_str)
    # print(debuffs_str)

    if is_special:
        special_str = "is_special=True,"
    else:
        m = re.match(r".+（MP:(\d+)）.*", text, re.UNICODE)
        if m:
            mp = m.group(1)
            mp_str = f"mp={mp},"

    skill_dec_str = f"Skill({is_fast_str} {atk_str} {special_str} {mp_str} {buffs_str} {debuffs_str} {adj_buffs_str})"
    return skill_dec_str


def gen_passive_skill_str(text):
    b = parse_passive_buff(text)
    print(text)
    return b


def gen_counter_hp_str(text):
    m = re.match(r".*】.*カウンター発生時.*通常攻撃.*(HP|ＨＰ)回復", text, re.UNICODE)
    if m:
        return "counter_hp=True,"
    return ""


def gen_killer_str(text):
    m = re.match(r".*】(\w+)の敵を攻撃.*(\d+)[％%]上昇.*", text, re.UNICODE)
    if m:
        killer = m.group(1)
        if killer == "猛牛系":
            return "killer=Killer.bull, "
        elif killer == "巨人系":
            return "killer=Killer.giant, "
        elif killer == "魔獣系":
            return "killer=Killer.beast, "
        elif killer == "精霊系":
            return "killer=Killer.fairy, "
        elif killer == "植物系":
            return "killer=Killer.plant, "
        elif killer == "昆虫系":
            return "killer=Killer.bug, "
        elif killer == "堅鉱系":
            return "killer=Killer.rock, "
        elif killer == "蠕獣系":
            return "killer=Killer.worm, "
        elif killer == "竜系":
            return "killer=Killer.dragon, "
        elif killer == "水棲系":
            return "killer=Killer.aquatic, "
        elif killer == "妖鬼系":
            return "killer=Killer.orge, "
        elif killer == "幽魔系":
            return "killer=Killer.undead, "
        else:
            raise ValueError

    return ""

def parsing_chara(html_text):
    parser = HtmlParser(html_text)

    chara = Chara()

    basics_table = parser.get_next_div()
    for tr in basics_table.table.find_all('tr'):
        col = tr.th.text
        val = tr.td.text
        if col == "名称":
            chara.name = val
        if col == "カテゴリ":
            if val == "冒険者":
                chara.job = "Adventurer"
            elif val == "アシスト":
                chara.job = "Assist"

    limit_break_status_table = parser.get_next_div()
    while "最大値" not in limit_break_status_table.text:
        limit_break_status_table = parser.get_next_div()

    #limit_break_status_table = parser.get_next_div()
    for tr in limit_break_status_table.table.find_all('tr'):
        if tr.td is None:
            continue
        col = tr.td
        val = col.find_next_sibling()
        print(col.text, val.text)
        if col.text == "HP":
            chara.hp = int(val.text)
        if col.text == "MP":
            chara.mp = int(val.text)
        if col.text == "物攻":
            chara.str = int(val.text.split("(")[0])
        if col.text == "魔攻":
            chara.mag = int(val.text.split("(")[0])
        if col.text == "防御":
            chara.end = int(val.text.split("(")[0])
        if col.text == "器用":
            chara.dex = int(val.text.split("(")[0])
        if col.text == "敏捷":
            chara.agi = int(val.text.split("(")[0])

    all_skills = []
    all_passive_skills = []

    if chara.job == "Adventurer":
        status_table_no_used = parser.get_next_div()
        special_skill = parser.get_next_div()
        special_skill_dec_str = gen_skill_str(special_skill.text, True)

    skills = parser.get_next_div()
    for s in skills.find_all("td"):
        skill_str = gen_skill_str(s.text)
        all_skills.append(skill_str)

    if chara.job == "Adventurer":
        all_skills.append(special_skill_dec_str)

    concated_skills = ',\n        '.join(all_skills)
    chara.skills = f"skills=[{concated_skills}],"

    if chara.job == "Adventurer":
        passive_skills = parser.get_next_div()
        for s in passive_skills.find_all("td"):
            passive_skill_str = gen_passive_skill_str(s.text)
            if passive_skill_str:
                all_passive_skills.append(passive_skill_str)
            if chara.killer == "":
                chara.killer = gen_killer_str(s.text)
            if chara.counter_hp == "":
                chara.counter_hp = gen_counter_hp_str(s.text)

        concated_passive_skills = ',\n        '.join(all_passive_skills)
        chara.passive_skills = f"passive_skills=[Skill(buffs=[{concated_passive_skills}])],"

    template = jinja2.Template("""
{{chara.job}}("{{chara.name}}", {{chara.hp}}, {{chara.mp}},
    {{chara.str}}, {{chara.end}}, {{chara.dex}}, {{chara.agi}}, {{chara.mag}},
    {{chara.skills}}
    {{chara.passive_skills}}
    {{chara.killer}}
    {{chara.counter_hp}}
),
    """)

    if chara.job == "Adventurer":
        out = template.render(chara=chara)
        print(out)
    else:
        for i, s in enumerate(all_skills):
            print("======================================================")
            if i == 0:
                continue
            elif i == 1:
                print("LV 60~76:")
            elif i == 2:
                print("LV 80:")
            else:
                raise
            chara.skills = f"skill={s}"
            out = template.render(chara=chara)
            print(out)


def parsing_chara_from_web(http_url):
    r = requests.get(http_url)
    html_text = r.text
    parsing_chara(html_text)


if __name__ == '__main__':
    with open('tmp.html', 'r', encoding="utf-8") as f:
        html_text_to_test = f.read()
    parsing_chara(html_text_to_test)

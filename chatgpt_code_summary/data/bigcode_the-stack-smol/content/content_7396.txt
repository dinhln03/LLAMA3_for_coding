import sys
import json

from rule_gens import RulesForGeneration
from generate_text import generating_player_text_from_templates, generating_team_text_from_templates

from transformers import GPT2Tokenizer

print("Constructing main file ....")
test_preds = []
js = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
rfg  = RulesForGeneration()
print("Constructed!!\n\n")

for game_idx in range(len(js)):

    # if game_idx == 0:
    team_stat = generating_team_text_from_templates(js, game_idx, tokenizer)
    player_stat = generating_player_text_from_templates(js, game_idx, tokenizer)

    sol = [rfg.generate_defeat_sentence(js[game_idx]).strip()]

    for k, v in team_stat.items():
        sol.append(v.strip())
    for k, v in player_stat.items():
        sol.append(v.strip())

    sol.append(rfg.generate_next_game_sentence(js[game_idx]).strip())

    final_sol = ' '.join(sol)

    if game_idx % 10 == 0:
        print()
        print(game_idx, final_sol)

    test_preds.append(final_sol)

with open(f'./output/{sys.argv[1]}', 'w') as f:
    f.write('\n'.join(test_preds))


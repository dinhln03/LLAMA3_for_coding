from pdb import set_trace as TT
import numpy as np
import scipy
from scipy.spatial import ConvexHull
import skimage
from skimage.morphology import disk
import skbio

global trg_image
trg_image = None

def diversity_calc(config):
   div_calc_name = config.FITNESS_METRIC
   return get_div_calc(div_calc_name)

def get_div_calc(div_calc_name):
   if div_calc_name == 'L2':
      calc_diversity = calc_diversity_l2
   elif div_calc_name == 'InvL2':
      calc_diversity = calc_homogeneity_l2
   elif div_calc_name == 'Differential':
      calc_diversity = calc_differential_entropy
   elif div_calc_name == 'Discrete':
      calc_diversity = calc_discrete_entropy_2
   elif div_calc_name == 'Hull':
      calc_diversity = calc_convex_hull
   elif div_calc_name == 'Sum':
      calc_diversity = sum_experience
   elif div_calc_name == 'Lifespans':  # or config.FITNESS_METRIC == 'ALP':
      calc_diversity = sum_lifespans
   elif div_calc_name == 'Lifetimes':
       calc_diversity = calc_mean_lifetime
   elif div_calc_name == 'Actions':
       calc_diversity = calc_mean_actions_matched
   elif div_calc_name == 'MapTest':
       calc_diversity = calc_local_map_entropy
   elif div_calc_name == 'MapTestText':
       calc_diversity = ham_text
       get_trg_image()
   elif div_calc_name == 'y_deltas':
       calc_diversity = calc_y_deltas
   elif div_calc_name == 'Scores' or config.FITNESS_METRIC == 'ALP':
       calc_diversity = calc_scores
   else:
       raise Exception('Unsupported fitness function: {}'.format(config.FITNESS_METRIC))
   return calc_diversity

def get_trg_image():
    from PIL import Image, ImageDraw, ImageFont
    font_size = 15
    try:
       font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
       try:
          font = ImageFont.truetype("LiberationMono-Regular.ttf", font_size)
       except OSError:
          font = ImageFont.truetype("SFNSMono.ttf", 32)
    global trg_image
    trg_image = Image.new(mode = "RGB", size=(50, 50))
    draw = ImageDraw.Draw(trg_image)
    draw.text((1,1), "Evo", font=font, fill=(255,0,0))
    draw.text((1,15), "NMMO", font=font, fill=(255,0,0))
    draw.text((1,32), "¯\_(ツ)_/¯", font=font, fill=(255,0,0))
    trg_image.save("trg_img.png")
    trg_image = (np.array(trg_image)[:, :, 0] / 255 * 8).astype(np.uint8)

def ham_text(individual, config):
    if trg_image is None:
       get_trg_image()
    map_arr = individual.chromosome.map_arr[10:-10, 10:-10]
    return -(trg_image != map_arr).sum()

def calc_map_entropies(individual, config, verbose=False):
    glob_ent = calc_global_map_entropy(individual, config)
    loc_ent = calc_local_map_entropy(individual, config)
    if verbose:
       print('global entropy: {}\nlocal entropy: {}'.format(glob_ent, loc_ent))

    return [glob_ent[0], loc_ent]

def calc_global_map_entropy(individual, config):
    # FIXME: hack to ignore lava borders
    b = config.TERRAIN_BORDER
    map_arr = individual.chromosome.map_arr[b:-b, b:-b]
    ent = scipy.stats.entropy(np.bincount(map_arr.reshape(-1), minlength=individual.n_tiles))
    ent = ent * 100 / np.log(individual.n_tiles)

    return [ent]

def calc_local_map_entropy(individual, config):
    # FIXME: hack to ignore lava borders
    b = config.TERRAIN_BORDER
    map_arr = individual.chromosome.map_arr[b:-b, b:-b]
    local_ent = skimage.filters.rank.entropy(map_arr, disk(3))
    local_ent = local_ent.mean() * 100 / np.log2(individual.n_tiles)

    return local_ent.item()

def get_pop_stats(agent_stats, pop=None):
   # Get list of all populations for which we need stats
   pops = agent_stats[0].keys() if pop is None else [pop]
   # Get 1D array of agent stats
   stats = [stats_i[p] for p in pops for stats_i in agent_stats]
   if len(stats[0].shape) == 2:
      # then rows correspond to agents so we stack them vertically (concatenate along axis 1)
      return np.vstack(stats)
   elif len(stats[0].shape) == 1:
      # then each agent has a scalar value so we concatenate along axis 0
      return np.hstack(stats)
   raise Exception("Oy! Dafuk type o' agent data is this?")

def contract_by_lifespan(agent_stats, lifespans):
   '''Pull agents close to their mean according to how short-lived they were. For punishing abundance of premature death
   when rewarding diversity.'''
   weights = sigmoid_lifespan(lifespans)
   n_agents = lifespans.shape[0]
   mean_agent = agent_stats.mean(axis=0)
   mean_agents = np.repeat(mean_agent.reshape(1, mean_agent.shape[0]), n_agents, axis=0)
   agent_deltas = mean_agents - agent_stats
   agent_skills = agent_stats + (weights * agent_deltas.T).T

   return agent_skills

def expand_by_lifespan(agent_stats, lifespans):
   '''Push agents further from their mean according to how short-lived they were. For punishing abundance of premature
   death when rewarding homogeneity.'''
   weights = sigmoid_lifespan(lifespans)
   n_agents = lifespans.shape[0]
   mean_agent = agent_stats.mean(axis=0)
   mean_agents = np.repeat(mean_agent.reshape(1, mean_agent.shape[0]), n_agents, axis=0)
   agent_deltas = mean_agents - agent_stats
   # Displace agents by at most 100 units (otherwise we will not punish agents at all if they are already perfectly
   # homogenous, for example.
   agent_deltas = agent_deltas / np.linalg.norm(agent_deltas) * 100
   agent_skills = agent_stats - (weights * agent_deltas.T).T

   return agent_skills

def calc_scores(agent_stats, skill_headers=None, verbose=False):
    scores = np.hstack(agent_stats['scores'])
    if verbose:
        print('scores: {}'.format(scores))
    return np.mean(scores)

def calc_mean_actions_matched(agent_stats, skill_headers=None, verbose=False):
    actions_matched = np.hstack(agent_stats['actions_matched'])
    if verbose:
        print(actions_matched)
#       print(agent_stats['lifespans'])
    return np.mean(actions_matched)

def calc_y_deltas(agent_stats, skill_headers=None, verbose=False):
    y_deltas = np.hstack(agent_stats['y_deltas'])
    if verbose:
        print('y_deltas: {}'.format(y_deltas))
    return np.mean(y_deltas)

def calc_mean_lifetime(agent_stats, skill_headers=None, verbose=False, pop=None):
    lifetimes = get_pop_stats(agent_stats['lifespans'], pop)
    if len(lifetimes) != 0:
        lifetimes = np.hstack(lifetimes)
    else:
        lifetimes = [0]
    mean_lifetime = lifetimes.mean()

    return mean_lifetime

def sum_lifespans(agent_stats, skill_headers=None, n_policies=1, verbose=False, pop=None):
   lifespans = get_pop_stats(agent_stats['lifespans'], pop=pop)
   score = lifespans.mean()
   if verbose:
      print('Mean lifespan, pop {}: {}'.format(pop, score))

   return score

def sum_experience(agent_stats, skill_headers=None, verbose=False, pop=None):
   '''Simply take the sum of XP over skills and agents.'''
   # No need to weight by lifespan, since high lifespan is a prerequisite for high XP.
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   a_skills = np.vstack(agent_skills)
   a_lifespans = np.hstack(lifespans)
   n_agents, n_skills = a_skills.shape
   mean_xp = a_skills.sum() / (n_agents * n_skills)

   if verbose:
      print('skills')
      print(a_skills.T)
      print('lifespans')
      print(a_lifespans)
      print('mean xp:', mean_xp)
      print()

   return mean_xp

def sigmoid_lifespan(x):
   # This basically assumes max lifespan is at least 100. Larger max lifespans won't really be a problem since this
   # function converges to 1.
   res = 1 / (1 + np.exp(0.1*(-x+50)))

   return res

def calc_differential_entropy(agent_stats, skill_headers=None, verbose=False, infos={}, pop=None, punish_youth=True):
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)

   a_skills = agent_skills
   a_lifespans = lifespans
   assert a_skills.shape[0] == a_lifespans.shape[0]

   if verbose:
      print(skill_headers)
      print(a_skills.transpose())
      print(len(agent_skills), 'populations')
      print('lifespans')
      print(a_lifespans)

   if punish_youth:
      # Below is an alternative way of weighting by lifespan
      # weights = sigmoid_lifespan(a_lifespans)
      # mean = np.average(a_skills, axis=0, weights=weights)
      # cov = np.cov(a_skills,rowvar=0, aweights=weights)

      # Instead, we'll just contract as usual
      a_skills = contract_by_lifespan(a_skills, a_lifespans)
   mean = np.average(a_skills, axis=0)
   cov = np.cov(a_skills,rowvar=0)
   gaussian = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
   infos['gaussian'] = gaussian
   score = gaussian.entropy()

   if verbose:
      print('score:', score)

   return score


def calc_convex_hull(agent_stats, skill_headers=None, verbose=False, infos={}, pop=None, punish_youth=True):
   '''Calculate the diversity of a population of agents in skill-space by computing the volume inside the convex hull of
   the agents when treated as points in this space.'''
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   agent_skills = np.vstack(agent_skills)
   n_skills = agent_skills.shape[1]

   lifespans = np.hstack(lifespans)
   if verbose:
      print('skills:')
      print(agent_skills.transpose())
      print('lifespans:')
      print(lifespans)
      print(len(agent_stats['lifespans']), 'populations')
   if punish_youth:
      agent_skills = contract_by_lifespan(agent_skills, lifespans)
   if n_skills == 1:
      # Max distance, i.e. a 1D hull
      score = agent_skills.max() - agent_skills.mean()
   else:
      try:
          hull = ConvexHull(agent_skills, qhull_options='QJ')
          infos['hull'] = hull
          score = hull.volume
          score = score ** (1 / n_skills)
      except Exception as e:
          print(e)
          score = 0
      if verbose:
         print('score:', score)

   return score

def calc_discrete_entropy_2(agent_stats, skill_headers=None, verbose=False, pop=None, punish_youth=True):
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   agent_skills_0 = agent_skills= np.vstack(agent_skills)
   lifespans = np.hstack(lifespans)
   n_agents = lifespans.shape[0]
   if n_agents == 1:
       return -np.float('inf')
   n_skills = agent_skills.shape[1]
   if verbose:
       print('skills')
       print(agent_skills_0.transpose())
       print('lifespans')
       print(lifespans)
   agent_skills = np.where(agent_skills == 0, 0.0000001, agent_skills)
   if punish_youth:
      # Below is a v funky way of punishing by lifespan
      # weights = sigmoid_lifespan(lifespans)
      # # contract population toward mean according to lifespan
      # # mean experience level for each agent
      # mean_skill = agent_skills.mean(axis=1)
      # # mean skill vector of an agent
      # mean_agent = agent_skills.mean(axis=0)
      # assert mean_skill.shape[0] == n_agents
      # assert mean_agent.shape[0] == n_skills
      # mean_skills = np.repeat(mean_skill.reshape(mean_skill.shape[0], 1), n_skills, axis=1)
      # mean_agents = np.repeat(mean_agent.reshape(1, mean_agent.shape[0]), n_agents, axis=0)
      # agent_deltas = agent_skills - mean_agents
      # skill_deltas = agent_skills - mean_skills
      # a_skills_skills = mean_agents + (weights * agent_deltas.transpose()).transpose()
      # a_skills_agents = mean_skills + (weights * skill_deltas.transpose()).transpose()
      # div_agents = skbio.diversity.alpha_diversity('shannon', a_skills_agents).mean()
      # div_skills = skbio.diversity.alpha_diversity('shannon', a_skills_skills.transpose()).mean()

      # We'll just do the usual
      a_skills = contract_by_lifespan(agent_skills, lifespans)
   div_agents = skbio.diversity.alpha_diversity('shannon', a_skills).mean()
   div_skills = skbio.diversity.alpha_diversity('shannon', a_skills.transpose()).mean()

 # div_lifespans = skbio.diversity.alpha_diversity('shannon', lifespans)
   score = -(div_agents * div_skills)#/ div_lifespans#/ len(agent_skills)**2
   score = score#* 100  #/ (n_agents * n_skills)
   if verbose:
       print('Score:', score)

   return score


def calc_discrete_entropy(agent_stats, skill_headers=None, pop=None):
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   agent_skills_0 = np.vstack(agent_skills)
   agent_lifespans = np.hstack(lifespans)
   weights = sigmoid_lifespan(agent_lifespans)
   agent_skills = agent_skills_0.transpose() * weights
   agent_skills = agent_skills.transpose()
   BASE_VAL = 0.0001
   # split between skill and agent entropy
   n_skills = len(agent_skills[0])
   n_pop = len(agent_skills)
   agent_sums = [sum(skills) for skills in agent_skills]
   i = 0

   # ensure that we will not be dividing by zero when computing probabilities

   for a in agent_sums:
       if a == 0:
           agent_sums[i] = BASE_VAL * n_skills
       i += 1
   skill_sums = [0 for i in range(n_skills)]

   for i in range(n_skills):

       for a_skills in agent_skills:
           skill_sums[i] += a_skills[i]

       if skill_sums[i] == 0:
           skill_sums[i] = BASE_VAL * n_pop

   skill_ents = []

   for i in range(n_skills):
       skill_ent = 0

       for j in range(n_pop):

           a_skill = agent_skills[j][i]

           if a_skill == 0:
               a_skill = BASE_VAL
           p = a_skill / skill_sums[i]

           if p == 0:
               skill_ent += 0
           else:
               skill_ent += p * np.log(p)
       skill_ent = skill_ent / (n_pop)
       skill_ents.append(skill_ent)

   agent_ents = []

   for j in range(n_pop):
       agent_ent = 0

       for i in range(n_skills):

           a_skill = agent_skills[j][i]

           if a_skill == 0:
               a_skill = BASE_VAL
           p = a_skill / agent_sums[j]

           if p == 0:
               agent_ent += 0
           else:
               agent_ent += p * np.log(p)
       agent_ent = agent_ent / (n_skills)
       agent_ents.append(agent_ent)
   agent_score =  np.mean(agent_ents)
   skill_score =  np.mean(skill_ents)
#  score = (alpha * skill_score + (1 - alpha) * agent_score)
   score = -(skill_score * agent_score)
   score = score * 100#/ n_pop**2
   print('agent skills:\n{}\n{}'.format(skill_headers, np.array(agent_skills_0.transpose())))
   print('lifespans:\n{}'.format(lifespans))
#  print('skill_ents:\n{}\nskill_mean:\n{}\nagent_ents:\n{}\nagent_mean:{}\nscore:\n{}\n'.format(
#      np.array(skill_ents), skill_score, np.array(agent_ents), agent_score, score))
   print('score:\n{}'.format(score))

   return score

def calc_homogeneity_l2(agent_stats, skill_headers=None, verbose=False, pop=None, punish_youth=True):
   '''Use L2 distance to punish agents for having high mean pairwise distance. Optimal state is all agents at the same
   point in skill-space, with maximal lifespans.'''
   if 'skills' not in agent_stats:
      raise Exception('We should be including dead agents in this calculation, so we should get at least some skill '
                      'stats back here')
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   assert len(agent_skills) == len(lifespans)
   if punish_youth:
      agent_skills = expand_by_lifespan(agent_skills, lifespans)
   n_agents = agent_skills.shape[0]
   a = agent_skills
   b = a.reshape(n_agents, 1, a.shape[1])
   # https://stackoverflow.com/questions/43367001/how-to-calculate-euclidean-distance-between-pair-of-rows-of-a-numpy-array
   distances = np.sqrt(np.einsum('ijk, ijk->ij', a - b, a - b))
   score = np.sum(distances) / n_agents ** 2

   if verbose:
      #  print(skill_headers)
      print('agent skills:\n{}'.format(a.transpose()))
      print('lifespans:\n{}'.format(lifespans))
      print('score:\n{}\n'.format(
         score))

   return -score


def calc_diversity_l2(agent_stats, skill_headers=None, verbose=False, pop=None, punish_youth=False):
   if 'skills' not in agent_stats:
      return 0
   agent_skills = get_pop_stats(agent_stats['skills'], pop)
   lifespans = get_pop_stats(agent_stats['lifespans'], pop)
   assert len(agent_skills) == len(lifespans)
   if punish_youth:
      agent_skills = contract_by_lifespan(agent_skills, lifespans)
   n_agents = agent_skills.shape[0]
   a = agent_skills
   b = a.reshape(n_agents, 1, a.shape[1])
   # https://stackoverflow.com/questions/43367001/how-to-calculate-euclidean-distance-between-pair-of-rows-of-a-numpy-array
   distances = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
   score = np.sum(distances) / n_agents ** 2

   if verbose:
#  print(skill_headers)
       print('agent skills:\n{}'.format(a.transpose()))
       print('lifespans:\n{}'.format(lifespans))
       print('score:\n{}\n'.format(
       score))

   return score

DIV_CALCS = [(calc_diversity_l2, 'mean pairwise L2'), (calc_differential_entropy, 'differential entropy'), (calc_discrete_entropy_2, 'discrete entropy'), (calc_convex_hull, 'convex hull volume'), (sum_lifespans, 'lifespans')]

#!/usr/bin/python
'''
Gapfilling function that utilizes pFBA and flux sampling to find most
parsimonious additional reactions to achieve minimum flux through the objective
Author: Matthew Jenior
'''
import pandas
import math
import copy
import time
import random

# Using Cobrapy 0.13.0
import cobra
import cobra.test
from cobra.flux_analysis.sampling import OptGPSampler
from cobra.manipulation.delete import *
from cobra.flux_analysis.parsimonious import add_pfba
from cobra.medium import find_boundary_types
from cobra.util import solver as sutil

# pFBA gapfiller
def pfba_gapfill(model, reaction_bag, obj=None, obj_lb=10., obj_constraint=False,
                 iters=1, tasks=None, task_lb=0.05, 
                 add_exchanges=True, extracellular='e', cores=4):
    '''
    Function that utilizes iterations of pFBA solution with a universal reaction bag 
    in order to gapfill a model.
    
    Parameters
    ----------
    model : cobra.Model
        Model to be gapfilled
    reaction_bag : cobra.Model
        Reaction bag reference to use during gapfilling
    obj : string
        Reaction ID for objective function in model to be gapfilled.
    obj_lb : float
        Lower bound for objective function
    obj_constraint : bool
        Sets objective as contstraint which must be maximized
    tasks : list or None
        List of reactions IDs (strings) of metabolic tasks 
        to set a minimum lower bound for
    task_lb : float
        Lower bound for any metabolic tasks
    iters : int
        Number of gapfilling rounds. Unique reactions from each round are 
        saved and the union is added simulatneously to the model
    add_exchanges : bool
        Identifies extracellular metabolites added during gapfilling that
        are not associated with exchange reactions and creates them
    extracellular : string
        Label for extracellular compartment of model
    cores : int
        Number of processors to utilize during flux sampling
    '''
    start_time = time.time()
    
    # Save some basic network info for downstream membership testing
    orig_rxn_ids = set([str(x.id) for x in model.reactions])
    orig_cpd_ids = set([str(y.id) for y in model.metabolites])
    univ_rxn_ids = set([str(z.id) for z in reaction_bag.reactions])
    
    # Find overlap in model and reaction bag
    overlap_rxn_ids = univ_rxn_ids.intersection(orig_rxn_ids)
    
    # Get model objective reaction ID
    if obj == None:
        obj = get_objective(model)
    else:
        obj = obj
    
    # Modify universal reaction bag
    new_rxn_ids = set()
    print('Creating universal model...')
    with reaction_bag as universal:

        # Remove overlapping reactions from universal bag, and reset objective if needed
        for rxn in overlap_rxn_ids: 
            universal.reactions.get_by_id(rxn).remove_from_model()
        
        # Set objective in universal if told by user
        # Made constraint as fraction of minimum in next step
        if obj_constraint:
            universal.add_reactions([model.reactions.get_by_id(obj)])
            universal.objective = obj
            orig_rxn_ids.remove(obj)
            orig_rxns = []
            for rxn in orig_rxn_ids: 
                orig_rxns.append(copy.deepcopy(model.reactions.get_by_id(rxn)))
        else:
            orig_rxns = list(copy.deepcopy(model.reactions))
            
        # Add pFBA to universal model and add model reactions
        add_pfba(universal)
        #universal = copy.deepcopy(universal) # reset solver
        universal.add_reactions(orig_rxns)
        
        # If previous objective not set as constraint, set minimum lower bound
        if not obj_constraint: 
            universal.reactions.get_by_id(obj).lower_bound = obj_lb
    
        # Set metabolic tasks that must carry flux in gapfilled solution
        if tasks != None:
            for task in tasks:
                try:                
                    universal.reactions.get_by_id(task).lower_bound = task_lb
                except:
                    print(task + 'not found in model. Ignoring.')
                    continue

        # Run FBA and save solution
        print('Optimizing model with combined reactions...')
        solution = universal.optimize()

        if iters > 1:
            print('Generating flux sampling object...')
            sutil.fix_objective_as_constraint(universal, fraction=0.99)
            optgp_object = OptGPSampler(universal, processes=cores)
        
            # Assess the sampled flux distributions
            print('Sampling ' + str(iters) + ' flux distributions...')
            flux_samples = optgp_object.sample(iters)
            rxns = list(flux_samples.columns)
            for distribution in flux_samples.iterrows():
                for flux in range(0, len(list(distribution[1]))):
                    if abs(list(distribution[1])[flux]) > 1e-6:
                        new_rxn_ids |= set([rxns[flux]]).difference(orig_rxn_ids)
        else:
            rxns = list(solution.fluxes.index)
            fluxes = list(solution.fluxes)
            for flux in range(0, len(fluxes)):
                if abs(fluxes[flux]) > 1e-6:
                    new_rxn_ids |= set([rxns[flux]])
    
    # Screen new reaction IDs
    if obj in new_rxn_ids: new_rxn_ids.remove(obj)
    for rxn in orig_rxn_ids:
        try:
            new_rxn_ids.remove(rxn)
        except:
            continue
    
    # Get reactions and metabolites to be added to the model
    print('Gapfilling model...')
    new_rxns = copy.deepcopy([reaction_bag.reactions.get_by_id(rxn) for rxn in new_rxn_ids])
    new_cpd_ids = set()
    for rxn in new_rxns: new_cpd_ids |= set([str(x.id) for x in list(rxn.metabolites)])
    new_cpd_ids = new_cpd_ids.difference(orig_cpd_ids)
    new_cpds = copy.deepcopy([reaction_bag.metabolites.get_by_id(cpd) for cpd in new_cpd_ids])
    
    # Copy model and gapfill 
    new_model = copy.deepcopy(model)
    new_model.add_metabolites(new_cpds)
    new_model.add_reactions(new_rxns)
    
    # Identify extracellular metabolites with no exchanges
    if add_exchanges == True:
        new_exchanges = extend_exchanges(new_model, new_cpd_ids, extracellular)
        if len(new_exchanges) > 0: new_rxn_ids |= new_exchanges
    
    duration = int(round(time.time() - start_time))
    print('Took ' + str(duration) + ' seconds to gapfill ' + str(len(new_rxn_ids)) + \
          ' reactions and ' + str(len(new_cpd_ids)) + ' metabolites.') 
    
    new_obj_val = new_model.slim_optimize()
    if new_obj_val > 1e-6:
        print('Gapfilled model objective now carries flux (' + str(new_obj_val) + ').')
    else:
        print('Gapfilled model objective still does not carry flux.')
    
    return new_model


# Adds missing exchanges for extracellulart metbaolites
def extend_exchanges(model, cpd_ids, ex):
    
    model_exchanges = set(find_boundary_types(model, 'exchange', external_compartment=ex))
    new_ex_ids = set()
    
    for cpd in cpd_ids:
        cpd = model.metabolites.get_by_id(cpd)
        if str(cpd.compartment) != ex:
            continue
        else:
            if bool(set(cpd.reactions) & model_exchanges) == False:
                try:
                    new_id = 'EX_' + cpd.id
                    model.add_boundary(cpd, type='exchange', reaction_id=new_id, lb=-1000.0, ub=1000.0)
                    new_ex_ids |= set([new_id])
                except ValueError:
                    pass

    return new_ex_ids


# Returns the reaction ID of the objective reaction
def get_objective(model):
    
    if len(list(model.objective.variables)) == 0:
        raise IndexError('Model has no objective set.')
    
    expression = str(model.objective.expression).split()
    if 'reverse' in expression[0]:
        obj_id = expression[2].split('*')[-1]
    else:
        obj_id = expression[0].split('*')[-1]
            
    return obj_id


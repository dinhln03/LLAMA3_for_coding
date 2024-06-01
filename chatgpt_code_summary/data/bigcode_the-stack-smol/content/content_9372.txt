"""
Copyright 2019 by Adam Lewicki
This file is part of the Game Theory library,
and is released under the "MIT License Agreement". Please see the LICENSE
file that should have been included as part of this package.
"""
import json


# ======================================================================================================================
# game tree object
class GameTree:
    # ---------------------------------- OBJECT PROPERTIES -------------------------------------------------------------
    # procedure of printing object properties
    def __repr__(self):
        """ return tree as JSON serialized dictionary """
        return self.pretty_print(self.__dict__)

    @staticmethod
    def pretty_print(dictionary: dict):
        """ return pretty printed dictionary as JSON serialized object """
        return json.dumps(dictionary, indent=4)

    # initialize object
    def __init__(self, nodes: dict = None, groups: dict = None, leafs: list = None, players_list: list = None):
        """
        GameTree class used to represent game tree:

        Attributes
        ----------
        nodes : dict
            dictionary of nodes;
        groups : dict
            dictionary of groups
        leafs : list
            list of leafs, calculated on demand
        players_list: list
            list of players names, indicating which game income from list is connected to which player
        """

        '''
        dictionary of nodes:
        Attributes
        ----------
        node : dict
            dictionary representing node;

            Attributes
            ----------
            value : float
                value of node (the prize for reaching the node)
            parents : dict
                parents of node - can be multiple, represented by dict of ids and connection values
            children : dict
                children of node - can be multiple, represented by dict of ids and connection values
            probability : float
                probability of node - 1 means there is no random choice
            branch : dict
                totals of branch, to avoid tree walking

                Attributes
                ----------
                value : float
                    total value of branch
                probability : float
                    probability of reaching this node in game
        '''

        # remember to add new attributes to add_node method default values setting
        self._nodes = {}
        # dictionary of knowledge groups
        self._groups = {} if groups is None else groups
        # dictionary of leafs
        self._leafs = [] if leafs is None else leafs
        self._players_list = [] if players_list is None else players_list

        # always add root
        self.add_node({
            'id': 'root',
            'player': '1',
        }) if nodes is None else nodes

    # ---------------------------------- NODES -------------------------------------------------------------------------
    def add_node(self, node: dict):
        """
        add node method. Runs basic validation before adding.

        :param dict node: dictionary of node's data
        """
        # check if it is not overriding existing node
        if node.get('id') is not None:
            if node['id'] in self._nodes:
                raise ValueError('tried to override node %s' % node['id'])
        else:
            raise ValueError('no id for node provided')

        # append node to list
        id_ = node['id']
        del node['id']

        # set default values for node
        # remember to add new attributes here and in __init__ root node
        node['player'] = '0' if node.get('player') is None else node['player']
        node['value'] = [0, 0] if node.get('value') is None else node['value']
        node['parents'] = {} if node.get('parents') is None else node['parents']
        node['children'] = {} if node.get('children') is None else node['children']
        node['probability'] = 1 if node.get('probability') is None else node['probability']
        node['branch'] = {} if node.get('branch') is None else node['branch']
        node['branch']['probability'] = 1 \
            if node['branch'].get('probability') is None else node['branch']['probability']

        # add player to the list of players if he is not there already
        if node['player'] not in self._players_list:
            self._players_list.append(node['player'])

        # add parenthood
        for parent in node['parents']:
            # noinspection PyTypeChecker
            self._nodes[parent]['children'][id_] = str(node['parents'][parent])

        # set depth to one more than first parent
        if node['parents']:
            node['depth'] = self._nodes[str(list(node['parents'].keys())[0])]['depth'] + 1
        else:
            node['depth'] = 0 if node.get('depth') is None else node['depth']

        # calculate total probability of node:
        # total probability equals sum of probabilities of parents multiplied by probability of node
        branch_probability = 0
        for parent in node['parents']:
            branch_probability += self._nodes[parent]['branch']['probability']
        node['branch']['probability'] = branch_probability * node['probability']

        # validate against the error of node not being connected to the rest of the tree via parents removal:
        if id_ is not 'root' and not node['parents']:
            raise ValueError('node [%s] is not connected to the tree - parents are empty' % id_)

        # add node
        self._nodes[id_] = node

    def add_vertex(self, id_: str, player: str, parents: dict):
        """
        add vertex from simplified function:

        :param str id_: id of the node
        :param str player: id of player owning the node
        :param dict parents: dictionary of parents for the node
        """
        self.add_node({
            'id': id_,
            'player': player,
            'parents': parents
        })

    def add_leaf(self, id_: str, value: list, parents: dict):
        """
        add leaf from simplified function:

        :param str id_: id of the node
        :param list value: list of node's values
        :param dict parents: dictionary of parents for the node
        """
        self.add_node({
            'id': id_,
            'value': value,
            'parents': parents
        })

    def copy_node(self, from_: str, to_: str):
        """
        create a copy of node's properties in another node

        :param str from_: origin node of properties
        :param str to_: destination node for properties
        """
        self._nodes[to_] = dict(self._nodes[from_])

    def change_node(self, node: dict):
        """
        change node method. Changes attributes provided in node dictionary

        :param dict node: dictionary of node's data
        """
        # check if it is not overriding existing node
        if node.get('id') is not None:
            if node['id'] not in self._nodes:
                raise ValueError('tried to change non-existing node %s' % node['id'])
        else:
            raise ValueError('no id for node provided')

        # change attributes
        id_ = node['id']
        del node['id']
        for attribute in node:
            self._nodes[id_][attribute] = node[attribute]

    # ---------------------------------- OBJECT BASIC METHODS ----------------------------------------------------------
    def get_parent(self, id_) -> str:
        """ get id of the parent node """
        return list(self._nodes[id_]['parents'].keys())[0]

    def get_player_index(self, id_) -> int:
        """ return player index from players list order """
        return self._players_list.index(self._nodes[id_]['player'])

    def get_path_to_node(self, id_: str, mode: str = 'nodes') -> list:
        """
        get path from root to the node
        :param str id_: id of the node you want to reach from root
        :param str mode: mode of return type, 'nodes' - make path with nodes id, 'moves' - make path with player choices
        """

        path_t = []
        node = id_

        while node is not 'root':
            if mode == 'nodes':
                path_t.insert(0, node)
            elif mode == 'moves':
                parent_ = self.get_parent(node)
                path_t.insert(0, self._nodes[parent_]['children'][node])
            else:
                raise ValueError('mode variable is not "nodes" nor "moves"')
            node = self.get_parent(node)

        if mode == 'nodes':
            path_t.insert(0, 'root')
        return path_t

    @staticmethod
    def _get_key(obj: dict, val: str) -> list:
        """
        get list of keys with specified value from obj dictionary
        :param dict obj: chosen dictionary
        :param str val: specified value
        """
        sublist = [key for (key, value) in obj.items() if value == val]
        if sublist:
            return sublist
        else:
            raise ValueError('key with value %s does not exist in %s' % (val, obj))

    def get_tree(self) -> dict:
        """ return copy of tree nodes structure dict"""
        return dict(self._nodes)

    # -------------- LEAFS -------------
    def calculate_leafs(self):
        """ calculate inner list of leafs ids """
        self._leafs = [node for node in self._nodes if not self._nodes[node]['children']]

    def get_leafs(self) -> list:
        """ return list of leafs ids. Will return empty list, if calculate_leafs() has not been called earlier. """
        return self._leafs[:]

    # -------------- GROUPS ------------
    def set_group(self, id_: str, player: str, group: list):
        """
        add list of ids to new group
        :param str id_: id of group
        :param str player: id of player owning the group
        :param list group: list of id's you want to create group with
        """
        self._groups[id_] = {
            'player': player,
            'group': group
        }

    def get_groups(self) -> dict:
        """ return dictionary of groups """
        return dict(self._groups)

    def get_groups_of_player(self, player: str) -> list:
        """ return list of all groups id's where player is the owner """
        return [group for group in self._groups if self._groups[group]['player'] == player]
    # ==================================================================================================================

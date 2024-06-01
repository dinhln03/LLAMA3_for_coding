from rlbot.utils.structures.game_data_struct import Physics, GameTickPacket, PlayerInfo
from rlbot.agents.base_agent import SimpleControllerState, BaseAgent
from states.state import State
from util.packet import ParsedPacket


class GroundSave(State):

    def score(self, parsed_packet: ParsedPacket, packet: GameTickPacket, agent: BaseAgent) -> float:
        return None

    def get_output(self, parsed_packet: ParsedPacket, packet: GameTickPacket, agent: BaseAgent) -> SimpleControllerState:
        return None
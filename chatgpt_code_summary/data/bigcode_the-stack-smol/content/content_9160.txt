import unittest

class IntcodeComputer():
    
    OP_ADD = 1    
    OP_MULTIPLY = 2    
    OP_INPUT = 3    
    OP_OUTPUT = 4
    OP_JUMP_TRUE = 5
    OP_JUMP_FALSE = 6
    OP_LESS_THAN = 7
    OP_EQUALS = 8
    OP_MOD_REL = 9
    OP_HALT = 99
    
    PARAM_MODE_POS = '0'
    PARAM_MODE_IMD = '1'
    PARAM_MODE_REL = '2'
    
    NOUN_ADDR = 1
    VERB_ADDR = 2
    RESULT_ADDR = 0
    START_ADDR = 0
    
    INIT_VAL = 0
    
    def __init__(self, data = []):
        self.inputs = []
        self.memory = []
        self.initial_memory = []
        if data:
            self.load_memory(data)
        
    def load_memory(self, data):
        self.initial_memory = self.normalize_memory(data)
        self.reset()
        
    def expand_memory(self, addr):
        needed_mem = addr - (len(self.memory) - 1)
        if needed_mem > 0:
            self.memory += ([self.INIT_VAL] * needed_mem)
        else:
            raise Exception(f'Cannot expand memory for addr {addr}')
        
    def check_addr(self, addr):
        if addr < 0:
            raise Exception(f'Addr {addr}, cannot be negative')
        
        if addr >= len(self.memory):
            self.expand_memory(addr)
            
        return addr
            
        
    def reset(self):
        if self.memory:
            del self.memory[:]
        self.memory = self.initial_memory.copy()
        
        if self.inputs:
            del self.inputs[:]
        self.inputs = []
        
        self.output = None
        self.last_input = None
        
        self.instruction_ptr = self.START_ADDR
        self.relative_base = self.START_ADDR
        
    def add_input(self, data):
        self.inputs.append(data)
        
    def print_program(self):
        print("Program: {:02d}{:02d}".format(self.memory[self.NOUN_ADDR],self.memory[self.VERB_ADDR]))

    def normalize_memory(self, intcode):
        if type(intcode) is str:
            return list(map(int, intcode.split(',')))

        elif type(intcode) is list:
            if type(intcode[0]) is str:
                return list(map(int, intcode))
            else:
                return intcode

        else:
            raise Exception('Corrupt intcode')
            
    def get_paramater(self, mode):
        param = self.memory[self.instruction_ptr]
        self.instruction_ptr += 1
        
        if mode == self.PARAM_MODE_POS:
            addr = self.check_addr(param)
            val = self.memory[addr]
        elif mode == self.PARAM_MODE_REL:
            addr = self.relative_base + param
            addr = self.check_addr(addr)
            val = self.memory[addr]
        elif mode == self.PARAM_MODE_IMD:
            val = param
        else:
            raise Exception(f"Unkown paramater mode: {param}")

        return val
        
    def set_paramater(self, mode, data):
        param = self.memory[self.instruction_ptr]
        self.instruction_ptr += 1
        
        if mode == self.PARAM_MODE_POS:
            addr = self.check_addr(param)
            self.memory[addr] = data
        elif mode == self.PARAM_MODE_REL:
            addr = self.relative_base + param
            addr = self.check_addr(addr)
            self.memory[addr] = data
        elif mode == self.PARAM_MODE_IMD:
            raise Exception("Set paramater can't be in immediate mode")
        else:
            raise Exception(f"Unkown paramater mode: {param}")            
        
    def parse_opcode(self):
        mode_opcode_str = '{:>05}'.format(str(self.memory[self.instruction_ptr]))
        
        # Reverse of the first three chars
        modes = mode_opcode_str[:3][::-1]
        
        # integer of the last two chars
        opcode = int(mode_opcode_str[3:])
                     
        self.instruction_ptr += 1
                     
        return modes, opcode
        

    def run(self):
        self.output = None
        
        while self.instruction_ptr < len(self.memory):
            param_mode, opcode = self.parse_opcode()

            if opcode == self.OP_HALT:
                return 0
                
            elif opcode == self.OP_ADD:
                in1 = self.get_paramater(param_mode[0])
                in2 = self.get_paramater(param_mode[1])
                self.set_paramater(param_mode[2], in1 + in2)
            
            elif opcode == self.OP_MULTIPLY:
                in1 = self.get_paramater(param_mode[0])
                in2 = self.get_paramater(param_mode[1])
                self.set_paramater(param_mode[2], in1 * in2)
            
            elif opcode == self.OP_INPUT:
                if self.inputs:
                    self.last_input = self.inputs.pop()
                    
                if self.last_input != None:
                    self.set_paramater(param_mode[0], self.last_input)
                else:
                    raise Exception(f"{self.last_input} is not a valid input")
                
            elif opcode == self.OP_OUTPUT:
                self.output = self.get_paramater(param_mode[0])
                return 1
                
            elif opcode == self.OP_JUMP_TRUE:
                do_jump = self.get_paramater(param_mode[0])
                new_addr = self.get_paramater(param_mode[1])
                if do_jump != 0:
                    self.instruction_ptr = new_addr
                
            elif opcode == self.OP_JUMP_FALSE:
                do_jump = self.get_paramater(param_mode[0])
                new_addr = self.get_paramater(param_mode[1])
                if do_jump == 0:
                    self.instruction_ptr = new_addr
                    
            elif opcode == self.OP_LESS_THAN:
                in1 = self.get_paramater(param_mode[0])
                in2 = self.get_paramater(param_mode[1])
                if in1 < in2:
                    self.set_paramater(param_mode[2], 1)
                else:
                    self.set_paramater(param_mode[2], 0)
                
            elif opcode == self.OP_EQUALS:
                in1 = self.get_paramater(param_mode[0])
                in2 = self.get_paramater(param_mode[1])
                if in1 == in2:
                    self.set_paramater(param_mode[2], 1)
                else:
                    self.set_paramater(param_mode[2], 0)
            
            elif opcode == self.OP_MOD_REL:
                val = self.get_paramater(param_mode[0])
                self.relative_base += val
                
            else:
                raise Exception(f'Unknown opcode {opcode} at addr {self.instruction_ptr}.')
                self.reset()
                
        return -1
    
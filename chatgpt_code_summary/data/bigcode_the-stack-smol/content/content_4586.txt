# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging
from urllib import response
from vinte_um import Jogador, VinteUm
import grpc
import helloworld_pb2
import helloworld_pb2_grpc
import time
import redis

def createLoginForm(stub):
        username = input("Digite seu login: ")
        password = input("Digite sua senha: ")

        _redis = redis.Redis(
        host= 'localhost',
        port= '6379',
        password = 'davi')

        _redis.set('username', username)
        value = _redis.get('username') 
        print("variavel do redis:", value)

        return stub.Login(helloworld_pb2.LoginRequest(username=username, password=password))

def runTurn(stub, auth_token):
        extraCard = input("Deseja cavar mais uma carta? S/N: ")
        return stub.TurnAction(helloworld_pb2.TurnRequest(auth_token=auth_token, dig = extraCard))

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('0.0.0.0:50051') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        login = createLoginForm(stub)
        print("Suas cartas são: ", login.message)
        
        while True:
                turnResponse = runTurn(stub, login.auth_token)  
                print("Suas cartas são: ", turnResponse.cards) 
                if turnResponse.message:
                        print(turnResponse.message)   
                if turnResponse.playing == "False":
                        break  
        winner = stub.VerifyTurn(helloworld_pb2.VerifyTurnRequest(auth_token=login.auth_token))
        print(winner.message)


if __name__ == '__main__':
    logging.basicConfig()
    run()

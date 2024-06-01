#!/usr/bin/env python2
# -*- encoding: utf-8 -*-

import pygame
import sys
import numpy as np

CONST_LOCK_FILE = "lock.txt"
#CONST_GRAPH_FILE = "../tsptours/graph.tsp"
CONST_GRAPH_FILE = "graph.tsp"
CONST_STOP = "STOP"
CONST_CUSTOM_FILE = None



def main():
    pygame.init()
    screen = pygame.display.set_mode((700,700))
    screen.fill((255,255,255))
    pygame.display.set_caption("Ant Colony TSP Solver - press ENTER to solve")
    graph = []
    tour = []
    cost = g = 0
    state = 0
    pygame.display.flip()
    while (True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                print "El usuario ha decidido cerrar la aplicación."
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and state == 0 and CONST_CUSTOM_FILE:
                #print "Agregando la posición del click", pygame.mouse.get_pos()
                data = np.loadtxt(CONST_CUSTOM_FILE, dtype=int, delimiter=',')
                for line in data:
                    line = (line[0]*7, line[1]*7)
                    graph.append(line)
                    pygame.draw.circle(screen, (0,0,0), line, 5, 0)
                pygame.display.flip()
                from_file = False
            elif event.type == pygame.MOUSEBUTTONDOWN and state == 0:
                #print "Agregando la posición del click", pygame.mouse.get_pos()
                graph.append(pygame.mouse.get_pos())
                pygame.draw.circle(screen, (0,0,0), pygame.mouse.get_pos(), 5, 0)
                pygame.display.flip()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                lock_file = open(CONST_LOCK_FILE, "w")
                lock_file.write("0");
                lock_file.close()    

                graph_file = open(CONST_GRAPH_FILE, "w")
                graph_file.write("NAME : %s\n" % CONST_GRAPH_FILE)
                graph_file.write("COMMENT : %s-city problem\n" % str(len(graph)))
                graph_file.write("TYPE : TSP\n")
                graph_file.write("DIMENSION : %s\n" % str(len(graph)))
                graph_file.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
                graph_file.write("NODE_COORD_SECTION\n") 

                for x in range(0, len(graph)):
                    #print "%d %d %d" % (x, graph[x][0], graph[x][1])
                    graph_file.write("%d %d %d" % (x, graph[x][0], graph[x][1]))
                    graph_file.write("\n")
                graph_file.write("EOF")
                graph_file.close()

                lock_file = open("lock.txt", "w")
                lock_file.write("1");
                lock_file.close()

                # Primera salida.
                tour = input() # [0, .., n-1, n]
                cost = input() # Costo del recorrido
                g = input() # Cantidad de iteraciones

                lock_file = open("lock.txt", "w")
                lock_file.write("0");
                lock_file.close()
                state = 1
        if state == 1:
            if tour != CONST_STOP:
                pygame.display.set_caption("Ant Colony TSP Solver - current length: " + str(cost) + " | iterations: " + str(g) + " (SOLVING...)")
                screen.fill((255,255,255))
                # Vuelve a dibujar los círculos
                for i in graph:
                    pygame.draw.circle(screen, (255,0,0), i, 5, 0)

                for i in range(0, len(tour)):
                    pygame.draw.line(screen, (255, 0, 0), graph[tour[i]], graph[tour[(i + 1) % len(tour)]])
                pygame.display.flip()
                # Salidas siguientes
                tour = input()

                if tour != CONST_STOP:
                    cost = input()
                    g = input()
            else:
                pygame.display.set_caption("Ant Colony TSP Solver - current length: " + str(cost) + " | iterations: " + str(g) + " (FINISHED)")
                finished = True
                state = 2

if __name__ == '__main__':
    if len(sys.argv) == 2:
        CONST_CUSTOM_FILE = sys.argv[1]

    main()
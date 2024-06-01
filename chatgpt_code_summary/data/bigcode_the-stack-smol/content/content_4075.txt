import Celula

class Labirinto:
    def __init__(self, num_rows, num_columns, order_to_check):

        # Indica a ordem que vai os vizinhos vao ser checados
        self.order_to_check = order_to_check

        # Numero de linhas no grid
        self.num_rows    = num_rows
        # Numero de colunas no grid
        self.num_columns = num_columns

        self.grid = []

        # Preenche o grid
        tmp_cell = Celula.Celula(0)
        for i in range(self.num_columns):
            self.grid.append([tmp_cell for x in range(self.num_rows)])


    # Printar o grid
    def __str__(self):
        grid_as_string = ""

        for i in range(self.num_columns):
            for j in range(self.num_rows):
                grid_as_string += f"{self.grid[i][j].get_value()} "

            grid_as_string += "\n"

        return grid_as_string
    

    # Adiciona a celula cell em [pos_y][pos_x]
    def insert(self, cell_value, pos_y, pos_x):
        self.grid[pos_y][pos_x] = Celula.Celula(cell_value)

        
        # Jeito rapido de resolver IndexError porque nao quero gastar muito tempo nesse codigo
        try:
            # Verificar se existe uma celula em cima
            if self.grid[pos_y-1][pos_x].get_value() != 0:
                self.grid[pos_y][pos_x].set_up(self.grid[pos_y-1][pos_x])
                self.grid[pos_y-1][pos_x].set_down(self.grid[pos_y][pos_x])

        except IndexError:
            pass

        try:
            # Verificar se existe uma celula embaixo
            if self.grid[pos_y+1][pos_x].get_value() != 0:
                self.grid[pos_y][pos_x].set_down(self.grid[pos_y+1][pos_x])
                self.grid[pos_y+1][pos_x].set_up(self.grid[pos_y][pos_x])

        except IndexError:
            pass

        try:
            # Verificar se existe uma celula na esquerda
            if self.grid[pos_y][pos_x-1].get_value() != 0:
                self.grid[pos_y][pos_x].set_left(self.grid[pos_y][pos_x])
                self.grid[pos_y][pos_x-1].set_right(self.grid[pos_y][pos_x])

        except IndexError:
            pass

        try:
            # Verificar se existe uma celula na direita
            if self.grid[pos_y+1][pos_x].get_value() != 0:
                self.grid[pos_y][pos_x].set_right(self.grid[pos_y+1][pos_x])
                self.grid[pos_y+1][pos_x].set_left(self.grid[pos_y][pos_x])

        except IndexError:
            pass

    
    def find_path(self, pos_x, pos_y):

        self.grid[pos_y][pos_x].visited = True
        
        # Se for a saida, printar a posicao dela!
        if self.grid[pos_y][pos_x].value == 2:
            print(f"Saida encontrada na posicao [{pos_x}][{pos_y}]!")


        # Verificar na ordem que foi recebida pela funcao
        for i in self.order_to_check:
            # Se existe alguem em cima, se esse alguem for diferente de None e de Zero, abrir uma recursao naquela posicao
            # pois eh um caminho!
            if i == "up" and self.grid[pos_y][pos_x].up != None and self.grid[pos_y][pos_x].up != 0:
                if not self.grid[pos_y][pos_x].up.visited:
                    self.find_path(pos_x, pos_y-1)
            
            # Se existe alguem na esquerda, se esse alguem for diferente de None e de Zero, abrir uma recursao naquela posicao
            # pois eh um caminho!
            if i == "left" and self.grid[pos_y][pos_x].left != None and self.grid[pos_y][pos_x].left != 0:
                if not self.grid[pos_y][pos_x].left.visited:
                    self.find_path(pos_x-1, pos_y)

            # Se existe alguem embaixo, se esse alguem for diferente de None e de Zero, abrir uma recursao naquela posicao
            # pois eh um caminho!
            if i == "down" and self.grid[pos_y][pos_x].down != None and self.grid[pos_y][pos_x].down != 0:
                if not self.grid[pos_y][pos_x].down.visited:
                    self.find_path(pos_x, pos_y+1)

            # Se existe alguem na direita, se esse alguem for diferente de None e de Zero, abrir uma recursao naquela posicao
            # pois eh um caminho!
            if i == "right" and self.grid[pos_y][pos_x].right != None and self.grid[pos_y][pos_x].right != 0:
                if not self.grid[pos_y][pos_x].right.visited:
                    self.find_path(pos_x+1, pos_y)

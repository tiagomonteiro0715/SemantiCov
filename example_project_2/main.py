"""
This program contains the logic of tic tac toe game along with the following custom methods 
play_tick_tac_toe() - main logical flow of the program
display_board() - to display the current status of the board
clear_board() -  to clear the board and put spaces on each place
place_maker() - to place the player symbol on the board
winner() - to declare the winner between player-1 and player-2
game_result() - to check if there is a winner between player-1 and player-2
select_symbol() - to assign symbol 'x' or 'o' to player-1 and remaining choice symbol to player-2
select_position() - to select the rows and column in between 1-3 for current player
play_again() - checks if user wants to play the game again
"""


class TicTacToe:
    
    """ First create a board using a 2d List, initially the board contains empty spaces, later on as the player will play the game
    we will put either 'X' or 'O' in those empty spaces """
    # default constructor
    def __init__(self):
        self.board=[[' ', ' ', ' '],
           [' ', ' ', ' '],
           [' ', ' ', ' ']]
    
    #this method will print current board status
    def display_board(self):
        print(self.board[0][0] + '|'+ self.board[0][1] + '|' + self.board[0][2])
        print('-+-+-')
        print(self.board[1][0] + '|' + self.board[1][1] + '|' + self.board[1][2])
        print('-+-+-')
        print(self.board[2][0] + '|' + self.board[2][1] + '|' + self.board[2][2])
    
    #this method is used to clear the board and put spaces on each place
    def clear_board(self):
        for i in range(3):
            for j in range(3):
                self.board[i][j]= " "

    #this method is used to place the player symbol on the board
    def place_marker(self, symbol, row, column):
        if self.board[row][column]== ' ':
            self.board[row][column]= symbol
            return True
        else:
            print("Position already filled \n enter a new place")
            return False
    
    #this helper method is used to declare the winner
    def winner(self, players_dict,a,b):
        if self.board[a][b] == players_dict['player-1']:
            print("Player-1 has Won the game with symbole= "+players_dict['player-1'])
           
        else:
            print("Player-2 has won the game with symbole= "+players_dict['player-2'])
            
   #this method is used to find if someone has won the game, either by row wise rule or column wise rule or diagonal rule 
    def game_result(self, players_dict):

        # first row comparison
        if self.board[0][0] == self.board[0][1] == self.board[0][2] != ' ':
            a=b=0
            self.winner(players_dict,a,b)
            return True
        
        # second row comparison
        elif self.board[1][0] == self.board[1][1] == self.board[1][2] != ' ':      
            a=1
            b=0
            self.winner(players_dict,a,b)
            return True

        # third row comparison
        elif self.board[2][0] == self.board[2][1] == self.board[2][2] != ' ':      
            a=2
            b=0
            self.winner(players_dict,a,b)
            return True
            
        # first column comparison
        elif self.board[0][0] == self.board[1][0] == self.board[2][0] != ' ':      
            a=b=0
            self.winner(players_dict,a,b)
            return True
        
        # second column comparison
        elif self.board[0][1] == self.board[1][1] == self.board[2][1] != ' ':      
            a=0
            b=1
            self.winner(players_dict,a,b)
            return True
        
        # third column cpmparison
        elif self.board[0][2] == self.board[1][2] == self.board[2][2] != ' ':      
            a=0
            b=2
            self.winner(players_dict,a,b)
            return True
        
        # first diaginal comparison
        elif self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':      
            a=b=0
            self.winner(players_dict,a,b) 
            return True

        
        # second diagonal comparison
        elif self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':      
            a=0
            b=2
            self.winner(players_dict,a,b)
            return True
        
        else:
            return False

    # this method will help player-1 to select either 'x' or 'o' and assign the other choice to player-2 and saves the choices in a dictionary
    def select_symbol(self):
        choice = input("Enter 'x' to select 'x' for player-1 or enter 'o' if you want to select 'o' for player-1:  ").lower()
        player_dict=dict()
        if(choice == 'x'):
            player_dict['player-1']="x"
            player_dict['player-2']="o"
            return player_dict
        elif(choice == 'o'):
            player_dict['player-1']="o"
            player_dict['player-2']="x"
            return player_dict
        else:
            print("Wrong choice entered, enter again \n")
            self.select_symbol()

    # this method will help the player to select the rows and column in between 1-3
    def select_position(self):
        choice_row = int(input("Enter the row value in between 1-3, at which you want to place the symbol:\t"))-1
        choice_column = int(input("Enter the column value in between 1-3, at which you want to place the symbol:\t"))-1
        list2=[]
        if((choice_row>=0 and choice_row<3) and (choice_column>=0 and choice_column<3)):
            list2.append(choice_row)
            list2.append(choice_column)
        else:
            print("Wrong choice of row or column entered, try again \n")
            self.select_position()
        return list2
    
    # this method will ask at the end of each game if the user wants to play the game again
    def play_again(self):
        ans= input("Enter y if you want to play again, else enter any key to exit:\t")
        if (ans=='y'):
            return True
        else:
            return False
    
    # this method contains the logical flow of the program
    def play_tick_tac_toe(self):
        
        print ("Welcome to Tic Tack Toe game \n")
        self.display_board()
        print ("Enter player-1 choice") 
        players_dict= self.select_symbol()  # It will call select_symbole method and save the choices of player-1 and player-2 in a dictionary
        print("Let the game begins \n \n")
        count =0
        turn= False
        while count<9:  # since the player can enter symbol anywhere on the 9 places, we will run a loop for players choices 
            self.display_board()
            
            if count>= 5:   # this if condition, we will check if there is a winner after 5 or more moves
                result= self.game_result(players_dict)  # this line will call game_result() method which will accepts a players dictionary comtaining player-1 and player-2 choices of 'x' and 'o' and declare the winner 
                if result:  # this if condition will executes if there is a winner 
                    self.display_board()
                    print("\nGame Over\n")
                    choice=self.play_again()    # this statement will ask the user if he/she wants to play the game again
                    if choice:
                        self.clear_board()    # this statement will clear the board for a new game
                        return True
                    else:
                        print ("\nGood Bye !!!")
                        return False
                
            if count==8:    # this if condition will check if there are 8 moves played without a winner, that means the game has resulted in a draw
                print("The game has resulted in a draw")
                choice=self.play_again()    # this statement will ask the user if he/she wants to play the game again
                if choice:
                    self.clear_board()      # this statement will clear the board for a new game
                    return True
                else:
                    print ("\nGood Bye !!!")
                    return False

            if not turn:    # this if condition will check if it is player-1 turn to play or not 
                print("Player-1 turn")
                placed1 = False
                while not placed1:  # this while loop will run untill player-1 has placed its symbol on an empty place on the board
                    player1_position = self.select_position()
                    placed1 = self.place_marker(players_dict['player-1'],player1_position[0],player1_position[1])
                count+= 1
                turn= True
                continue

            else:   # this else condition will execute if it is player-2 turn to play 
                print("Player-2 turn")
                placed2 =False
                while(not placed2):     # this while loop will run untill player-2 has placed its symbol on an empty place on the board
                    player2_position = self.select_position()
                    placed2 =self.place_marker(players_dict['player-2'],player2_position[0],player2_position[1])
                count+= 1
                turn = False
                continue

# this main method will create an object of TicTacToe class and keep on calling  play_tick_tac_toe() untill the user wishes to play the game
if __name__ == '__main__':
    obj = TicTacToe()
    choice= obj.play_tick_tac_toe()
    while (choice):
        choice=obj.play_tick_tac_toe()

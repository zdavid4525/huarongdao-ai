from __future__ import annotations

import heapq
import math
from copy import deepcopy
import time
import argparse
import sys
from typing import Optional

#====================================================================================

char_goal = '1'
char_single = '2'
WIDTH = 4
HEIGHT = 5
ONE_BY_TWO = "Rect"
TWO_BY_TWO = "Square"
ONE_BY_ONE = "Dot"

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle. At any time, there will be 10 piece objects:
        - One 2x2 represented by Piece.b_type == "Square"
        - 5 1x2 pieces (can be horizontal or vertical) represented by Piece.b_type = "Rect"
        - 4 1x1 pieces represented by Piece.b_type = "Dot"
    """
    is_goal: bool
    b_type: str
    coord_x: int
    coord_y: int
    orientation: str

    def __init__(self, is_goal: bool, b_type: str, coord_x: int, coord_y: int, orientation: str) -> None:
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece. x = 0 <==> rightmost col
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece. y = 0 <==> topmost row
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """
        self.is_goal = is_goal
        self.b_type = b_type
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.b_type, self.coord_x, self.coord_y, self.orientation)

    def __eq__(self, other: Piece) -> bool:
        return (self.is_goal == other.is_goal and self.b_type == other.b_type and self.coord_x == other.coord_x and
                self.coord_y == other.coord_y and self.orientation == other.orientation)


class Board:
    """
    Board class for setting up the playing board.
    """
    width: int
    height: int
    pieces: list[Piece]
    grid: list[list[str]]

    def __init__(self, pieces: list[Piece]):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """
        self.width = WIDTH
        self.height = HEIGHT
        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()


    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.
        """
        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.b_type == ONE_BY_ONE:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def get_empty_spaces(self) -> list[tuple[int, int]]:
        # Returns [(x_1, y_1), (x_2, y_2)] s.t. the spaces at (x_1, y_1), (x_2, y_2) are unoccupied.
        empty_spaces = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == ".":
                    empty_spaces.append((j, i))
        return empty_spaces

    def move_piece(self, piece: Piece, way: str) -> None:
        # Pre: There is enough space to make this move, i.e. the swapped pieces will be
        if way == "u":
            if piece.b_type == ONE_BY_ONE:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y - 1][piece.coord_x] = "2"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y - 1][piece.coord_x] = "<"
                self.grid[piece.coord_y - 1][piece.coord_x + 1] = ">"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y - 1][piece.coord_x] = "^"
                self.grid[piece.coord_y][piece.coord_x] = "v"
            elif piece.b_type == TWO_BY_TWO:
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = "."
                self.grid[piece.coord_y - 1][piece.coord_x] = "1"
                self.grid[piece.coord_y - 1][piece.coord_x + 1] = "1"
            piece.coord_y -= 1
        elif way == "d":
            if piece.b_type == ONE_BY_ONE:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "2"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "<"
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = ">"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "^"
                self.grid[piece.coord_y + 2][piece.coord_x] = "v"
            elif piece.b_type == TWO_BY_TWO:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y + 2][piece.coord_x] = "1"
                self.grid[piece.coord_y + 2][piece.coord_x + 1] = "1"
            piece.coord_y += 1
        elif way == "l":
            if piece.b_type == ONE_BY_ONE:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x - 1] = "2"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y][piece.coord_x - 1] = "<"
                self.grid[piece.coord_y][piece.coord_x] = ">"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x - 1] = "^"
                self.grid[piece.coord_y + 1][piece.coord_x - 1] = "v"
            elif piece.b_type == TWO_BY_TWO:
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = "."
                self.grid[piece.coord_y][piece.coord_x - 1] = "1"
                self.grid[piece.coord_y + 1][piece.coord_x - 1] = "1"
            piece.coord_x -= 1
        else:  # way == "r"
            if piece.b_type == ONE_BY_ONE:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "2"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "<"
                self.grid[piece.coord_y][piece.coord_x + 2] = ">"
            elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 1] = "^"
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = "v"
            elif piece.b_type == TWO_BY_TWO:
                self.grid[piece.coord_y][piece.coord_x] = "."
                self.grid[piece.coord_y + 1][piece.coord_x] = "."
                self.grid[piece.coord_y][piece.coord_x + 2] = "1"
                self.grid[piece.coord_y + 1][piece.coord_x + 2] = "1"
            piece.coord_x += 1

    def display(self):
        """
        Print out the current board.
        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()

    def to_str(self):
        res = ""
        for row in self.grid:
            for elem in row:
                res += elem
        return res


class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """
    board: Board
    f: float
    g: float
    parent: Optional[State]
    id: int

    def __init__(self, board: Board, parent: Optional[State] = None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = math.inf
        self.g = math.inf
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.

    def is_goal(self) -> bool:
        for piece in self.board.pieces:
            if piece.b_type == TWO_BY_TWO and piece.coord_x == 1 and piece.coord_y == 3:
                return True
        return False

    def get_successors(self) -> list[State]:
        successors = []

        successors.extend(self._get_one_by_one_successors())  # every successor obtained by moving some 1x1 once
        successors.extend(self._get_one_by_two_successors())  # every successor obtained by moving some 1x2 once
        successors.extend(self._get_two_by_two_successors())  # every successor obtained by moving some 2x2 once
        return successors

    def _get_one_by_one_successors(self) -> list[State]:
        successors = []

        for piece in self.board.pieces:  # there should be 4 total
            if piece.b_type == ONE_BY_ONE:
                if can_move_up(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "u")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_down(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "d")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_left(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "l")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_right(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "r")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)

        return successors

    def _get_one_by_two_successors(self) -> list[State]:
        successors = []

        for piece in self.board.pieces:  # both vertical and horizontal - there should be 5
            if piece.b_type == ONE_BY_TWO:
                if can_move_up(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "u")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_down(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "d")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_left(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "l")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)
                if can_move_right(piece, self.board):
                    succ_board = deepcopy(self.board)
                    new_piece = copy_and_replace(piece, succ_board)
                    succ_board.move_piece(new_piece, "r")

                    succ_state = State(succ_board, None)
                    successors.append(succ_state)

        return successors

    def _get_two_by_two_successors(self) -> list[State]:
        piece = None
        for candidate in self.board.pieces:
            if candidate.b_type == TWO_BY_TWO:
                piece = candidate

        successors = []

        # there is only one 2x2
        if can_move_up(piece, self.board):
            succ_board = deepcopy(self.board)
            new_piece = copy_and_replace(piece, succ_board)
            succ_board.move_piece(new_piece, "u")

            succ_state = State(succ_board, None)
            successors.append(succ_state)
        if can_move_down(piece, self.board):
            succ_board = deepcopy(self.board)
            new_piece = copy_and_replace(piece, succ_board)
            succ_board.move_piece(new_piece, "d")

            succ_state = State(succ_board, None)
            successors.append(succ_state)
        if can_move_left(piece, self.board):
            succ_board = deepcopy(self.board)
            new_piece = copy_and_replace(piece, succ_board)
            succ_board.move_piece(new_piece, "l")

            succ_state = State(succ_board, None)
            successors.append(succ_state)
        if can_move_right(piece, self.board):
            succ_board = deepcopy(self.board)
            new_piece = copy_and_replace(piece, succ_board)
            succ_board.move_piece(new_piece, "r")

            succ_state = State(succ_board, None)
            successors.append(succ_state)

        return successors


def can_move_up(piece: Piece, board: Board) -> bool:
    if piece.coord_y <= 0:
        return False

    # y-coord > 0
    if piece.b_type == ONE_BY_ONE or (piece.b_type == ONE_BY_TWO and piece.orientation == "v"):
        return board.grid[piece.coord_y - 1][piece.coord_x] == "."
    if piece.b_type == TWO_BY_TWO or (piece.b_type == ONE_BY_TWO and piece.orientation == "h"):
        return (board.grid[piece.coord_y - 1][piece.coord_x] == "." and
                board.grid[piece.coord_y - 1][piece.coord_x + 1] == ".")

    return False


def can_move_down(piece: Piece, board: Board) -> bool:
    if piece.coord_y >= HEIGHT - 1:
        return False

    # y-coord < 4
    if piece.b_type == ONE_BY_ONE:
        return board.grid[piece.coord_y + 1][piece.coord_x] == "."
    elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
        return piece.coord_y < 3 and board.grid[piece.coord_y + 2][piece.coord_x] == "."
    elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
        return (board.grid[piece.coord_y + 1][piece.coord_x] == "." and
                board.grid[piece.coord_y + 1][piece.coord_x + 1] == ".")
    elif piece.b_type == TWO_BY_TWO:
        return (piece.coord_y < 3 and board.grid[piece.coord_y + 2][piece.coord_x] == "." and
                board.grid[piece.coord_y + 2][piece.coord_x + 1] == ".")

    return False


def can_move_left(piece: Piece, board: Board) -> bool:
    if piece.coord_x <= 0:
        return False

    # x-coord > 0
    if piece.b_type == ONE_BY_ONE or (piece.b_type == ONE_BY_TWO and piece.orientation == "h"):
        return board.grid[piece.coord_y][piece.coord_x - 1] == "."
    if piece.b_type == TWO_BY_TWO or (piece.b_type == ONE_BY_TWO and piece.orientation == "v"):
        return (board.grid[piece.coord_y][piece.coord_x - 1] == "." and
                board.grid[piece.coord_y + 1][piece.coord_x - 1] == ".")

    return False


def can_move_right(piece: Piece, board: Board) -> bool:
    if piece.coord_x >= WIDTH - 1:
        return False

    # x-coord < WIDTH - 1
    if piece.b_type == ONE_BY_ONE:
        return board.grid[piece.coord_y][piece.coord_x + 1] == "."
    elif piece.b_type == ONE_BY_TWO and piece.orientation == "v":
        return (board.grid[piece.coord_y][piece.coord_x + 1] == "." and
                board.grid[piece.coord_y + 1][piece.coord_x + 1] == ".")
    elif piece.b_type == ONE_BY_TWO and piece.orientation == "h":
        return piece.coord_x < 2 and board.grid[piece.coord_y][piece.coord_x + 2] == "."
    elif piece.b_type == TWO_BY_TWO:
        return (piece.coord_x < 2 and board.grid[piece.coord_y][piece.coord_x + 2] == "." and
                board.grid[piece.coord_y + 1][piece.coord_x + 2] == ".")


def copy_and_replace(old_piece: Piece, board: Board) -> Piece:
    new_piece = deepcopy(old_piece)
    for i in range(len(board.pieces)):
        if board.pieces[i].__eq__(old_piece):
            board.pieces[i] = new_piece
    return new_piece


def manhatten_dist(board: Board) -> int:
    x_curr, y_curr = None, None
    for piece in board.pieces:
        if piece.b_type == TWO_BY_TWO:
            x_curr, y_curr = piece.coord_x, piece.coord_y

    return abs(x_curr - 1) + abs(y_curr - 3)


def advanced_heuristic(state: State) -> int:
    # It should be admissible. It should dominate the Manhattan distance heuristic.
    return 0


def get_solution(state: State):
    soln = []
    while state is not None:
        soln.append(state)
        state = state.parent
    return soln


def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """
    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:
        for x, ch in enumerate(line):
            if ch == '^': # found vertical piece
                pieces.append(Piece(False, ONE_BY_TWO, x, line_index, 'v'))
            elif ch == '<': # found horizontal piece
                pieces.append(Piece(False, ONE_BY_TWO, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, ONE_BY_ONE, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, TWO_BY_TWO, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()
    board = Board(pieces)
    
    return board


def write_to_file(filename, soln):
    """
    Write solution into a given file.
    """
    out = open(filename, "w")

    for i in range(len(soln)):
        for i, line in enumerate(soln[i].board.grid):
            for ch in line:
                print(ch, end='', file=out)
            print("", file=out)
        print("", file=out)

    out.close()


def dfs(s: State) -> list[State]:
    explored = {s.board.to_str()}
    frontier = [s]

    while frontier:
        curr = frontier.pop()
        if curr.is_goal():
            return get_solution(curr)
        for successor in curr.get_successors():
            if successor.board.to_str() not in explored:
                successor.parent = curr
                explored.add(successor.board.to_str())
                frontier.append(successor)
    print("no goal found")


def a_star(s: State) -> list[State]:
    s.g = 0
    s.f = manhatten_dist(s.board)
    frontier = [(s.f, s.id, s)]
    heapq.heapify(frontier)
    explored = {s.board.to_str()}

    while frontier:
        curr_f, _, curr = heapq.heappop(frontier)
        if curr.is_goal():
            return get_solution(curr)

        for successor in curr.get_successors():
            if successor.board.to_str() not in explored:
                tentative_g = curr.g + 1
                if tentative_g < successor.g:
                    successor.parent = curr
                    successor.g = tentative_g
                    successor.f = successor.g + manhatten_dist(successor.board)
                    explored.add(successor.board.to_str())
                    if all(successor.board.grid != o[2].board.grid for o in frontier):
                        heapq.heappush(frontier, (successor.f, successor.id, successor))
    print("no goal found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)
    initial = State(board, None)

    soln = None
    if args.algo == "astar":
        soln = a_star(initial)
    else:  # args.algo == "dfs"
        soln = dfs(initial)

    write_to_file(args.outputfile, soln)
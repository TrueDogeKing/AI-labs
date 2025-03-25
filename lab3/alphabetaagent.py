import copy

import time

class AlphaBetaAgent:
    def __init__(self, my_token, depth=4):
        self.my_token = my_token
        self.opponent_token = 'o' if my_token == 'x' else 'x'
        self.depth = depth
        self.thinking=0

    def decide(self, game):
        start_time = time.time()
        _, move = self.alphabeta(game, self.depth, float('-inf'), float('inf'), True)
        end_time = time.time()
        self.thinking+=end_time-start_time
        return move

    def alphabeta(self, game, depth, alpha, beta, maximizing_player):
        if game.game_over or depth == 0:
            return self.evaluate(game), None
        
        valid_moves = game.possible_drops()
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.drop_token(move)
                eval_value, _ = self.alphabeta(new_game, depth - 1, alpha, beta, False)
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_move = move
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in valid_moves:
                new_game = copy.deepcopy(game)
                new_game.drop_token(move)
                eval_value, _ = self.alphabeta(new_game, depth - 1, alpha, beta, True)
                if eval_value < min_eval:
                    min_eval = eval_value
                    best_move = move
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, game):
        if game.wins == self.my_token:
            return 1000
        elif game.wins == self.opponent_token:
            return -1000
        else:
            return self.score_position(game)

    def score_position(self, game):
        score = 0
        for four in game.iter_fours():
            score += self.evaluate_four(four)
        return score

    def evaluate_four(self, four):
        score = 0
        if four.count(self.my_token) == 4:
            score += 100
        elif four.count(self.my_token) == 3 and four.count('_') == 1:
            score += 10
        elif four.count(self.my_token) == 2 and four.count('_') == 2:
            score += 5
        if four.count(self.opponent_token) == 3 and four.count('_') == 1:
            score -= 80
        return score
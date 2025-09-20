from random import choice, randrange, shuffle
from time import time
from Deck import Deck  # kept for compatibility if needed elsewhere
from players.RandomPlayer import RandomPlayer
import sys
import numpy as np
sys.path.append("..")


class MCPlayerLite(RandomPlayer):
    """
    Versione ottimizzata:
    - Simulazioni su interi (valori 0..39) invece di oggetti Card durante i loop hot.
    - Niente .remove(card) lineare: usiamo pop(randrange(...)).
    - Ridotte allocazioni: costruiamo 'remaining' una volta per simulazione.
    - Early-exit logico nelle prese quando è impossibile raggiungere/superare la predizione target.
    - In play_card: per ogni ripetizione generiamo UN solo scenario (mani avversarie) e valutiamo TUTTE
      le mie carte candidate su quello scenario (variance reduction, più stabilità a parità di rep).
    - make_prediction_last_round lasciato invariato, come richiesto.
    """

    def __init__(self, name="MCPlayerLite"):
        super().__init__(name)
        self.time_limit = 1   # tempo max per decisione (secondi)
        self.max_reps   = 10000    # simulazioni max per decisione

    # ------------------- Utility interne (solo interi) -------------------
    @staticmethod
    def _deal_opponents_hands_int(remaining_vals, num_players, my_pos, cards_per_player):
        """Restituisce una lista di mani (liste di int). La mano di my_pos verrà riempita a parte."""
        # mischiamo e splittiamo a blocchi per performance
        pool = remaining_vals[:]
        shuffle(pool)
        hands = [[] for _ in range(num_players)]
        idx = 0
        for p in range(num_players):
            if p == my_pos:
                continue
            hands[p] = pool[idx: idx + cards_per_player]
            idx += cards_per_player
        return hands

    @staticmethod
    def _simulate_tricks_int(hands, my_pos, to_play, forced_first=None, target_pred=None):
        """
        Simula 'to_play' prese su mani rappresentate come liste di INT.
        - hands: list[list[int]] lunghezze coerenti con to_play per ciascun giocatore.
        - forced_first: se non None, è il valore INT che io (my_pos) devo giocare alla prima presa.
        - target_pred: intero o None. Se settato, abilita early-exit logico.
        Ritorna catches: lista prese vinte per ciascun giocatore.
        """
        num_players = len(hands)
        catches = [0] * num_players
        for t in range(to_play):
            played = []
            for p in range(num_players):
                if p == my_pos and t == 0 and forced_first is not None:
                    # gioca specifica carta: rimuovi per valore
                    # (lista corta: next(...) è OK; meglio ancora: pop su indice trovato)
                    hp = hands[p]
                    # trova indice della carta forced_first
                    # assumiamo che il valore sia presente
                    fi = next(i for i, v in enumerate(hp) if v == forced_first)
                    card = hp.pop(fi)
                else:
                    hp = hands[p]
                    # pop random O(1) atteso
                    i = randrange(len(hp))
                    card = hp.pop(i)
                played.append(card)
            # vincitore = indice del max valore
            winner = played.index(max(played))
            catches[winner] += 1

            # early-exit: se target_pred è definito e sono io il giocatore di interesse
            if target_pred is not None:
                my_c = catches[my_pos]
                remaining_tricks = to_play - (t + 1)
                # impossibile raggiungere target
                if my_c > target_pred:
                    break
                if my_c + remaining_tricks < target_pred:
                    break
        return catches

    # ------------------- PREDIZIONE (mano > 1 carta) -------------------
    def make_prediction(self, game_state):
        start_time   = time()
        num_players  = game_state["num_players"]
        hand_size    = game_state["hand_size"]
        order        = game_state["players_position"]
        my_pos       = order.index(self.id)

        possible_preds = list(range(hand_size + 1))
        preds_wins     = {p: 0 for p in possible_preds}

        my_vals = [c.value() for c in self.hand]

        for _ in range(self.max_reps):
            if time() - start_time > self.time_limit:
                break

            # scegli una mia predizione candidata
            my_pred = choice(possible_preds)

            # remaining di interi per gli altri
            remaining = [v for v in range(40) if v not in my_vals]

            # mani iniziali (liste di int)
            hands = self._deal_opponents_hands_int(
                remaining_vals=remaining,
                num_players=num_players,
                my_pos=my_pos,
                cards_per_player=hand_size,
            )
            # la mia mano in int
            hands[my_pos] = my_vals[:]

            # regola "somma != hand_size" se sono l'ultimo a parlare
            if my_pos == num_players - 1 and hand_size > 1:
                # somma predizioni: usa quelle esistenti per i precedenti, mie = my_pred, successivi random
                preds = [0] * num_players
                for p in range(num_players):
                    if p < my_pos:
                        prev = game_state["predictions_made"][p]
                        preds[p] = int(prev) if prev is not None else 0
                    elif p == my_pos:
                        preds[p] = my_pred
                    else:
                        preds[p] = choice(possible_preds)
                if sum(preds) == hand_size:
                    # micro-aggiustamento
                    my_pred = my_pred - 1 if my_pred > 0 else my_pred + 1

            # Simula l'intero round come int
            catches = [0] * num_players
            for _t in range(hand_size):
                played = []
                for p in range(num_players):
                    i = randrange(len(hands[p]))
                    played.append(hands[p].pop(i))
                winner = played.index(max(played))
                catches[winner] += 1

            if my_pred == catches[my_pos]:
                preds_wins[my_pred] += 1

        # scelta migliore + correzione extra se sono davvero ultimo nel gioco corrente
        self.prediction = max(preds_wins, key=preds_wins.get)
        preds_made = game_state["predictions_made"]
        if order[-1] == self.id and hand_size > 1:
            safe_sum = sum(int(x) for x in preds_made if x is not None)
            if safe_sum + int(self.prediction) == hand_size:
                self.prediction = int(self.prediction) - 1 if self.prediction > 0 else int(self.prediction) + 1

        return int(self.prediction)

    # ------------------- PREDIZIONE (mano = 1 carta) -------------------
    def make_prediction_last_round(self, game_state):
        cards_values = [card.value() for card in game_state["cards_visible"]]
        remaining_cards = [v for v in range(0, 41) if v not in cards_values]
        if max(cards_values) > np.mean(remaining_cards):
            self.prediction = 0
        else:
            self.prediction = 1
        return self.prediction

    # ------------------- SCELTA CARTA -------------------
    def play_card(self, game_state):
        start_time   = time()
        order        = game_state["players_position"]
        my_pos       = order.index(self.id)
        num_players  = game_state["num_players"]
        hand_size    = game_state["hand_size"]
        to_play      = hand_size - game_state["turn"]

        # miei valori (int) e target pred (int o 0 se None)
        my_vals  = [c.value() for c in self.hand]
        target   = int(self.prediction or 0)
        scores   = {v: 0 for v in my_vals}

        # carte già giocate nella presa corrente (se non è la prima del round)
        already_played_vals = []
        if game_state.get("turn", 0) > 0:
            already_played_vals = [c.value() for c in game_state.get("played_cards", {}).values()]

        for _ in range(self.max_reps):
            if time() - start_time > self.time_limit:
                break

            # Remaining valori per tutti tranne me (e tranne già giocati se non prima presa)
            remaining = [v for v in range(40) if v not in my_vals and v not in already_played_vals]

            # Costruisci uno scenario: mani avversari per il resto del round
            hands_base = self._deal_opponents_hands_int(
                remaining_vals=remaining,
                num_players=num_players,
                my_pos=my_pos,
                cards_per_player=to_play,
            )

            # Valuta TUTTE le candidate nello stesso scenario (variance reduction)
            for cand in my_vals:
                # Copie superficiali delle mani (solo liste di int, leggere)
                hands = [h[:] for h in hands_base]
                # Mia mano residua per questo scenario: my_vals con una carta in più perché devo avere to_play carte
                # Nota: nel round reale ho 'to_play' carte; qui partiamo da my_vals (>= to_play), ma per la simulazione
                # delle ultime 'to_play' prese, consideriamo solo una copia e popperemo.
                # Costruiamo mano iniziale con una selezione casuale di 'to_play' valori dalle mie (inclusa la candidata)
                # per rispettare la cardinalità corretta.
                # Assicuriamoci che la candidata sia presente:
                # - includiamo la candidata
                # - scegliamo a caso (to_play-1) tra le altre
                my_pool = my_vals[:]
                my_pool.remove(cand)
                # se servono altre carte, prendiamo a caso fino a raggiungere to_play-1
                if len(my_pool) >= to_play - 1:
                    # selezione semplice: mischiamo e prendiamo la testa
                    shuffle(my_pool)
                    my_hand_sim = my_pool[:to_play - 1] + [cand]
                else:
                    # caso limite: se per qualche motivo ho esattamente 'to_play' carte, prendo tutte
                    my_hand_sim = my_vals[:]
                hands[my_pos] = my_hand_sim

                catches = self._simulate_tricks_int(
                    hands=hands,
                    my_pos=my_pos,
                    to_play=to_play,
                    forced_first=cand,
                    target_pred=target,
                )

                if catches[my_pos] == target:
                    scores[cand] += 1

        # scegli la carta con score migliore e rimuovila dalla mano reale
        best_val = max(scores, key=scores.get)
        best_idx = next(i for i, c in enumerate(self.hand) if c.value() == best_val)
        return self.hand.pop(best_idx)

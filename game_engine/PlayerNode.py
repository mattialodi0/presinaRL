

class PlayerNode:
    def __init__(self, player):
        self.player = player
        self.next = None


def link_players(players):
    if not players:
        return None

    head = PlayerNode(players[0])
    current = head

    for i in range(1, len(players)):
        new_node = PlayerNode(players[i])
        current.next = new_node
        current = new_node

    current.next = head  # Make it circular
    return head

def remove_player(head, player_id):
    if head is None:
        return None

    current = head
    prev = None

    while True:
        if current.player.id == player_id:
            if prev is not None:
                prev.next = current.next
            else:
                # Removing the head node
                tail = head
                while tail.next != head:
                    tail = tail.next
                if tail == head:  # Only one node in the list
                    return None
                tail.next = head.next
                head = head.next
            return head

        prev = current
        current = current.next

        if current == head:
            break  # Came back to the head, player not found

    return head  # Player not found, return original head

def goto_player(head, player_id):
    if head is None:
        return None

    current = head

    while True:
        if current.player.id == player_id:
            return current

        current = current.next

        if current == head:
            break  # Came back to the head, player not found

    return None  # Player not found
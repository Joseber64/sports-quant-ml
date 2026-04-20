match_id = str(pick.get('match_id', ''))
if not match_id:
    return

# Simplificación de obtención de EV
current_ev = pick.get('ev') or 0.0
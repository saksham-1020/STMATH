# import math
# from .core import Value

# def profit_percent(cp, sp):
#     if cp <= 0: raise ValueError("Cost Price must be positive")
#     cp_val = cp.data if isinstance(cp, Value) else cp
#     sp_val = sp.data if isinstance(sp, Value) else sp
#     return ((sp_val - cp_val) / cp_val) * 100.0

# def loss_percent(cp, sp):
#     if cp <= 0: raise ValueError("Cost Price must be positive")
#     cp_val = cp.data if isinstance(cp, Value) else cp
#     sp_val = sp.data if isinstance(sp, Value) else sp
#     return ((cp_val - sp_val) / cp_val) * 100.0

# def work_done_together(days_list):
#     if not days_list: return 0.0
#     reciprocal_sum = sum(1.0/d for d in days_list if d > 0)
#     return 1.0 / reciprocal_sum

# def emi_calc(principal, rate_annual, tenure_years):
#     r = rate_annual / (12 * 100)
#     n = tenure_years * 12
#     if r == 0: return principal / n
#     power_factor = (1 + r)**n
#     emi = (principal * r * power_factor) / (power_factor - 1)
#     return emi

# def boat_streams(boat_speed, stream_speed, distance):
#     u_speed = boat_speed - stream_speed
#     d_speed = boat_speed + stream_speed
#     t_up = distance / u_speed if u_speed > 0 else float('inf')
#     t_down = distance / d_speed
#     return t_up, t_down

# def train_crossing_time(len1, speed1, len2=0, speed2=0, opposite=True):
#     v_rel = (speed1 + speed2) if opposite else abs(speed1 - speed2)
#     if v_rel == 0: return float('inf')
#     return (len1 + len2) / v_rel

# def avg_speed(segments):
#     # Segments format: [(distance, speed), (distance, speed)...]
#     total_dist = sum(s[0] for s in segments)
#     total_time = sum(s[0] / s[1] for s in segments if s[1] > 0)
#     return total_dist / total_time if total_time > 0 else 0.0

# def compound_interest_continuous(p, r, t):
#     # Advanced continuous compounding using custom exp
#     from .special import exp_custom
#     return p * exp_custom(r * t)
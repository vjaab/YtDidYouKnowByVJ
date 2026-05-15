import numpy as np

def easeInOutQuad(t):
    t = max(0.0, min(1.0, t))
    return 2*t*t if t < 0.5 else 1 - pow(-2*t + 2, 2) / 2

def test_lambda_capture():
    duration = 112.2
    interval = 10.0
    display_dur = 7.0
    
    lambdas = []
    
    for start_t in np.arange(1.0, duration, interval):
        current_dur = min(display_dur, duration - start_t)
        if current_dur < 1.0:
            continue
        
        # New style with default argument
        f = lambda t, cd=current_dur: 1.0 + 0.25 * easeInOutQuad(t / cd)
        lambdas.append((start_t, current_dur, f))
    
    print(f"Total lambdas: {len(lambdas)}")
    
    # Check first lambda
    st, cd, f = lambdas[0]
    print(f"First lambda: start_t={st}, expected cd={cd}")
    val = f(cd) # Should be 1.25
    print(f"Value at t={cd}: {val}")
    
    # Check if it's using its own cd, not the last one
    last_cd = lambdas[-1][1]
    print(f"Last cd: {last_cd}")
    val_oob = f(cd + 1.0) # t > cd, should be clamped by easeInOutQuad
    print(f"Value at t={cd + 1.0} (clamped): {val_oob}")

test_lambda_capture()

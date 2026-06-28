import time
import datetime
import sys
import os
from collections import deque
import pandas as pd
import yfinance as yf
import pytz
import numpy as np 
import math
import pickle
import pytz


# 1. Update the path FIRST before any custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
site_packages_path = os.path.join(current_dir, 'venv', 'lib', 'python3.13', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)

# 2. Import the Rust binary and create a GLOBAL alias for all files
try:
    import eqie_phidias
    sys.modules['eqie_core'] = eqie_phidias  # Tricks all other scripts into seeing 'eqie_core'
    import eqie_core
except ImportError:
    import eqie_core # Fallback

# 3. NOW import your local Python agents safely
from core import evaluator, risk_agent

def fetch_overnight_levels(max_logical_range=300.0):
    """
    Fetches the CME Nasdaq 100 (/NQ) overnight high and low,
    with a synthetic override for anomalous rollover data.
    """
    sys.stdout.write("[System] Reaching out to Yahoo Finance for overnight data... ")
    sys.stdout.flush()
    try:
        # NQ=F is the Yahoo Finance ticker for the continuous Nasdaq 100 Futures contract
        nq = yf.Ticker("NQ=F")
        
        # Fetch last 5 days of 1-minute data to safely bridge the weekend gap
        df = nq.history(period="5d", interval="1m")
        
        if df.empty:
            raise ValueError("No data returned.")

        # Standardize timezone to US/Central
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('US/Central')
        
        now = pd.Timestamp.now(tz='US/Central')
        
        # Define the end of the overnight session (8:30 AM today)
        cutoff_time = now.replace(hour=8, minute=30, second=0, microsecond=0)
        
        # If running pre-market, evaluate data up to 'now'. Otherwise, hard stop at 8:30 AM.
        end_time = now if now < cutoff_time else cutoff_time
        
        # The overnight session started 15.5 hours before the 8:30 AM bell (17:00 PM prior day)
        start_time = end_time - pd.Timedelta(hours=15, minutes=30)
        
        # Slice the timezone-aware dataframe
        overnight_data = df[(df.index >= start_time) & (df.index <= end_time)]
        
        if overnight_data.empty:
             raise ValueError("Overnight slice is empty.")
             
        onh = overnight_data['High'].max()
        onl = overnight_data['Low'].min()
        
        # --- THE ANOMALY OVERRIDE MECHANISM ---
        calc_range = onh - onl
        if calc_range > max_logical_range:
            print(f"Failed. Anomalous rollover range detected ({calc_range:.2f} pts).")
            sys.stdout.write("[System] Deploying synthetic pre-market containment levels... ")
            sys.stdout.flush()
            
            # Override: Shrink the window to just the European/Pre-market session (last 3 hours)
            synth_start = end_time - pd.Timedelta(hours=3)
            synth_data = overnight_data[overnight_data.index >= synth_start]
            
            if not synth_data.empty:
                onh = synth_data['High'].max()
                onl = synth_data['Low'].min()
            else:
                # Absolute fallback: Anchor a tight bracket to the final pre-market print
                last_px = overnight_data['Close'].iloc[-1]
                onh = last_px + 75.0
                onl = last_px - 75.0
        
        print("Success.")
        return round(onh, 2), round(onl, 2)
        
    except Exception as e:
        print(f"Failed ({e}).")
        print("[Warning] Defaulting to infinite levels. Strategy 1 will remain offline today.")
        return -float('inf'), float('inf')

# ==========================================
# 1. DUAL-ASSET MARKET CONTEXT ENGINE
# ==========================================

class PerformanceTracker:
    def __init__(self):
        self.trade_pnls = []
        
        # ML Evaluation Matrix
        self.true_positives = 0   # Profitable Executions
        self.false_positives = 0  # Losing Executions
        self.false_negatives = 0  # Missed Opportunities (Vetoed or missed moves)

    def log_trade(self, pnl):
        self.trade_pnls.append(pnl)
        if pnl > 0:
            self.true_positives += 1
        else:
            self.false_positives += 1

    def log_missed_opportunity(self):
        """Call this when the market completes a valid structural move, but the engine was flat."""
        self.false_negatives += 1

    def get_precision(self):
        # Precision: When we pull the trigger, how often are we right? (Win Rate)
        total_executed_trades = self.true_positives + self.false_positives
        if total_executed_trades == 0: return 0.0
        return self.true_positives / total_executed_trades

    def get_recall(self):
        # Recall: Out of all the profitable moves the market made today, how many did we catch?
        total_market_opportunities = self.true_positives + self.false_negatives
        if total_market_opportunities == 0: return 0.0
        return self.true_positives / total_market_opportunities

    def get_f1_score(self):
        # The Harmonic Mean of Precision and Recall
        precision = self.get_precision()
        recall = self.get_recall()
        
        if (precision + recall) == 0: return 0.0
        
        return 2 * ((precision * recall) / (precision + recall))

    def get_trade_sharpe(self):
        if len(self.trade_pnls) < 2: return 0.0
        mean_pnl = sum(self.trade_pnls) / len(self.trade_pnls)
        variance = sum((x - mean_pnl) ** 2 for x in self.trade_pnls) / (len(self.trade_pnls) - 1)
        std_dev = variance ** 0.5
        if std_dev == 0: return 99.9 
        return mean_pnl / std_dev
    
    def get_win_rate(self):
        return self.get_precision() * 100.0
    
    
class KalmanFilter:
    """
    Tracks the dynamic, cointegrated relationship between two assets tick-by-tick.
    It solves for the equation: NQ = (Beta * ES) + Intercept + Error
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # State transition variance (how fast the true relationship is allowed to change)
        self.Vw = delta / (1 - delta)
        # Measurement variance (how much noise we expect in the tick data)
        self.R = R
        # Initial state: [Hedge Ratio (Beta), Intercept]
        self.beta = np.zeros(2)
        # Covariance matrix of the state estimation
        self.P = np.zeros((2, 2))

    def update(self, x_price, y_price):
        """
        x_price: The independent variable (e.g., ES)
        y_price: The dependent variable (e.g., NQ)
        Returns: The dynamic beta and the prediction error (the stationary spread).
        """
        # Create the observation vector [ES Price, 1.0 (for the intercept)]
        F = np.array([x_price, 1.0])
        
        # 1. Prediction Step (Expand the uncertainty matrix)
        self.P = self.P + (np.eye(2) * self.Vw)
        
        # 2. Calculate the expected Y price based on our current Beta
        y_hat = np.dot(F, self.beta)
        
        # 3. The Error: The actual Y price minus what the model predicted.
        # THIS is the truly stationary spread we want to trade.
        spread_error = y_price - y_hat 
        
        # 4. Update Step (Kalman Gain)
        Q = np.dot(np.dot(F, self.P), F.T) + self.R
        K = np.dot(self.P, F.T) / Q
        
        # 5. Refine our Beta and Covariance based on the new tick
        self.beta = self.beta + (K * spread_error)
        self.P = self.P - np.outer(K, np.dot(F, self.P))
        
        return self.beta[0], spread_error
    

class MarketContext:
    def __init__(self, symbols):
        self.data = {
            sym: {
                'overnight_high': -float('inf'),
                'overnight_low': float('inf'),
                'ny_open_price': 0.0,
                'vwap_history': deque(maxlen=20),
                'price_history': deque(maxlen=60),
                'ema_9': 0.0,
                'ema_21': 0.0,
                'ema_50': 0.0,
                'macro_prices': deque(maxlen=60),  
                'macro_trend': "NEUTRAL",          
                'tick_counter': 0                  
            } for sym in symbols
        }
        
        # --- NEW: STATISTICAL ARBITRAGE TRACKERS ---
        self.kalman_filter = KalmanFilter()
        # We can use a longer memory here because the Kalman spread is genuinely stationary
        self.stationary_spread_history = deque(maxlen=300) 
        self.current_spread_z_score = 0.0
        self.dynamic_es_qty = 1 # Default

    def update_tick(self, symbol, price, vwap, current_time_obj):
        # ... [Keep your existing MTF, Overnight, and EMA Ribbon logic exactly as is] ...

        # --- 3. STATISTICAL ARBITRAGE: Kalman Cointegration Tracking ---
        # We only update the filter when we have valid data for both legs
        if "/NQU26:XCME" in self.data and "/ESU26:XCME" in self.data:
            nq_history = self.data["/NQU26:XCME"]['price_history']
            es_history = self.data["/ESU26:XCME"]['price_history']
            
            if len(nq_history) > 0 and len(es_history) > 0:
                nq_price = nq_history[-1]
                es_price = es_history[-1]
                
                # A. Feed the tick to the Kalman Filter
                dynamic_beta, spread_error = self.kalman_filter.update(es_price, nq_price)
                
                # B. Track the resulting stationary error
                self.stationary_spread_history.append(spread_error)
                
                # C. Calculate the Z-Score of the True Spread
                if len(self.stationary_spread_history) >= 60:
                    # Using NumPy for speed here instead of pure Python loops
                    mean_spread = np.mean(self.stationary_spread_history)
                    std_dev = np.std(self.stationary_spread_history)
                    
                    if std_dev > 0:
                        self.current_spread_z_score = (spread_error - mean_spread) / std_dev

                # D. Calculate the Hedge Ratio (Beta * (NQ Multiplier / ES Multiplier))
                multiplier_ratio = 20.0 / 50.0 
                raw_hedge_ratio = dynamic_beta * multiplier_ratio
                
                # Update the required ES contracts for the entry logic
                self.dynamic_es_qty = max(1, round(2 * raw_hedge_ratio))

    def get_vwap_slope(self, symbol):
        ctx = self.data.get(symbol)
        if not ctx or len(ctx['vwap_history']) < 20: return 0.0
        return ctx['vwap_history'][-1] - ctx['vwap_history'][0]

    def get_dynamic_volatility_multiplier(self, symbol, base_range):
        ctx = self.data.get(symbol)
        if not ctx or len(ctx['price_history']) < 60: return 1.0
        price_range = max(ctx['price_history']) - min(ctx['price_history'])
        return max(0.66, min(price_range / base_range, 1.50))

    def is_at_low_extreme(self, symbol, current_price, threshold_points=3.0):
        ctx = self.data.get(symbol)
        if not ctx or ctx['session_low'] == float('inf'): return False
        return (current_price - ctx['session_low']) <= threshold_points

    def is_at_high_extreme(self, symbol, current_price, threshold_points=3.0):
        ctx = self.data.get(symbol)
        if not ctx or ctx['session_high'] == -float('inf'): return False
        return (ctx['session_high'] - current_price) <= threshold_points

# ==========================================
# 2. PYTHON EXECUTION WRAPPER
# ==========================================
class ExecutionWrapper:
    """Safely routes trades directly to the Rust Gatekeeper via shared memory."""
    def __init__(self, rust_engine, mode="LIVE"):
        self.engine = rust_engine
        self.mode = mode
        
    def send_signal(self, symbol, qty, side, price, sl, tp, order_type="MARKET", limit_price=0.0):
        if self.mode == "BACKTEST":
            return 
        
        try:
            # Route order into the Rust memory space where Gatekeeper picks it up
            self.engine.send_signal(symbol, qty, side, price, sl, tp)
        except Exception as e:
            print(f"\n[Bridge Error] Failed to route order to Rust engine: {e}")

# ==========================================
# 3. THE APEX ENGINE (HEADLESS READY)
# ==========================================
def main(MODE="LIVE"):
    print(f"[System] Initializing EQIE_PHIDIAS2 Apex Engine (Mode: {MODE})...")
    
    # Initialize data feed client
    client = eqie_core.QuantowerClient()

    # Initialize direct-to-Rithmic Rust engine
    rust_engine = eqie_core.create_execution_engine()
    shared_engine = ExecutionWrapper(rust_engine, mode=MODE)
    
    historical_ticks = None
    total_ticks = 0
    tick_idx = 0
    simulated_time = None
    
    if MODE == "LIVE":
        try:
            client.connect("192.168.64.4", 21000) 
            print("[System] Socket connected. Network gateway established.")
        except Exception as e:
            print(f"[CRITICAL] Failed to connect: {e}")
            return
    else:
        try:
            df = pd.read_csv("historical_tick_data.csv")
            historical_ticks = df[['Price', 'Volume', 'Side']].to_numpy()
            total_ticks = len(historical_ticks)
            simulated_time = datetime.datetime.now().replace(hour=9, minute=0, second=0)
            print(f"[Sandbox] Simulating {total_ticks:,} historical ticks through the live matrix...")
        except Exception as e:
            print(f"[Sandbox Error] Failed to load historical data: {e}")
            return

    # Asset Dictionaries & Initialization
    SYMBOLS = ["/NQU26:XCME", "/ESU26:XCME", "/MNQU26:XCME", "/MCLU26:XCME", "/MGCZ26:XCME"]
    market_context = MarketContext(SYMBOLS)
    
    # --- AUTOMATED OVERNIGHT CALIBRATION --
    if MODE == "LIVE":
        onh, onl = fetch_overnight_levels()
        
        market_context.data["/NQU26:XCME"]['overnight_high'] = onh
        market_context.data["/NQU26:XCME"]['overnight_low'] = onl
        
        if onh != -float('inf'):
            print(f"[System] Overnight Levels Locked | High: {onh:.2f} | Low: {onl:.2f}")
            
    elif MODE == "BACKTEST":
        # Keep the static spoof levels for historical sandbox runs
        market_context.data["/NQU26:XCME"]['overnight_high'] = 30690.00
        market_context.data["/NQU26:XCME"]['overnight_low'] = 30660.00
    
    # Broadcast dynamic subscription requests via the Rust method
    print("[System] Registering asset subscription matrix with headless router...")
    for sym in SYMBOLS:
        try:
            client.subscribe(sym)
            print(f"[System] Subscription verified and sent for: {sym}")
        except Exception as e:
            print(f"[System] Fatal Error during contract subscription for {sym}: {e}")
            sys.exit(1)
            
    # Asset Specific Parameters
    ASSET_PROFILES = {
        "/NQU26:XCME": {"mult": 20.0, "base_stop": 4.50, "scale_out": 6.00, "detach": 5.00, "base_vol": 5.00, "max_stretch": 60.0},
        "/ESU26:XCME": {"mult": 50.0, "base_stop": 2.50, "scale_out": 4.00, "detach": 3.00, "base_vol": 3.00, "max_stretch": 10.0},
        "/MNQU26:XCME": {"mult": 2.0, "base_stop": 4.50, "scale_out": 6.00, "detach": 5.00, "base_vol": 5.00, "max_stretch": 60.0},
        "/MCLU26:XCME": {"mult": 100.0, "base_stop": 0.25, "scale_out": 0.40, "detach": 0.20, "base_vol": 0.30, "max_stretch": 1.00},
        "/MGCZ26:XCME": {"mult": 10.0, "base_stop": 4.50, "scale_out": 6.00, "detach": 5.00, "base_vol": 5.00, "max_stretch": 15.0} 
    }
    
    # NEW: Instantiate the Performance Evaluator
    performance = PerformanceTracker()
    
    # Absolute State Variables
    STARTING_BALANCE = 50000.00
    PROFIT_TARGET = 53000.00          
    DAILY_PROFIT_CAP = 1000.00        
    ROUND_TRIP_FEE = 1.50
    
    # --- TRADEDAY EOD DRAWDOWN VARIABLES ---
    INTRADAY_HIGH_WATER_MARK = STARTING_BALANCE
    DRAWDOWN_FLOOR = STARTING_BALANCE - 2000.00

    # Execution State
    net_balance = STARTING_BALANCE
    exec_state = "FLAT"
    active_symbol = None
    active_strategy = None
    position_qty = 0
    has_scaled_out = False
    
    # NEW: Overlay Tracking
    is_hedged = False
    
    entry_price = 0.0
    stop_loss_price = 0.0
    highest_price_seen = 0.0
    lowest_price_seen = 0.0

    # System Tracking Variables
    net_equity = 0.0
    pending_timestamp = 0.0
    last_ping_time = time.time()
    last_exit_time = 0.0  
    prev_price_check = {s: 0.0 for s in SYMBOLS}
    consecutive_losses = {s: 0 for s in SYMBOLS}
    cooldown_until = 0.0

    # NEW: Tracks if a structural level has already been consumed today
    overnight_high_broken = {s: False for s in SYMBOLS}
    overnight_low_broken = {s: False for s in SYMBOLS}

    
    highest_z_seen = 0.0
    lowest_z_seen = 0.0
    
    # Time Filters (Double Window - CDT)
    S1_START = datetime.time(8, 25, 0)
    S1_END = datetime.time(12, 0, 0)
    S2_START = datetime.time(13, 0, 0)   
    S2_END = datetime.time(15, 0, 0)     
    
    # NEW: Stand down on breakouts until the market stabilizes post-open
    BREAKOUT_ALLOWED_FROM = datetime.time(9, 15, 0)     

    print(f"[System] Handshake complete. Engine is live and hunting.")

    try:
        while True:
            # --- 0. DATA INGESTION TRIBUTARY ---
            current_imbalance = 0.0
            if MODE == "LIVE":
                current_time = time.time()
                central = pytz.timezone('US/Central')
                now = datetime.datetime.now(central)
                if current_time - last_ping_time > 1.0:
                    client.send_ping()
                    last_ping_time = current_time

                tick_data = client.read_next_tick()
                if not tick_data:
                    time.sleep(0.001)
                    continue
            else:
                if tick_idx >= total_ticks:
                    # NEW: Force close any open positions on the final tick of the backtest
                    if position_qty > 0:
                        net_balance += (net_equity - (ROUND_TRIP_FEE * position_qty))
                        print(f"\n[Sandbox] Data exhausted. Auto-flattening open position. Realized PnL: ${(net_equity - (ROUND_TRIP_FEE * position_qty)):.2f}")
                    print(f"\n[Sandbox] Backtest complete. Final Balance: ${net_balance:.2f}")
                    break
                
                # Advance simulated clock by 10ms per tick
                simulated_time += datetime.timedelta(milliseconds=10)
                now = simulated_time
                current_time = now.timestamp()
                
                row = historical_ticks[tick_idx]
                tick_idx += 1
                
                symbol_raw = "/NQU26:XCME"
                price, size, side_val = row[0], row[1], row[2]
                side_str = "BUY" if side_val == 1.0 else "SELL"
                
                tick_data = client.process_backtest_tick(symbol_raw, price, size, side_str)

            current_clock = now.time()
            
            symbol, price, size, vwap, val, vah, delta, z_score, current_imbalance = tick_data
            if symbol not in SYMBOLS: continue
            
            current_time_str = now.strftime("%H:%M:%S")
            profile = ASSET_PROFILES[symbol]
            
            # Update Context Memory
            vol_mod = market_context.get_dynamic_volatility_multiplier(symbol, profile["base_vol"])
            dyn_stop = round(max(2.00, min(profile["base_stop"] * vol_mod, 6.00)) * 4) / 4
            dyn_detach = round(max(3.00, min(profile["detach"] * vol_mod, 8.00)) * 4) / 4
            vwap_slope = market_context.get_vwap_slope(symbol)

            # --- 1. STATE MACHINE RESOLUTION (LIMIT ORDER MANAGER) ---
            if exec_state.startswith("PENDING") and active_symbol == symbol:
                time_in_pending = current_time - pending_timestamp
                
                # A. The Pending Fill Simulation (Waiting for the tape to cross our limit price)
                # In a live environment, if the traded price crosses our limit price, we are filled.
                is_filled = False
                if exec_state == "PENDING_LONG" and price <= entry_price:
                    is_filled = True
                elif exec_state == "PENDING_SHORT" and price >= entry_price:
                    is_filled = True
                elif exec_state == "PENDING_EXIT":
                    # Exits are usually immediate or we assume filled for safety during EOD
                    is_filled = True 

                # B. Order Filled Successfully
                if is_filled and time_in_pending > 0.1: 
                    if exec_state == "PENDING_LONG": 
                        exec_state = "LONG"
                        highest_z_seen = z_score
                        print(f"\n[{current_time_str}] ✅ [FILL CONFIRMED] LONG {position_qty}x {symbol} @ {entry_price:.2f}")
                    elif exec_state == "PENDING_SHORT": 
                        exec_state = "SHORT"
                        lowest_z_seen = z_score
                        print(f"\n[{current_time_str}] ✅ [FILL CONFIRMED] SHORT {position_qty}x {symbol} @ {entry_price:.2f}")
                    elif exec_state == "PENDING_EXIT":
                        exec_state = "FLAT"
                        active_symbol = None
                        last_exit_time = current_time
                        
                # C. The Order Chaser (Timeout Mechanism)
                # If 1.5 seconds pass and we haven't been filled, the market is running away. Chase it.
                elif not is_filled and time_in_pending > 1.5 and exec_state in ["PENDING_LONG", "PENDING_SHORT"]:
                    chase_side = "BUY" if exec_state == "PENDING_LONG" else "SELL"
                    print(f"\n[{current_time_str}] ⚠️ [ORDER CHASER] Limit order untouched. Chasing with MARKET order at {price:.2f}.")
                    
                    # Fire a market order to guarantee entry
                    shared_engine.send_signal(active_symbol, position_qty, chase_side, price, 0.0, 0.0, order_type="MARKET")
                    
                    # Update our official entry price to the worse price we just paid
                    entry_price = price 
                    
                    # Force the state to active
                    exec_state = "LONG" if exec_state == "PENDING_LONG" else "SHORT"
                    if exec_state == "LONG": highest_z_seen = z_score
                    else: lowest_z_seen = z_score

            # --- 2. UPDATE TRAILING DRAWDOWN (NEW PLACEMENT) ---
            current_account_value = net_balance + net_equity
            daily_pnl = current_account_value - STARTING_BALANCE

            if current_account_value > INTRADAY_HIGH_WATER_MARK:
                INTRADAY_HIGH_WATER_MARK = current_account_value
                DRAWDOWN_FLOOR = INTRADAY_HIGH_WATER_MARK - 2000.00

            # --- 3. GUARDIAN CONDITIONS (TRADEDAY EOD ALIGNED) ---
            # A. The Portfolio Kill Switch (Enforces the real-time EOD trailing floor)
            if current_account_value <= DRAWDOWN_FLOOR:
                if position_qty > 0 and exec_state != "PENDING_EXIT":
                    shared_engine.send_signal(active_symbol, position_qty, "SELL" if exec_state == "LONG" else "BUY", price, 0.0, 0.0)
                print(f"\n[{current_time_str}] 💀 [FATAL] TRADEDAY EOD DRAWDOWN FLOOR TOUCHED (${DRAWDOWN_FLOOR:,.2f}).")
                print(f"     ➔ Current Account Value: ${current_account_value:,.2f} | Emergency Flatten command deployed.")
                sys.exit(0)

            # B. The Target Evaluation Target Check
            if current_account_value >= PROFIT_TARGET:
                if position_qty > 0 and exec_state != "PENDING_EXIT":
                    shared_engine.send_signal(active_symbol, position_qty, "SELL" if exec_state == "LONG" else "BUY", price, 0.0, 0.0)
                print(f"\n[{current_time_str}] 🏆 [VICTORY] EVALUATION PASSED! Profit Target reached: ${current_account_value:,.2f}")
                sys.exit(0)

            # --- 4. SESSION MANAGEMENT ---
            is_active_session = (S1_START <= current_clock <= S1_END) or (S2_START <= current_clock <= S2_END)
            
            if not is_active_session and position_qty > 0 and exec_state != "PENDING_EXIT" and active_symbol == symbol:
                
                # NEW: Drop the shield before End of Day liquidation
                if is_hedged:
                    hedge_side = "BUY" if exec_state == "LONG" else "SELL"
                    shared_engine.send_signal("/MNQU26:XCME", 10, hedge_side, price, 0.0, 0.0)
                    is_hedged = False
                    print(f"\n[{current_time_str}] 🛡️ [OVERLAY CLEANUP] Dropping Micro-Hedge prior to EOD Flatten.")

                shared_engine.send_signal(active_symbol, position_qty, "SELL" if exec_state == "LONG" else "BUY", price, 0.0, 0.0)
                exec_state = "PENDING_EXIT"
                pending_timestamp = current_time
                net_balance += (net_equity - (ROUND_TRIP_FEE * position_qty))
                position_qty = 0
                print(f"\n[{current_time_str}] 🛑 EOD FLATTEN | Session Closed.")
                continue 

            # --- 5. UI UPDATES (THROTTLED) ---
            status_prefix = f"[{MODE}]" if is_active_session else "[SLEEP]"
            if active_symbol == symbol or active_symbol is None:
                # Throttle LIVE updates to roughly twice per second to prevent terminal freezing
                if MODE == "LIVE" and (current_time - getattr(market_context, 'last_ui_update', 0) > 0.5):
                    sys.stdout.write(f"\r{status_prefix} {symbol} | Pos: {position_qty} | PnL: ${daily_pnl:+.2f} | Z: {z_score:+.2f} | State: {exec_state}    ")
                    sys.stdout.flush()
                    market_context.last_ui_update = current_time
                # Keep BACKTEST fast-forward UI
                elif MODE == "BACKTEST" and tick_idx % 1000 == 0:
                    sys.stdout.write(f"\r{status_prefix} {symbol} | Pos: {position_qty} | PnL: ${daily_pnl:+.2f} | Z: {z_score:+.2f} | State: {exec_state}    ")
                    sys.stdout.flush()

            # --- 6. OFFENSIVE TRADE MANAGEMENT & SCALE-OUT (ACTIVE ASSET ONLY) ---
            if active_symbol == symbol and exec_state in ["LONG", "SHORT"]:
                trigger_full_exit = False
                exit_reason = ""

                if exec_state == "LONG": 
                    highest_price_seen = max(highest_price_seen, price)
                    highest_z_seen = max(highest_z_seen, z_score)

                    # TRIGGER SHIELD: Violent downside order flow detected against our long position
                    if z_score <= -3.0 and delta < -100 and not is_hedged:
                        print(f"\n[{current_time_str}] 🛡️ [OVERLAY TRIGGERED] Flash crash detected! Deploying Micro-Hedge.")
                        # Short 10 Micro NQ contracts to perfectly offset 1 Standard NQ contract
                        shared_engine.send_signal("/MNQU26:XCME", 10, "SELL", price, 0.0, 0.0) 
                        is_hedged = True
                    
                    # DROP SHIELD: The panic subsides, order flow normalizes
                    elif is_hedged and z_score > -1.0 and delta > -20:
                        print(f"\n[{current_time_str}] 🕊️ [OVERLAY LIFTED] Volatility subsided. Unwinding Micro-Hedge.")
                        # Buy back the 10 Micros to un-pause the main PnL
                        shared_engine.send_signal("/MNQU26:XCME", 10, "BUY", price, 0.0, 0.0)
                        is_hedged = False
                    
                    if not has_scaled_out and position_qty == 2 and highest_price_seen >= entry_price + profile["scale_out"]:
                        shared_engine.send_signal(symbol, 1, "SELL", price, 0.0, 0.0)
                        position_qty = 1
                        has_scaled_out = True
                        stop_loss_price = entry_price + 1.00 
                        scale_profit = (profile["scale_out"] * profile["mult"]) - ROUND_TRIP_FEE
                        net_balance += scale_profit
                        print(f"\n[{current_time_str}] 💰 [SCALE OUT] 1 {symbol} Contract secured at +{profile['scale_out']} pts. Runner stop at {stop_loss_price:.2f}")

                    if has_scaled_out and highest_price_seen >= entry_price + dyn_detach:
                        proposed_stop = min(max(stop_loss_price, vah), highest_price_seen - 2.50)
                        if proposed_stop > stop_loss_price: stop_loss_price = proposed_stop
                    
                    # Custom Exit: Mean Reversion Target Hit
                    if active_strategy == "MEAN_REVERSION" and price >= ctx['ema_21']:
                        trigger_full_exit = True
                        exit_reason = "Mean_Target_Reached"

                    if price <= stop_loss_price:
                        trigger_full_exit = True
                        exit_reason = "Trailing_Stop_Or_Breakeven"
                    elif highest_price_seen >= entry_price + dyn_detach and highest_z_seen >= 3.0 and (highest_z_seen - z_score) >= 1.75:
                        trigger_full_exit = True
                        exit_reason = "Velocity_Exhaustion"

                elif exec_state == "SHORT":
                    lowest_price_seen = min(lowest_price_seen, price)
                    lowest_z_seen = min(lowest_z_seen, z_score)

                    # TRIGGER SHIELD: Violent upside squeeze against our short position
                    if z_score >= 3.0 and delta > 100 and not is_hedged:
                        print(f"\n[{current_time_str}] 🛡️ [OVERLAY TRIGGERED] Upside squeeze detected! Deploying Micro-Hedge.")
                        # Long 10 Micro NQ contracts
                        shared_engine.send_signal("/MNQU26:XCME", 10, "BUY", price, 0.0, 0.0) 
                        is_hedged = True
                    
                    # DROP SHIELD: The squeeze exhausts
                    elif is_hedged and z_score < 1.0 and delta < 20:
                        print(f"\n[{current_time_str}] 🕊️ [OVERLAY LIFTED] Volatility subsided. Unwinding Micro-Hedge.")
                        shared_engine.send_signal("/MNQU26:XCME", 10, "SELL", price, 0.0, 0.0)
                        is_hedged = False
                    
                    if not has_scaled_out and position_qty == 2 and lowest_price_seen <= entry_price - profile["scale_out"]:
                        shared_engine.send_signal(symbol, 1, "BUY", price, 0.0, 0.0)
                        position_qty = 1
                        has_scaled_out = True
                        stop_loss_price = entry_price - 1.00 
                        scale_profit = (profile["scale_out"] * profile["mult"]) - ROUND_TRIP_FEE
                        net_balance += scale_profit
                        print(f"\n[{current_time_str}] 💰 [SCALE OUT] 1 {symbol} Contract secured at +{profile['scale_out']} pts. Runner stop at {stop_loss_price:.2f}")

                    if has_scaled_out and lowest_price_seen <= entry_price - dyn_detach:
                        proposed_stop = max(min(stop_loss_price, val), lowest_price_seen + 2.50)
                        if proposed_stop < stop_loss_price: stop_loss_price = proposed_stop

                    # Custom Exit: Mean Reversion Target Hit
                    if active_strategy == "MEAN_REVERSION" and price <= ctx['ema_21']:
                        trigger_full_exit = True
                        exit_reason = "Mean_Target_Reached"

                    if price >= stop_loss_price:
                        trigger_full_exit = True
                        exit_reason = "Trailing_Stop_Or_Breakeven"
                    elif lowest_price_seen <= entry_price - dyn_detach and lowest_z_seen <= -3.0 and (z_score - lowest_z_seen) >= 1.75:
                        trigger_full_exit = True
                        exit_reason = "Velocity_Exhaustion"

                if trigger_full_exit:
                    # Drop the shield if it's currently active before closing the main position!
                    if is_hedged:
                        hedge_side = "BUY" if exec_state == "LONG" else "SELL"
                        shared_engine.send_signal("/MNQU26:XCME", 10, hedge_side, price, 0.0, 0.0)
                        is_hedged = False
                        print(f"\n[{current_time_str}] 🛡️ [OVERLAY CLEANUP] Dropping Micro-Hedge prior to full exit.")

                    shared_engine.send_signal(symbol, position_qty, "SELL" if exec_state == "LONG" else "BUY", price, 0.0, 0.0)
                    exec_state = "PENDING_EXIT"
                    pending_timestamp = current_time
                    net_pnl = net_equity - (ROUND_TRIP_FEE * position_qty)
                    net_balance += net_pnl
                    position_qty = 0
                    active_strategy = None
                    cooldown_until = current_time + 300 
                    
                    # Log the trade to the Performance Evaluator
                    performance.log_trade(net_pnl)
                    
                    # Update the Circuit Breaker
                    if net_pnl < 0:
                        consecutive_losses[symbol] += 1
                    else:
                        consecutive_losses[symbol] = 0
                        
                    current_sharpe = performance.get_trade_sharpe()
                    win_rate = performance.get_win_rate()
                    f1_score = performance.get_f1_score() # NEW: Fetch F1-Score

                    print(f"\n[{current_time_str}] ⚪ {symbol} CLOSED | {exit_reason} | PnL: ${net_pnl:+.2f}")
                    # NEW: Print F1-Score to the terminal
                    print(f"     ➔ [SYSTEM METRICS] Win Rate: {win_rate:.1f}% | Trade Sharpe: {current_sharpe:.2f} | F1-Score: {f1_score:.3f}")

            # --- 7. GLOBAL ENTRY LOGIC (MULTI-STRATEGY MATRIX) ---
            # Veto entry if this specific asset has hit the circuit breaker (3 consecutive losses)
            if exec_state == "FLAT" and active_symbol is None and is_active_session and current_time > cooldown_until and consecutive_losses[symbol] < 3:
                
                ctx = market_context.data[symbol]
                
                # --- PROBABILISTIC REGIME CLASSIFIER (ASSET-NORMALIZED) ---
                # 1. Calculate the Asset Normalization Multiplier
                # Anchor to NQ's base_vol of 5.00. 
                # (e.g., NQ = 1.0, ES = 0.6, MCL = 0.06)
                asset_multiplier = profile["base_vol"] / 5.00
                
                # 2. Calculate Streaming Metrics
                recent_prices = list(ctx['price_history'])[-30:]
                local_range = max(recent_prices) - min(recent_prices) if len(recent_prices) == 30 else (5.0 * asset_multiplier)
                
                # 3. Dynamic Inflections & Steepness (k)
                # If the threshold shrinks, the curve steepness must increase inversely
                inflection_high_vol = 12.0 * asset_multiplier
                k_high_vol = 0.8 / asset_multiplier
                
                inflection_chop = 5.0 * asset_multiplier
                k_chop = 1.2 / asset_multiplier
                
                # --- STATE VARIABLES & LOGIC INITIALIZATION ---
                ribbon_bullish = ctx['ema_9'] > ctx['ema_21'] > ctx['ema_50']
                ribbon_bearish = ctx['ema_9'] < ctx['ema_21'] < ctx['ema_50']
                # Increased from 2.0 to 3.5 to allow the engine to ride violent trend momentum
                is_overextended = abs(z_score) >= 2.25

                # 4. Logistic Mapping (Calculate probabilities from 0.0 to 1.0)
                prob_high_vol = 1 / (1 + math.exp(-k_high_vol * (local_range - inflection_high_vol)))
                prob_consolidation = 1 / (1 + math.exp(k_chop * (local_range - inflection_chop)))
                
                # Trend: Peaks when neither chopping nor exploding
                trend_alignment = 1.0 if (ribbon_bullish and vwap_slope > 0) or (ribbon_bearish and vwap_slope < 0) else 0.0
                prob_trend = max(0.0, trend_alignment - prob_high_vol - prob_consolidation)

                # 5. Dynamic Parameter Scaling (Weighted by Probability & Live Flow)
                # Instead of static limits, scale the requirements directly against the live Z-score variance
                live_volatility_scalar = max(1.0, abs(z_score))
                
                # Require higher immediate momentum (Z-Score) to enter trades during high volatility
                # Lowered standard deviation thresholds to allow execution in standard tape conditions
                dyn_z_req = (prob_consolidation * 0.35) + (prob_trend * 0.75) + (prob_high_vol * 1.25)
                
                base_tolerance = 4.5 * asset_multiplier # Increased from 3.5
                min_tolerance = 3.0 * asset_multiplier
                max_tolerance = 8.0 * asset_multiplier

                # Add a hard veto: If the combined probability of chop is too high, abort.
                if prob_consolidation > 0.85:
                    signal_type = None # Veto all entries, the market is a woodchipper today.
                
                # Expand the structural tolerance when the tape is moving fast, shrink it when dead
                retest_tolerance = ((prob_consolidation * min_tolerance) + (prob_trend * base_tolerance) + (prob_high_vol * max_tolerance)) * (live_volatility_scalar * 0.5)

                # 6. Primary State Assignment for Strategy Locking
                if prob_high_vol > 0.65:
                    regime = "HIGH_VOLATILITY"
                elif prob_consolidation > 0.60:
                    regime = "CONSOLIDATION"
                else:
                    regime = "TRENDING"
                
                # Adaptive Stabilization (Unleashed for the morning rush)
                # Let the first 1 to 5 minutes of the 8:30 AM open settle, then start hunting immediately
                if regime == "HIGH_VOLATILITY" and prob_high_vol > 0.85:
                    is_stabilized = current_clock >= datetime.time(8, 35, 0)
                elif regime == "HIGH_VOLATILITY":
                    is_stabilized = current_clock >= datetime.time(8, 32, 0)
                else:
                    is_stabilized = current_clock >= datetime.time(8, 31, 0)
                
                # Signal Triggers
                signal_type = None
                signal_side = None

                # 1. Validate that overnight levels actually exist
                valid_overnight = ctx['overnight_high'] != -float('inf') and ctx['overnight_low'] != float('inf')

                # ---------------------------------------------------------
                # STRATEGY 1: Pure Momentum Breakout
                # ---------------------------------------------------------
                breakout_buffer = 1.25 # Allow execution closer to the actual liquidity pool
                if is_stabilized and regime != "CONSOLIDATION" and valid_overnight:
                    if not overnight_high_broken[symbol] and price > (ctx['overnight_high'] + breakout_buffer):
                        if z_score > dyn_z_req: # Now tracks immediate velocity, not session delta
                            signal_type, signal_side = "OVERNIGHT_BREAKOUT", "BUY"
                            overnight_high_broken[symbol] = True 
                            
                    elif not overnight_low_broken[symbol] and price < (ctx['overnight_low'] - breakout_buffer):
                        if z_score < -dyn_z_req: 
                            signal_type, signal_side = "OVERNIGHT_BREAKDOWN", "SELL"
                            overnight_low_broken[symbol] = True

                # ---------------------------------------------------------
                # STRATEGY 2: Trend Following (DYNAMIC)
                # ---------------------------------------------------------
                elif is_stabilized and regime != "CONSOLIDATION" and not is_overextended:
                    macro_bullish = ribbon_bullish and vwap_slope > 0.5
                    macro_bearish = ribbon_bearish and vwap_slope < -0.5
                    
                    # Changed 'price > vwap' to '>=' to catch exact slope rides
                    if macro_bullish and price >= vwap and z_score > (dyn_z_req * 0.8):
                        signal_type, signal_side = "RIBBON_TREND", "BUY"
                    elif macro_bearish and price <= vwap and z_score < -(dyn_z_req * 0.8):
                        signal_type, signal_side = "RIBBON_TREND", "SELL"

                # ---------------------------------------------------------
                # STRATEGY 3: Structural Pullback (Adaptive VWAP Bounce)
                # ---------------------------------------------------------
                # Fixed the `0 <` to `0 <=` so the engine executes exactly on the VWAP line
                elif vwap_slope > 0.0 and 0 <= (price - vwap) <= (retest_tolerance * 1.5) and z_score > 0.5:
                    signal_type, signal_side = "PULLBACK_BOUNCE", "BUY"
                elif vwap_slope < 0.0 and 0 <= (vwap - price) <= (retest_tolerance * 1.5) and z_score < -0.5:
                    signal_type, signal_side = "PULLBACK_REJECT", "SELL"

                # ---------------------------------------------------------
                # STRATEGY 4: Statistical Arbitrage (Relative Value Spread)
                # ---------------------------------------------------------
                elif market_context.current_spread_z_score >= 2.5:
                    signal_type, signal_side = "STAT_ARB_SPREAD", ("SELL" if symbol == "/NQU26:XCME" else "BUY")
                    qty_to_trade = 2 if symbol == "/NQU26:XCME" else market_context.dynamic_es_qty
                    
                elif market_context.current_spread_z_score <= -2.5:
                    signal_type, signal_side = "STAT_ARB_SPREAD", ("BUY" if symbol == "/NQU26:XCME" else "SELL")
                    qty_to_trade = 2 if symbol == "/NQU26:XCME" else market_context.dynamic_es_qty

                # ---------------------------------------------------------
                # STRATEGY 5: Mean Reversion (The Chop / Rubber-Band Snap)
                # ---------------------------------------------------------
                elif is_stabilized and (regime == "CONSOLIDATION" or is_overextended):
                    if price > ctx['ema_50'] and z_score <= -1.5: # Removed static delta
                        signal_type, signal_side = "MEAN_REVERSION", "SELL"
                    elif price < ctx['ema_50'] and z_score >= 1.5:
                        signal_type, signal_side = "MEAN_REVERSION", "BUY"

                # ---------------------------------------------------------
                # STRATEGY 6: Price Action Absorption (L1 Compatible Iceberg Fade)
                # ---------------------------------------------------------
                elif is_stabilized and regime != "TRENDING":
                    if z_score > 2.5 and vwap_slope <= 0.0 and price <= prev_price_check[symbol] + 0.25: 
                        signal_type, signal_side = "ABSORPTION_FADE", "SELL"
                    elif z_score < -2.5 and vwap_slope >= 0.0 and price >= prev_price_check[symbol] - 0.25:
                        signal_type, signal_side = "ABSORPTION_FADE", "BUY"

                # ---------------------------------------------------------
                # STRATEGY 7: VPOC Magnetism (Volume Node Mean Reversion)
                # ---------------------------------------------------------
                elif is_stabilized and regime == "CONSOLIDATION":
                    if price > (vah + (5.0 * asset_multiplier)) and z_score < -1.5:
                        signal_type, signal_side = "VPOC_MAGNET", "SELL"
                    elif price < (val - (5.0 * asset_multiplier)) and z_score > 1.5:
                        signal_type, signal_side = "VPOC_MAGNET", "BUY"

                # ---------------------------------------------------------
                # EXECUTION DISPATCHER & TIERED RISK SCALER
                # ---------------------------------------------------------
                # Proceed directly to risk tiering if a signal was generated
                if signal_type:
                    # 1. Evaluate Setup Conviction (Risk Tiering)
                    if signal_type == "STAT_ARB_SPREAD":
                        target_symbol = symbol
                        target_qty = qty_to_trade
                        tier_label = "[TIER 1: ARBITRAGE]"
                    else:
                        if regime == "CONSOLIDATION":
                            risk_multiplier = 0.50
                            tier_label = "[TIER: CONSOLIDATION SCALP]"
                        elif regime == "HIGH_VOLATILITY":
                            risk_multiplier = 0.25
                            tier_label = "[TIER: HIGH VOL SHIELD]"
                        else:
                            risk_multiplier = max(0.25, prob_trend)
                            tier_label = f"[TIER: TREND ALIGNED ({prob_trend * 100:.0f}%)]"
                            
                            # B. Assign Target Asset and Target Quantity based on Risk Multiplier
                            if symbol == "/NQU26:XCME":
                                if risk_multiplier >= 0.75:
                                    target_symbol = symbol
                                    target_qty = 2  # High confidence: Trade standard Minis
                                else:
                                    target_symbol = "/MNQU26:XCME"
                                    # Low confidence: Downgrade to Micros and scale exact size (1 to 10)
                                    target_qty = max(1, int(10 * risk_multiplier))
                                    
                            elif symbol == "/ESU26:XCME":
                                target_symbol = symbol
                                # 2 Minis if confident, 1 Mini if uncertain
                                target_qty = 2 if risk_multiplier >= 0.75 else 1
                                
                            else:
                                # Default scaling for standard Micros (MNQ, MCL, MGC)
                                # Assumes a base max allocation of 4 units
                                target_symbol = symbol
                                target_qty = max(1, int(4 * risk_multiplier))

                        # 2. Fire the Routed Order (Passive Execution)
                        # 2. Fire the Routed Order (Adaptive Execution)
                        if prev_price_check[symbol] == 0.0 or (signal_side == "BUY" and price > prev_price_check[symbol]) or (signal_side == "SELL" and price < prev_price_check[symbol]):
                            
                            # Determine the appropriate limit price based on live tape velocity.
                            tick_size = 0.25 if target_symbol == "/ESU26:XCME" else (0.01 if target_symbol == "/MCLU26:XCME" else (0.10 if target_symbol == "/MGCZ26:XCME" else 0.50))
                            
                            # Calculate aggressiveness based on the standard deviation of real-time volume velocity
                            aggression_ticks = math.ceil(abs(z_score) * 1.5)
                            
                            if regime == "HIGH_VOLATILITY" or signal_type in ["OVERNIGHT_BREAKOUT", "OVERNIGHT_BREAKDOWN"]:
                                # Maximum Aggression: Cross the spread deeply to guarantee a fill on breakouts
                                limit_price = price + (tick_size * aggression_ticks) if signal_side == "BUY" else price - (tick_size * aggression_ticks)
                            elif regime == "TRENDING":
                                # Standard Aggression: Pay the spread + 1 tick
                                limit_price = price + (tick_size * 2) if signal_side == "BUY" else price - (tick_size * 2)
                            else:
                                # Passive Execution: Sit on the bid/ask during absorption or chop
                                limit_price = price
                            
                            # Apply stops relative to our intended limit price
                            dyn_stop_calc = limit_price - dyn_stop if signal_side == "BUY" else limit_price + dyn_stop
                            dyn_tp_calc = limit_price + 15.0 if signal_side == "BUY" else limit_price - 15.0
                            
                            # Dispatch the explicit LIMIT order
                            shared_engine.send_signal(
                                target_symbol, target_qty, signal_side, limit_price, dyn_stop_calc, dyn_tp_calc, 
                                order_type="LIMIT", limit_price=limit_price
                            )  
                            
                            exec_state = "PENDING_LONG" if signal_side == "BUY" else "PENDING_SHORT"
                            active_symbol = target_symbol 
                            active_strategy = signal_type
                            position_qty = target_qty
                            has_scaled_out = False
                            pending_timestamp = current_time
                            
                            # Our entry price is now the limit price we are hoping to get, not a market execution
                            entry_price = limit_price
                            
                            if signal_side == "BUY":
                                highest_price_seen = limit_price
                                stop_loss_price = dyn_stop_calc
                                print(f"\n[{current_time_str}] 🟢 {target_symbol} {tier_label} | {signal_type} LONG ({target_qty}x) | POSTING LIMIT @ {limit_price:.2f}")
                            else:
                                lowest_price_seen = limit_price
                                stop_loss_price = dyn_stop_calc
                                print(f"\n[{current_time_str}] 🔴 {target_symbol} {tier_label} | {signal_type} SHORT ({target_qty}x) | POSTING LIMIT @ {limit_price:.2f}")

            prev_price_check[symbol] = price

    except KeyboardInterrupt:
        print("\n[System] Manual shutdown received.")
    except Exception as e:
        print(f"\n[System] Fatal Error: {e}")
    finally:
        print("[System] Engine offline.")

# ==========================================
# 4. EXECUTION ROUTER (THE HYBRID TOGGLE)
# ==========================================
if __name__ == "__main__":
    # Change MODE to "LIVE" for weekday execution or "BACKTEST" for weekend simulation
    MODE = "LIVE" 
    main(MODE)

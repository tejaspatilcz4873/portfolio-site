# app_flipkart_qc.py
import streamlit as st
import numpy as np, pandas as pd, random, math, time, io
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Flipkart Q-Commerce Simulator")

# ---------------------------
# CONFIG / ASSUMPTIONS (Express optimized, zone-bound riders)
# ---------------------------
CITY_DEFAULTS = {
    "Delhi-NCR": {"orders_per_day": 10000, "dark_stores": 80, "mean_travel_min": 10.0},
    "Pune": {"orders_per_day": 2000, "dark_stores": 25, "mean_travel_min": 8.0},
}

ASSUMPTIONS = {
    "handling_time_min": 2.0,
    "relocation_penalty_min": 20,
    "rider_cost_per_hr": 120.0,
    "simulation_horizon_hrs": 24,
    "batching_window_mins": 10,
    "batch_travel_savings": 0.30,
    # Express speedups (operational)
    "express_travel_speedup": 0.60,
    "express_handling_speedup": 0.60,
    "express_relocation_factor": 0.5
}

# fixed simulation params (no UI)
FIXED_DISPATCH_EPOCH_MIN = 1
FIXED_RANDOM_SEED = 42

def bimodal_hourly_profile():
    hours = np.arange(24)
    peak1 = np.exp(-0.5*((hours-9)/1.8)**2)
    peak2 = np.exp(-0.5*((hours-20)/1.8)**2)
    base = 0.15 * np.ones_like(hours)
    weights = base + 2.5*peak1 + 3.5*peak2
    return (weights / weights.sum()).tolist()

HOURLY_PROFILE = bimodal_hourly_profile()

# ---------------------------
# SIM ENGINE (rolling-horizon + priority + batching)
# ---------------------------
def simulate_dispatch_streamlit(
    city="Delhi-NCR",
    total_riders=3500,
    sla_mix=None,
    fleet_sharing=True,
    demand_multiplier=1.0,
    hybrid_enabled=False,
    shift_frac=0.0,
    s_to_std=0.7,
    random_seed=FIXED_RANDOM_SEED
):
    """
    Riders are zone-bound and there is a single unified pool.
    hybrid_enabled: fraction (shift_frac) of actual Express orders are relabeled to Standard/EcoSaver.
    Shifted orders are NOT given dispatch priority; they are handled exactly like native Standard/EcoSaver.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    city_info = CITY_DEFAULTS[city]
    orders_per_day = int(city_info["orders_per_day"] * demand_multiplier)
    num_stores = city_info["dark_stores"]
    mean_travel = city_info["mean_travel_min"]
    horizon_minutes = ASSUMPTIONS["simulation_horizon_hrs"] * 60.0
    
    # UPDATED SLA PROMISES (as requested)
    SLA_PROMISE = {"Express":10.0, "Standard":20.0, "EcoSaver":30.0}

    if sla_mix is None:
        sla_mix = {"Express":0.30, "Standard":0.50, "EcoSaver":0.20}

    # create stores
    stores = [{"id": i, "demand_weight": random.uniform(0.6,1.4)} for i in range(num_stores)]
    total_w = sum(s["demand_weight"] for s in stores) or 1.0
    for s in stores:
        s["demand_prob"] = s["demand_weight"] / total_w

    # create riders (single unified pool — zone-bound)
    riders = []
    for r in range(total_riders):
        zones = [s["id"] for s in stores]
        probs = [s["demand_prob"] for s in stores]
        zone = int(np.random.choice(zones, p=probs))
        riders.append({
            "id": r,
            "zone": int(zone),
            "next_free_min": 0.0,
            "busy_minutes": 0.0,
            "assignments": 0,
            "current_task": None
        })

    # generate orders by Poisson per hour (using the provided sla_mix)
    hourly_means = [orders_per_day * w for w in HOURLY_PROFILE]
    orders = []
    oid = 0
    for hr, lam in enumerate(hourly_means):
        n = np.random.poisson(max(0.0, float(lam)))
        for _ in range(n):
            minute = hr*60 + random.uniform(0,60)
            tiers = list(sla_mix.keys()); probs = list(sla_mix.values())
            if sum(probs) <= 0:
                probs = [1.0/len(probs)] * len(probs)
            tier = np.random.choice(tiers, p=probs)
            zone = int(np.random.choice([s["id"] for s in stores], p=[s["demand_prob"] for s in stores]))
            orders.append({"id": oid, "arrive_min": minute, "zone": zone, "tier": tier, "shifted": False})
            oid += 1
    orders.sort(key=lambda x: x["arrive_min"])

    # If hybrid is enabled: randomly pick actual Express orders and re-label them as shifted->Standard/EcoSaver
    if hybrid_enabled and shift_frac > 0:
        express_orders = [o for o in orders if o["tier"] == "Express"]
        num_to_shift = int(round(len(express_orders) * shift_frac))
        if num_to_shift > 0 and express_orders:
            to_shift = random.sample(express_orders, min(num_to_shift, len(express_orders)))
            for o in to_shift:
                # decide whether to go to Standard or Eco
                if random.random() < s_to_std:
                    o["tier"] = "Standard"
                else:
                    o["tier"] = "EcoSaver"
                o["shifted"] = True

    # helpers
    def sample_travel(mean_override=None):
        m = mean_travel if mean_override is None else mean_override
        return max(1.0, random.gauss(m, m*0.25))

    pending = []
    completed = []
    last_idx = 0
    now = 0.0
    total_orders = len(orders)
    epoch = 0

    while now <= horizon_minutes and (last_idx < total_orders or pending):
        # ingest arrivals
        while last_idx < total_orders and orders[last_idx]["arrive_min"] <= now:
            pending.append(orders[last_idx]); last_idx += 1

        # clear finished tasks (rider becomes idle)
        for r in riders:
            ct = r.get("current_task")
            if ct is not None and r["next_free_min"] <= now:
                r["current_task"] = None

        # categorize pending
        express = [o for o in pending if o["tier"]=="Express"]
        standard = [o for o in pending if o["tier"]=="Standard"]
        ecosaver = [o for o in pending if o["tier"]=="EcoSaver"]

        # Express operational speedups (UPDATED)
        express_travel_mean = mean_travel * ASSUMPTIONS.get("express_travel_speedup", 1.0)
        express_handling = ASSUMPTIONS["handling_time_min"] * ASSUMPTIONS.get("express_handling_speedup", 1.0)

        # -------------------------
        # 1. EXPRESS (zone-bound, prefer idle riders, filter feasible waits)
        # -------------------------
        expr_assigned = set()
        for o in sorted(express, key=lambda x: x["arrive_min"]):
            sla = SLA_PROMISE[o["tier"]]
            # compute max wait that still allows meeting SLA on avg travel+handling
            max_wait_allowed = max(0.0, sla - (express_travel_mean + express_handling))

            # consider only riders in same zone
            zone_all_riders = [r for r in riders if r["zone"] == o["zone"]]
            if not zone_all_riders:
                continue
            idle_zone_riders = [r for r in zone_all_riders if r["next_free_min"] <= now]
            if idle_zone_riders:
                candidates = idle_zone_riders
            else:
                candidates = [r for r in zone_all_riders if (r["next_free_min"] - now) <= max_wait_allowed]
                if not candidates:
                    candidates = zone_all_riders

            best = None
            best_metric = float("inf")
            for r in candidates:
                start = max(now, r["next_free_min"])
                est_finish = start + express_travel_mean + express_handling
                est_delivery = est_finish - o["arrive_min"]
                lateness = est_delivery - sla
                if (lateness < best_metric) or (math.isclose(lateness, best_metric) and est_finish < (best and best.get("est_finish", float("inf")) or float("inf"))):
                    best_metric = lateness
                    best = {"rider": r, "est_finish": est_finish, "est_delivery": est_delivery}
            if best is not None:
                rider_obj = best["rider"]
                travel = sample_travel(express_travel_mean)
                start_time = max(now, rider_obj["next_free_min"])
                finish_time = start_time + travel + express_handling
                rider_obj["next_free_min"] = finish_time
                rider_obj["busy_minutes"] += (finish_time - start_time)
                rider_obj["assignments"] += 1
                rider_obj["zone"] = o["zone"]
                rider_obj["current_task"] = {"order_id": o["id"], "tier": o["tier"], "start_min": start_time, "end_min": finish_time, "zone": o["zone"]}
                completed.append({
                    "order_id": o["id"],
                    "arrive_min": o["arrive_min"],
                    "start_min": start_time,
                    "finish_min": finish_time,
                    "tier": o["tier"],
                    "promise_min": SLA_PROMISE[o["tier"]],
                    "delivery_min": finish_time - o["arrive_min"],
                    "met_sla": (finish_time - o["arrive_min"]) <= SLA_PROMISE[o["tier"]],
                    "assigned_rider": rider_obj["id"],
                    "relocation_min": 0.0
                })
                expr_assigned.add(o["id"])
        pending = [p for p in pending if p["id"] not in expr_assigned]

        # -------------------------
        # 2. STANDARD: priority after Express.
        # New rule: eligible if waited >= 2.0 min OR (idle rider exists in zone AND no pending Express in same zone)
        # (Changed wait threshold to 2.0 minutes as requested)
        # -------------------------
        std_to_assign = []
        for o in pending:
            if o["tier"] != "Standard":
                continue
            waited = now - o["arrive_min"]
            zone_riders = [r for r in riders if r["zone"] == o["zone"]]
            idle_zone = any(r["next_free_min"] <= now for r in zone_riders)
            express_pending_in_zone = any(e for e in express if e["zone"] == o["zone"])
            if (waited >= 2.0) or (idle_zone and not express_pending_in_zone):
                std_to_assign.append(o)
        std_to_assign = sorted(std_to_assign, key=lambda x: x["arrive_min"])

        std_assigned = set()
        for o in std_to_assign:
            zone_riders = [r for r in riders if r["zone"] == o["zone"]]
            if not zone_riders:
                continue
            best = None; best_finish = float("inf")
            for r in zone_riders:
                start = max(now, r["next_free_min"])
                est_finish = start + mean_travel + ASSUMPTIONS["handling_time_min"]
                if est_finish < best_finish:
                    best_finish = est_finish; best = r
            if best is None:
                continue
            start_time = max(now, best["next_free_min"])
            travel = sample_travel()
            finish_time = start_time + travel + ASSUMPTIONS["handling_time_min"]
            best["next_free_min"] = finish_time; best["busy_minutes"] += (finish_time - start_time); best["assignments"] += 1
            best["current_task"] = {"order_id": o["id"], "tier": o["tier"], "start_min": start_time, "end_min": finish_time, "zone": o["zone"]}
            completed.append({
                "order_id": o["id"], "arrive_min": o["arrive_min"], "start_min": start_time,
                "finish_min": finish_time, "tier": o["tier"], "promise_min": SLA_PROMISE[o["tier"]],
                "delivery_min": finish_time - o["arrive_min"], "met_sla": (finish_time - o["arrive_min"])<=SLA_PROMISE[o["tier"]],
                "assigned_rider": best["id"], "relocation_min": 0.0
            })
            std_assigned.add(o["id"])
        pending = [p for p in pending if p["id"] not in std_assigned]

        # -------------------------
        # 3. ECOSAVER batching per zone (lowest priority). Shifted eco orders receive same treatment as native eco.
        #    IMPORTANT: Do NOT dispatch EcoSaver batches in a zone if there are pending Standard orders in that same zone.
        # -------------------------
        eco_by_zone = {}
        for o in [p for p in pending if p["tier"]=="EcoSaver"]:
            eco_by_zone.setdefault(o["zone"], []).append(o)
        eco_assigned = set()
        for zone, olist in eco_by_zone.items():
            # If there are any pending Standard orders in this zone, skip eco batching for now.
            std_in_zone = any(p for p in pending if p["tier"] == "Standard" and p["zone"] == zone)
            if std_in_zone:
                # skip batching in this zone to preserve priority for Standard orders
                continue

            olist = sorted(olist, key=lambda x: x["arrive_min"])
            batch = []
            for o in olist:
                waited = now - o["arrive_min"]
                # normal eco window and count-based batching; opportunistic 2-min rule allowed only when no Standards in zone
                if waited >= ASSUMPTIONS["batching_window_mins"] or len(olist) >= 3:
                    batch.append(o)
                    if len(batch) >= 4: break
                else:
                    if waited >= 2.0:
                        batch.append(o)
                        if len(batch) >= 4: break
            if not batch: continue
            zone_riders = [r for r in riders if r["zone"] == zone]
            if not zone_riders:
                continue
            best = None; best_finish = float("inf")
            for r in zone_riders:
                start = max(now, r["next_free_min"])
                est_travel_total = mean_travel + (len(batch)-1) * mean_travel * (1 - ASSUMPTIONS["batch_travel_savings"])
                finish = start + est_travel_total + len(batch) * ASSUMPTIONS["handling_time_min"]
                if finish < best_finish:
                    best_finish = finish; best = r
            if best is None:
                continue
            start_time = max(now, best["next_free_min"])
            travel_first = sample_travel()
            travels = [travel_first] + [max(1.0, random.gauss(mean_travel*(1-ASSUMPTIONS["batch_travel_savings"]), mean_travel*0.2)) for _ in range(len(batch)-1)]
            current = start_time
            for idx, o in enumerate(batch):
                travel = travels[idx]
                finish_time = current + travel + ASSUMPTIONS["handling_time_min"]
                best["busy_minutes"] += (finish_time - current)
                best["assignments"] += 1
                best["current_task"] = {"order_id": o["id"], "tier": o["tier"], "start_min": start_time, "end_min": finish_time, "zone": zone}
                completed.append({
                    "order_id": o["id"], "arrive_min": o["arrive_min"], "start_min": start_time,
                    "finish_min": finish_time, "tier": o["tier"], "promise_min": SLA_PROMISE[o["tier"]],
                    "delivery_min": finish_time - o["arrive_min"], "met_sla": (finish_time - o["arrive_min"])<=SLA_PROMISE[o["tier"]],
                    "assigned_rider": best["id"], "relocation_min": 0.0
                })
                eco_assigned.add(o["id"])
                current = finish_time
            best["next_free_min"] = current
            best["zone"] = zone
        pending = [p for p in pending if p["id"] not in eco_assigned]

        now += FIXED_DISPATCH_EPOCH_MIN
        epoch += 1
        if epoch > 50000:
            break

    df = pd.DataFrame(completed)
    total_orders_sim = len(df)
    if total_orders_sim > 0:
        sla_by_tier = df.groupby("tier").agg(
            orders=("order_id","count"),
            sla_met_pct=("met_sla", lambda x: 100*sum(x)/len(x)),
            avg_delivery_min=("delivery_min","mean")
        ).reset_index()
    else:
        sla_by_tier = pd.DataFrame(columns=["tier","orders","sla_met_pct","avg_delivery_min"])
    overall_sla = 100*df["met_sla"].mean() if total_orders_sim>0 else None
    avg_delivery = df["delivery_min"].mean() if total_orders_sim>0 else None
    utilizations = [r["busy_minutes"]/horizon_minutes for r in riders]
    avg_util = 100* (sum(utilizations)/len(utilizations)) if utilizations else 0.0
    opd = total_orders_sim / ASSUMPTIONS["simulation_horizon_hrs"]
    oph_per_active_rider = opd / (sum(1 for r in riders if r["busy_minutes"]>0) or 1)
    total_rider_hours = sum(r["busy_minutes"] for r in riders)/60.0
    labour_cost = total_rider_hours * ASSUMPTIONS["rider_cost_per_hr"]
    cost_per_order = labour_cost / (total_orders_sim or 1)

    result = {
        "city": city, "orders_simulated": total_orders_sim, "overall_sla_pct": overall_sla,
        "avg_delivery_min": avg_delivery, "avg_util_pct": avg_util, "orders_per_hour": opd,
        "oph_per_active_rider": oph_per_active_rider, "cost_per_order": cost_per_order,
        "sla_breakdown": sla_by_tier, "orders_df": df
    }
    return result

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Flipkart Quick-Commerce: Rider Deployment Simulator")
st.markdown("Rolling-horizon dispatch + strict priority (Express > Standard > EcoSaver). Hybrid SLA relabels some Express orders to slower tiers but does NOT change dispatch priority.")

with st.sidebar:
    st.header("Simulation inputs")
    city = st.selectbox("City", options=["Delhi-NCR","Pune"], index=0)
    riders = st.slider("Total riders", min_value=100, max_value=8000, value=3500, step=50)
    demand = st.slider("Demand multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    st.markdown("SLA Mix (percent). Values will be normalized automatically.")
    col1, col2, col3 = st.columns(3)
    with col1:
        exp_pct = st.number_input("Express %", min_value=0, max_value=100, value=60)
    with col2:
        std_pct = st.number_input("Standard %", min_value=0, max_value=100, value=30)
    with col3:
        eco_pct = st.number_input("EcoSaver %", min_value=0, max_value=100, value=10)
    fleet_sharing = st.checkbox("Fleet sharing (dynamic) — not used for zone-bound dispatch", value=True)

    st.markdown("---")
    hybrid_enabled = st.checkbox("Enable Hybrid SLA (shift some Express orders to slower tiers)", value=False)
    if hybrid_enabled:
        shift_pct = st.slider("Shift % of Express → slower tiers", min_value=0, max_value=100, value=20)
        split_to_std_pct = st.slider("Of shifted: % → Standard (remainder → EcoSaver)", min_value=0, max_value=100, value=70)
    else:
        shift_pct = 20
        split_to_std_pct = 70

    # show before / after split immediately for clarity (preview only)
    total_pct = (exp_pct + std_pct + eco_pct) or 1
    raw_sla_mix = {
        "Express": max(0.0, exp_pct/total_pct),
        "Standard": max(0.0, std_pct/total_pct),
        "EcoSaver": max(0.0, eco_pct/total_pct)
    }

    st.markdown("**Order split — BEFORE hybrid**")
    st.write(f"Express: {raw_sla_mix['Express']*100:.1f}%, Standard: {raw_sla_mix['Standard']*100:.1f}%, EcoSaver: {raw_sla_mix['EcoSaver']*100:.1f}%")

    if hybrid_enabled and raw_sla_mix["Express"] > 0:
        shift_frac_preview = shift_pct / 100.0
        s_to_std_preview = split_to_std_pct / 100.0
        s_to_eco_preview = 1.0 - s_to_std_preview
        express_share = raw_sla_mix["Express"]
        shifted = express_share * shift_frac_preview
        adj_express = max(0.0, express_share - shifted)
        adj_standard = raw_sla_mix["Standard"] + shifted * s_to_std_preview
        adj_ecosaver = raw_sla_mix["EcoSaver"] + shifted * s_to_eco_preview
        total_adj = adj_express + adj_standard + adj_ecosaver or 1.0
        sla_mix_preview = {"Express": adj_express/total_adj, "Standard": adj_standard/total_adj, "EcoSaver": adj_ecosaver/total_adj}
        st.markdown("**Order split — AFTER hybrid (preview)**")
        st.write(f"Express: {sla_mix_preview['Express']*100:.1f}%, Standard: {sla_mix_preview['Standard']*100:.1f}%, EcoSaver: {sla_mix_preview['EcoSaver']*100:.1f}%")
    else:
        st.markdown("**Order split — AFTER hybrid (preview)**")
        st.write("Hybrid disabled or no Express share — preview equals BEFORE split.")

    st.markdown("---")
    run_btn = st.button("Run simulation")

# prepare final sla_mix to feed simulator (apply hybrid preview)
total_pct = (exp_pct + std_pct + eco_pct) or 1
raw_sla_mix = {
    "Express": max(0.0, exp_pct/total_pct),
    "Standard": max(0.0, std_pct/total_pct),
    "EcoSaver": max(0.0, eco_pct/total_pct)
}

if hybrid_enabled and raw_sla_mix["Express"] > 0:
    shift_frac_val = shift_pct / 100.0
    s_to_std_val = split_to_std_pct / 100.0
    s_to_eco_val = 1.0 - s_to_std_val
    express_share = raw_sla_mix["Express"]
    shifted_share = express_share * shift_frac_val
    adj_express = max(0.0, express_share - shifted_share)
    adj_standard = raw_sla_mix["Standard"] + shifted_share * s_to_std_val
    adj_ecosaver = raw_sla_mix["EcoSaver"] + shifted_share * s_to_eco_val
    total_adj = adj_express + adj_standard + adj_ecosaver or 1.0
    sla_mix = {"Express": adj_express/total_adj, "Standard": adj_standard/total_adj, "EcoSaver": adj_ecosaver/total_adj}
else:
    sla_mix = raw_sla_mix
    shift_frac_val = 0.0
    s_to_std_val = 0.0

if run_btn:
    start = time.time()
    with st.spinner("Running simulation..."):
        res = simulate_dispatch_streamlit(
            city=city,
            total_riders=riders,
            sla_mix=sla_mix,
            fleet_sharing=fleet_sharing,
            demand_multiplier=demand,
            hybrid_enabled=hybrid_enabled,
            shift_frac=shift_frac_val,
            s_to_std=(s_to_std_val if hybrid_enabled else 0.0),
            random_seed=FIXED_RANDOM_SEED
        )
    duration = time.time() - start
    st.success(f"Simulation finished in {duration:.2f}s — Orders simulated: {res['orders_simulated']:,}")
    
    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Overall SLA met (%)", f"{res['overall_sla_pct']:.2f}" if res['overall_sla_pct'] is not None else "N/A")
    kpi2.metric("Avg delivery (min)", f"{res['avg_delivery_min']:.2f}" if res['avg_delivery_min'] is not None else "N/A")
    kpi3.metric("Avg rider util (%)", f"{res['avg_util_pct']:.2f}")
    kpi4.metric("Cost per order (₹)", f"{res['cost_per_order']:.2f}")
    
    # SLA table
    st.subheader("SLA breakdown by tier")
    st.dataframe(res["sla_breakdown"])
    
    # Plots: hourly throughput
    df = res["orders_df"]
    if not df.empty:
        df["finish_hr"] = (df["finish_min"]//60).astype(int)
        hourly = df.groupby("finish_hr").size().reindex(range(24), fill_value=0)
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        ax.plot(hourly.index, hourly.values, marker='o')
        ax.set_xlabel("Hour of day"); ax.set_ylabel("Orders finished"); ax.set_title("Hourly throughput")
        st.pyplot(fig)
        
        # Download CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()
        st.download_button("Download order-level CSV", data=csv_bytes, file_name=f"flipkart_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    else:
        st.info("No orders in this run (check demand multiplier / city).")
else:
    st.info("Set inputs in the sidebar and click **Run simulation**.")
